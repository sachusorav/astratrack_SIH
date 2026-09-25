"""
ASTRATRACK — 3D Aerospace Mission Control Ground Station Dashboard

Combines high-fidelity 3D vector visual rendering with Dear PyGui dashboard:
1. TOP BAR: Branding, Status Badge, FPS/Latency, Mission Clock, Quick Triggers (Judge Demo, PDF Export)
2. MAIN VIEW: Live 3D Viewport with orbit mouse controls, view mode selector, and 3D visual toggles
3. TELEMETRY PANEL: Real-time 3D & sensor telemetry (Pos, Vel, Range, SNR, Pan/Tilt, Boresight Error, Latency)
4. CONTROL PANEL: Scenario selector (all 9 3D presets), Target speed, Disturbance preset, Pipeline toggles, PID gains
5. PERFORMANCE PANEL: Real-time rolling plots (Error, Confidence, FPS) & KPI Scorecard
"""

import os
import sys
import time
import math
import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from typing import Optional

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from simulator.simulation import Simulation3D
from simulator.scenarios import list_scenarios, get_scenario_by_id_or_key, SCENARIO_REGISTRY
from simulator.renderer import ViewMode
from control.camera_controller import ControllerMode
from metrics.collector import MetricsCollector
from metrics.evaluator import PerformanceEvaluator
from metrics.report import PerformanceReport
from metrics.exporters import ReportExporter
from demo.director import JudgeDemoDirector, DemoPhase
from demo.timeline_overlay import draw_demo_timeline


class Dashboard3D:
    """Aerospace Mission Control 3D Ground Station Dashboard."""

    def __init__(self, sim: Simulation3D):
        self.sim = sim
        self.running = False
        self.in_judge_demo = False

        # Demo & Metrics
        self.judge_demo = JudgeDemoDirector()
        self.metrics_collector = MetricsCollector(deg_per_pixel=sim.camera.config.fov_deg / 640.0)

        # Plot histories
        self._max_history = 200
        self._time_hist = []
        self._error_hist = []
        self._conf_hist = []
        self._fps_hist = []
        self._frame_count = 0

        # Mouse orbit state for DPG viewport image
        self._mouse_drag_active = False
        self._last_mouse_pos = (0, 0)

    def setup(self):
        """Build Dear PyGui context, dark theme, texture registry, and UI layout."""
        dpg.create_context()

        # Aerospace Dark Theme (Curated Control Station Palette)
        with dpg.theme() as dark_theme:
            with dpg.theme_component(dpg.mvAll):
                # Backgrounds & Window Panels
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (11, 14, 20, 255))         # Cyber Navy #0B0E14
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (19, 25, 36, 255))          # Slate Navy #131924
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (17, 23, 34, 255))          # Floating Popup Slate
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (15, 20, 30, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (26, 36, 56, 255))

                # Interactive Inputs & Frames
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (24, 32, 48, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (34, 46, 68, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (42, 58, 86, 255))

                # Buttons & Sliders
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (0, 229, 255, 255))        # Glowing Cyan #00E5FF
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (0, 176, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (26, 40, 64, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (38, 60, 98, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 229, 255, 255))

                # Text & Elevation Borders
                dpg.add_theme_color(dpg.mvThemeCol_Text, (230, 238, 250, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (35, 48, 70, 255))            # Elevation Border
                dpg.add_theme_color(dpg.mvThemeCol_Separator, (35, 48, 70, 255))

                # Roundings & Item Padding
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)

        dpg.bind_theme(dark_theme)

        # Texture Registry for 3D Viewport Frame
        blank = np.zeros((self.sim.height, self.sim.width, 4), dtype=np.float32)
        with dpg.texture_registry():
            dpg.add_raw_texture(
                width=self.sim.width,
                height=self.sim.height,
                default_value=blank.flatten(),
                format=dpg.mvFormat_Float_rgba,
                tag="viewport_texture",
            )

        # Keyboard & Mouse Event Handlers
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=lambda: self.sim.toggle_pause())
            dpg.add_key_press_handler(dpg.mvKey_R, callback=lambda: self.sim.reset())
            dpg.add_key_press_handler(dpg.mvKey_K, callback=lambda: self._on_toggle_kalman())
            dpg.add_key_press_handler(dpg.mvKey_P, callback=lambda: self._on_toggle_prediction())
            dpg.add_key_press_handler(dpg.mvKey_D, callback=lambda: self._on_toggle_detector())
            dpg.add_key_press_handler(dpg.mvKey_X, callback=lambda: self._on_toggle_detector())
            dpg.add_key_press_handler(dpg.mvKey_C, callback=lambda: self._on_toggle_control_mode())
            dpg.add_key_press_handler(dpg.mvKey_V, callback=lambda: self.sim.renderer.cycle_view_mode())
            dpg.add_key_press_handler(dpg.mvKey_F5, callback=lambda: self._start_judge_demo())

            # Instant scenario selection 1-9
            for idx in range(1, 10):
                key_code = getattr(dpg, f"mvKey_{idx}")
                dpg.add_key_press_handler(key_code, callback=lambda s, a, u=str(idx): self._on_key_select_scenario(u))

            # Arrow keys & WASD live control
            dpg.add_key_press_handler(dpg.mvKey_Up, callback=lambda: self._adjust_target_speed(+10.0))
            dpg.add_key_press_handler(dpg.mvKey_Down, callback=lambda: self._adjust_target_speed(-10.0))
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=lambda: self.sim.renderer.rotate_orbit(delta_az=-5.0, delta_el=0.0))
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=lambda: self.sim.renderer.rotate_orbit(delta_az=+5.0, delta_el=0.0))
            dpg.add_key_press_handler(dpg.mvKey_W, callback=lambda: self.sim.renderer.rotate_orbit(delta_az=0.0, delta_el=+5.0))
            dpg.add_key_press_handler(dpg.mvKey_S, callback=lambda: self.sim.renderer.rotate_orbit(delta_az=0.0, delta_el=-5.0))

        # Main Window
        with dpg.window(tag="main_window", no_title_bar=True, no_resize=True, no_move=True):
            # ZONE 1: TOP BAR
            self._create_top_bar()

            # ZONES 2, 3, 4 (Middle Horizontal Layout)
            with dpg.group(horizontal=True):
                # ZONE 2: 3D VIEWPORT
                self._create_main_view()
                # ZONE 3: TELEMETRY PANEL
                self._create_telemetry_panel()
                # ZONE 4: CONTROL PANEL
                self._create_control_panel()

            dpg.add_spacer(height=4)

            # ZONE 5: PERFORMANCE PANEL & PLOTS
            self._create_performance_panel()

        # Create DPG Viewport
        dpg.create_viewport(
            title="ASTRATRACK - 3D Aerospace FSOC Tracking Simulator (SIH 2026)",
            width=1440, height=920, min_width=1280, min_height=800
        )
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_window", True)

        # Register Proportional Viewport Resize Callback
        dpg.set_viewport_resize_callback(self._on_viewport_resize)
        dpg.show_viewport()
        
        # Trigger initial layout calculation to fit current window resolution
        self._on_viewport_resize()

    # ------------------------------------------------------------------ #
    # ZONE 1: TOP BAR (RESPONSIVE & OVERFLOW SAFE)
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # ZONE 1: TOP BAR (RESPONSIVE & OVERFLOW SAFE)
    # ------------------------------------------------------------------ #
    def _create_top_bar(self):
        with dpg.child_window(tag="win_top_bar", height=56, border=True):
            with dpg.group(horizontal=True):
                # Left branding section
                dpg.add_text("ASTRATRACK", color=(0, 230, 118))
                dpg.add_text("| 3D FSOC Aerospace Simulator", color=(140, 150, 170))

                dpg.add_spacer(width=10)
                dpg.add_text("STATUS:")
                dpg.add_button(label="[ SEARCHING ]", tag="top_status_badge", small=True)

                dpg.add_spacer(width=10)
                dpg.add_text("RATE:")
                dpg.add_text("60.0 FPS", tag="top_fps_text", color=(0, 230, 118))

                dpg.add_spacer(width=8)
                dpg.add_text("LATENCY:")
                dpg.add_text("08.5 ms", tag="top_latency_text", color=(0, 176, 255))

                dpg.add_spacer(width=8)
                dpg.add_text("CLOCK:")
                dpg.add_text("T+00:00.0", tag="top_clock_text", color=(255, 214, 0))

                # Right trigger controls (Compact & Responsive)
                dpg.add_spacer(width=15)
                dpg.add_button(label="START JUDGE DEMO (90s)", callback=self._start_judge_demo, tag="btn_judge_demo", small=True)
                dpg.add_button(label="Export Audit PDF", callback=self._export_pdf_report, small=True)
                dpg.add_button(label="Pause (Space)", callback=self.sim.toggle_pause, small=True)
                dpg.add_button(label="Reset (R)", callback=self.sim.reset, small=True)

    # ------------------------------------------------------------------ #
    # ZONE 2: MAIN VIEW (3D VIEWPORT)
    # ------------------------------------------------------------------ #
    def _create_main_view(self):
        with dpg.child_window(tag="win_zone2", border=True):
            with dpg.group(horizontal=True):
                dpg.add_text("ZONE 2: 3D OPTICAL VIEWPORT", color=(0, 176, 255))
                dpg.add_spacer(width=15)
                dpg.add_text("View:", color=(140, 150, 170))
                views = ["ORBIT", "GROUND STATION", "TARGET CHASE", "SENSOR FULL"]
                dpg.add_combo(views, default_value="ORBIT", callback=self._on_view_mode_select, width=125, tag="combo_view_mode")

            # 3D texture viewport frame (Dynamically scaled by _on_viewport_resize)
            dpg.add_image("viewport_texture", tag="viewport_image")

            # Render feature toggles
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Trail", default_value=True, callback=self._on_toggle_render_feature, user_data="trail")
                dpg.add_checkbox(label="FOV", default_value=True, callback=self._on_toggle_render_feature, user_data="fov")
                dpg.add_checkbox(label="LOS", default_value=True, callback=self._on_toggle_render_feature, user_data="los")
                dpg.add_checkbox(label="Grid", default_value=True, callback=self._on_toggle_render_feature, user_data="grid")
                dpg.add_checkbox(label="Axes", default_value=True, callback=self._on_toggle_render_feature, user_data="axes")
                dpg.add_checkbox(label="HUD", default_value=True, callback=self._on_toggle_render_feature, user_data="hud")
                dpg.add_checkbox(label="PiP", default_value=True, callback=self._on_toggle_render_feature, user_data="pip")

    # ------------------------------------------------------------------ #
    # ZONE 3: TELEMETRY PANEL
    # ------------------------------------------------------------------ #
    def _create_telemetry_panel(self):
        with dpg.child_window(tag="win_zone3", border=True):
            dpg.add_text("ZONE 3: REAL-TIME 3D TELEMETRY", color=(0, 176, 255))
            dpg.add_separator()

            with dpg.table(header_row=False):
                dpg.add_table_column(width_fixed=True, init_width_or_weight=150)
                dpg.add_table_column(width_fixed=True, init_width_or_weight=170)

                with dpg.table_row():
                    dpg.add_text("Target Position (3D):")
                    dpg.add_text("[0.0, 0.0, 0.0] m", tag="telem_pos_3d", color=(220, 230, 245))

                with dpg.table_row():
                    dpg.add_text("Target Velocity:")
                    dpg.add_text("+00.0 m/s", tag="telem_vel_3d", color=(220, 230, 245))

                with dpg.table_row():
                    dpg.add_text("Slant Range / SNR:")
                    dpg.add_text("0.0 m (00.0 dB)", tag="telem_range_snr", color=(220, 230, 245))

                with dpg.table_row():
                    dpg.add_text("Detection Confidence:")
                    dpg.add_text("0.95 (HIGH)", tag="telem_conf", color=(0, 230, 118))

                with dpg.table_row():
                    dpg.add_text("Gimbal Pan Angle:")
                    dpg.add_text("+00.0°", tag="telem_pan", color=(220, 230, 245))

                with dpg.table_row():
                    dpg.add_text("Gimbal Tilt Angle:")
                    dpg.add_text("+00.0°", tag="telem_tilt", color=(220, 230, 245))

                with dpg.table_row():
                    dpg.add_text("Boresight Error:")
                    dpg.add_text("0.0 px (0.00°)", tag="telem_err", color=(255, 82, 82))

                with dpg.table_row():
                    dpg.add_text("Control Command:")
                    dpg.add_text("Pan: 0.0, Tilt: 0.0", tag="telem_ctrl", color=(0, 176, 255))

                with dpg.table_row():
                    dpg.add_text("Pipeline Latency:")
                    dpg.add_text("08.5 ms", tag="telem_lat", color=(0, 176, 255))

            dpg.add_spacer(height=8)
            dpg.add_text("RE-ACQUISITION PERFORMANCE", color=(255, 171, 0))
            dpg.add_separator()
            dpg.add_text("Losses: 0  |  Recoveries: 0 (100%)", tag="telem_losses")
            dpg.add_text("Last Recovery: 0.00 s  |  Mean: 0.00 s", tag="telem_rec_time")

    # ------------------------------------------------------------------ #
    # ZONE 4: CONTROL PANEL
    # ------------------------------------------------------------------ #
    def _create_control_panel(self):
        with dpg.child_window(tag="win_zone4", border=True):
            dpg.add_text("ZONE 4: MISSION CONTROL PANEL", color=(0, 176, 255))
            dpg.add_separator()

            # Scenario selection (lists all 9 scenarios)
            scenarios = [f"{s.id} - {s.name}" for s in list_scenarios()]
            dpg.add_text("3D Scenario Preset:")
            dpg.add_combo(scenarios, default_value=scenarios[1], callback=self._on_scenario_select, tag="combo_scenario", height_mode=dpg.mvComboHeight_Large)

            # Target speed
            dpg.add_text("Target Speed (m/s):")
            dpg.add_slider_float(min_value=20.0, max_value=350.0, default_value=120.0,
                                 callback=self._on_speed_change, tag="slider_target_speed")

            # Disturbance preset
            dpg.add_text("Disturbance Intensity:")
            dpg.add_combo(["NORMAL", "LIGHT DISTURBANCE", "MODERATE DISTURBANCE", "SEVERE DISTURBANCE", "EXTREME STRESS TEST"],
                          default_value="NORMAL", callback=self._on_disturbance_preset_change, tag="combo_disturbance", height_mode=dpg.mvComboHeight_Regular)

            dpg.add_spacer(height=6)
            dpg.add_text("PIPELINE TOGGLES (SIH DEMO SWITCHES):", color=(0, 230, 118))
            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_button(label="Detector: Classical (D)", callback=self._on_toggle_detector, tag="btn_toggle_det")
                dpg.add_button(label="Kalman: ON (K)", callback=self._on_toggle_kalman, tag="btn_toggle_kalman")

            with dpg.group(horizontal=True):
                dpg.add_button(label="Prediction: ON (P)", callback=self._on_toggle_prediction, tag="btn_toggle_pred")
                dpg.add_button(label="Control: PID (C)", callback=self._on_toggle_control_mode, tag="btn_toggle_ctrl")

            dpg.add_spacer(height=6)
            dpg.add_text("PID GAIN TUNING:")
            dpg.add_slider_float(label="Kp", min_value=0.05, max_value=1.5, default_value=0.35,
                                 callback=self._on_pid_change, tag="slider_kp")
            dpg.add_slider_float(label="Ki", min_value=0.00, max_value=0.1, default_value=0.008,
                                 callback=self._on_pid_change, tag="slider_ki")
            dpg.add_slider_float(label="Kd", min_value=0.00, max_value=0.5, default_value=0.08,
                                 callback=self._on_pid_change, tag="slider_kd")

            dpg.add_spacer(height=6)
            dpg.add_text("LIVE KEYBOARD CONTROLS GUIDE:", color=(255, 214, 0))
            dpg.add_separator()
            dpg.add_text("[1-9] Select Scenario 1 to 9 | [X/D] AI/Classical Det", color=(180, 190, 210), tag="guide_text_1")
            dpg.add_text("[K] Kalman | [P] Prediction | [C] Control | [V] Views", color=(180, 190, 210), tag="guide_text_2")
            dpg.add_text("[UP/DOWN] Target Speed +/-10 m/s | [Space] Pause | [R] Reset", color=(180, 190, 210), tag="guide_text_3")
            dpg.add_text("[LEFT/RIGHT/W/S] Live Orbit Camera Rotation", color=(180, 190, 210), tag="guide_text_4")

    # ------------------------------------------------------------------ #
    # ZONE 5: PERFORMANCE PANEL & LIVE PLOTS
    # ------------------------------------------------------------------ #
    def _create_performance_panel(self):
        with dpg.child_window(tag="win_zone5", border=True):
            with dpg.group(horizontal=True):
                dpg.add_text("ZONE 5: PERFORMANCE ANALYTICS & LIVE PLOTS", color=(0, 176, 255))
                dpg.add_spacer(width=20)
                dpg.add_text("Acq Time:", color=(140, 150, 170))
                dpg.add_text("0.42 s", tag="perf_acq_time", color=(0, 230, 118))
                dpg.add_spacer(width=15)
                dpg.add_text("Lock Retention:", color=(140, 150, 170))
                dpg.add_text("95.2%", tag="perf_lock_ret", color=(0, 230, 118))
                dpg.add_spacer(width=15)
                dpg.add_text("RMS Error:", color=(140, 150, 170))
                dpg.add_text("0.45°", tag="perf_rms_err", color=(0, 230, 118))

            with dpg.group(horizontal=True):
                # Plot 1: Boresight Tracking Error
                with dpg.plot(label="Boresight Tracking Error (deg)", tag="plot_err"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="plot_err_xaxis")
                    with dpg.plot_axis(dpg.mvYAxis, label="Degrees", tag="plot_err_yaxis"):
                        dpg.add_line_series([], [], label="Error (deg)", tag="plot_err_series")

                # Plot 2: Detection Confidence
                with dpg.plot(label="Detector Confidence", tag="plot_conf"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="plot_conf_xaxis")
                    with dpg.plot_axis(dpg.mvYAxis, label="Score [0-1]", tag="plot_conf_yaxis"):
                        dpg.add_line_series([], [], label="Confidence", tag="plot_conf_series")

                # Plot 3: Frame Rate
                with dpg.plot(label="System Frame Rate (FPS)", tag="plot_fps"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="plot_fps_xaxis")
                    with dpg.plot_axis(dpg.mvYAxis, label="FPS", tag="plot_fps_yaxis"):
                        dpg.add_line_series([], [], label="Rate", tag="plot_fps_series")

    # ------------------------------------------------------------------ #
    # Dynamic Viewport Resize Handler (Proportional Responsiveness)
    # ------------------------------------------------------------------ #
    def _on_viewport_resize(self, sender=None, app_data=None):
        """Recalculate layout geometry dynamically whenever the window is resized."""
        if not dpg.is_dearpygui_running():
            return

        vw = dpg.get_viewport_width()
        vh = dpg.get_viewport_height()

        if vw < 300 or vh < 300:
            return

        # Usable width and height
        aw = max(1150, vw - 35)
        avail_h = max(600, vh - 90)

        top_height = max(420, int(avail_h * 0.62))
        bottom_height = max(220, avail_h - top_height - 15)

        # Proportional top column widths: 48% (Zone 2), 26% (Zone 3), 26% (Zone 4)
        z2_width = int(aw * 0.48)
        z3_width = int(aw * 0.26)
        z4_width = aw - z2_width - z3_width - 16

        # Zone 1 (Top Bar)
        if dpg.does_item_exist("win_top_bar"):
            dpg.configure_item("win_top_bar", width=aw)

        # Zone 2, Zone 3, Zone 4 (Equal Row Height!)
        if dpg.does_item_exist("win_zone2"):
            dpg.configure_item("win_zone2", width=z2_width, height=top_height)
        if dpg.does_item_exist("win_zone3"):
            dpg.configure_item("win_zone3", width=z3_width, height=top_height)
        if dpg.does_item_exist("win_zone4"):
            dpg.configure_item("win_zone4", width=z4_width, height=top_height)

        # Rescale Zone 2 Viewport Texture Frame to fill available vertical and horizontal space
        if dpg.does_item_exist("viewport_image"):
            img_w = max(400, z2_width - 25)
            img_h = max(280, top_height - 75)
            dpg.configure_item("viewport_image", width=img_w, height=img_h)

        # Re-calculate Zone 4 text wrapping & control widget widths dynamically
        wrap_w = max(220, z4_width - 30)
        for tag in ["guide_text_1", "guide_text_2", "guide_text_3", "guide_text_4"]:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, wrap=wrap_w)

        w_ctrl = max(180, z4_width - 80)
        for tag in ["combo_scenario", "slider_target_speed", "combo_disturbance", "slider_kp", "slider_ki", "slider_kd"]:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, width=w_ctrl)

        w_btn = max(90, int((z4_width - 35) / 2))
        for tag in ["btn_toggle_det", "btn_toggle_kalman", "btn_toggle_pred", "btn_toggle_ctrl"]:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, width=w_btn)

        # Zone 5 Performance Panel & 3 Live Plots
        if dpg.does_item_exist("win_zone5"):
            dpg.configure_item("win_zone5", width=aw, height=bottom_height)

        plot_w = max(280, int((aw - 50) / 3))
        plot_h = max(140, bottom_height - 55)
        for tag in ["plot_err", "plot_conf", "plot_fps"]:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, width=plot_w, height=plot_h)

    # ------------------------------------------------------------------ #
    # Handlers & Callbacks
    # ------------------------------------------------------------------ #

    def _start_judge_demo(self):
        self.in_judge_demo = True
        self.judge_demo.reset()

    def _on_scenario_select(self, sender, app_data):
        scen_key = str(app_data).split("-")[0].strip()
        self.sim.load_scenario(scen_key)

    def _on_speed_change(self, sender, app_data):
        self.sim.set_target_speed(float(app_data))

    def _on_key_select_scenario(self, key: str):
        """Instant scenario switch via keyboard number keys (1-9)."""
        self.sim.load_scenario(key)
        scen_list = list_scenarios()
        for s in scen_list:
            if s.id == key or getattr(s, "key", "") == key:
                dpg.set_value("combo_scenario", f"{s.id} - {s.name}")
                break

    def _adjust_target_speed(self, delta: float):
        """Adjust target speed up/down via keyboard arrow keys."""
        if self.sim.target is not None:
            cur_speed = getattr(self.sim.target.config, "speed", 100.0)
            new_speed = max(10.0, min(500.0, cur_speed + delta))
            self.sim.set_target_speed(new_speed)
            dpg.set_value("slider_target_speed", new_speed)

    def _on_disturbance_preset_change(self, sender, app_data):
        self.sim.set_disturbance_level(str(app_data))

    def _on_view_mode_select(self, sender, app_data):
        vm_map = {
            "ORBIT": ViewMode.ORBIT,
            "GROUND STATION": ViewMode.GROUND_STATION,
            "TARGET CHASE": ViewMode.TARGET_CHASE,
            "SENSOR FULL": ViewMode.SENSOR_FULL,
        }
        if app_data in vm_map:
            self.sim.renderer.view_mode = vm_map[app_data]

    def _on_toggle_render_feature(self, sender, app_data, user_data):
        if user_data == "trail":
            self.sim.renderer.show_trajectory_trail = app_data
        elif user_data == "fov":
            self.sim.renderer.show_camera_fov = app_data
        elif user_data == "los":
            self.sim.renderer.show_los_beam = app_data
        elif user_data == "grid":
            self.sim.renderer.show_ground_grid = app_data
        elif user_data == "axes":
            self.sim.renderer.show_coordinate_axes = app_data
        elif user_data == "hud":
            self.sim.renderer.show_telemetry_hud = app_data
        elif user_data == "pip":
            self.sim.renderer.show_pip_sensor = app_data

    def _on_toggle_detector(self):
        cur = getattr(self.sim, "detector_type", "classical")
        new_det = "ai" if cur == "classical" else "classical"
        self.sim.set_detector_type(new_det)
        lbl = f"Detector: {'AI YOLO' if new_det == 'ai' else 'Classical'} (D)"
        dpg.set_item_label("btn_toggle_det", lbl)

    def _on_toggle_kalman(self):
        pipe = self.sim.ensure_pipeline()
        enabled = not pipe.kalman.prediction_enabled
        self.sim.set_kalman_enabled(enabled)
        dpg.set_item_label("btn_toggle_kalman", f"Kalman: {'ON' if enabled else 'OFF'} (K)")

    def _on_toggle_prediction(self):
        pipe = self.sim.ensure_pipeline()
        enabled = not pipe.kalman.prediction_enabled
        self.sim.set_prediction_enabled(enabled)
        dpg.set_item_label("btn_toggle_pred", f"Prediction: {'ON' if enabled else 'OFF'} (P)")

    def _on_toggle_control_mode(self):
        pipe = self.sim.ensure_pipeline()
        cur_mode = pipe.controller.mode
        if cur_mode == ControllerMode.PID_CONTROL:
            new_mode = ControllerMode.P_CONTROL
            lbl = "Control: P (C)"
        elif cur_mode == ControllerMode.P_CONTROL:
            new_mode = ControllerMode.DIRECT
            lbl = "Control: Direct (C)"
        else:
            new_mode = ControllerMode.PID_CONTROL
            lbl = "Control: PID (C)"
        self.sim.set_controller_mode(new_mode)
        dpg.set_item_label("btn_toggle_ctrl", lbl)

    def _on_pid_change(self, sender=None, app_data=None):
        kp = dpg.get_value("slider_kp")
        ki = dpg.get_value("slider_ki")
        kd = dpg.get_value("slider_kd")
        self.sim.set_pid_gains(kp, ki, kd)

    def _check_viewport_mouse_drag(self):
        """Poll mouse drag over viewport image for 3D orbit camera manipulation."""
        if dpg.is_item_hovered("viewport_image") and dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            mx, my = dpg.get_mouse_pos(local=False)
            if self._mouse_drag_active:
                dx = mx - self._last_mouse_pos[0]
                dy = my - self._last_mouse_pos[1]
                self.sim.renderer.rotate_orbit(delta_az=dx * 0.4, delta_el=-dy * 0.4)
            self._last_mouse_pos = (mx, my)
            self._mouse_drag_active = True
        elif not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            self._mouse_drag_active = False

    def _export_pdf_report(self):
        evaluator = PerformanceEvaluator()
        telem = self.sim.get_telemetry()
        status, checklist = evaluator.evaluate(telem)
        run_id = f"RUN_3D_{int(time.time())}"
        report = PerformanceReport(
            run_id=run_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            scenario=self.sim.scenario.name,
            configuration={"width": self.sim.width, "height": self.sim.height},
            disturbance_settings={"preset": "NORMAL"},
            detector=getattr(self.sim, "detector_type", "classical").upper(),
            controller="PID",
            filter_settings={"enabled": True},
            metrics=telem,
            pass_fail_status=status,
            criteria_checklist=checklist
        )
        reports_dir = os.path.join(WORKSPACE_ROOT, "outputs", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        pdf_path = os.path.join(reports_dir, f"Performance_Audit_{run_id}.pdf")
        out_path = ReportExporter.export_pdf(report, pdf_path)

        json_path = os.path.join(reports_dir, f"Performance_Audit_{run_id}.json")
        ReportExporter.export_json(report, json_path)

        final_path = out_path or json_path
        print(f"[INFO] 3D Performance Audit exported to: {final_path}")

        # Modal Notification Popup in DPG
        if dpg.does_item_exist("modal_export_notice"):
            dpg.delete_item("modal_export_notice")

        with dpg.window(label="Audit Report Exported", modal=True, show=True, tag="modal_export_notice", width=520, height=180, pos=(450, 320)):
            dpg.add_text("★ Performance Audit Report Successfully Exported!", color=(0, 230, 118))
            dpg.add_separator()
            dpg.add_text(f"File Path:\n{final_path}", wrap=490, color=(220, 230, 245))
            dpg.add_spacer(height=12)
            dpg.add_button(label="   OK   ", callback=lambda: dpg.delete_item("modal_export_notice"))

    # ------------------------------------------------------------------ #
    # Tick & GUI Loop
    # ------------------------------------------------------------------ #

    def _tick(self):
        """Step 3D simulation, render frame, and update DPG UI displays."""
        t0 = time.perf_counter()

        # Handle viewport mouse orbit drag
        self._check_viewport_mouse_drag()

        # Step simulation physics & perception
        self.sim.step()

        # Render 3D frame
        frame_bgr = self.sim.render()

        # Handle Judge Demo overlay if active
        if self.in_judge_demo:
            phase, is_done = self.judge_demo.step(self.sim.fixed_dt)
            frame_bgr = draw_demo_timeline(frame_bgr, phase, self.judge_demo.sim_time, 90.0)
            dpg.set_item_label("top_status_badge", f"[ {phase.value} ]")

        # Convert BGR uint8 to Float32 RGBA for DPG texture
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
        dpg.set_value("viewport_texture", rgba.flatten())

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Update Telemetry & UI
        self._update_telemetry_ui(latency_ms)

    def _update_telemetry_ui(self, latency_ms: float):
        telem = self.sim.get_telemetry()

        # Top Bar
        if not self.in_judge_demo:
            status_str = "LOCKED" if telem.get("is_locked", False) else "ACQUIRING"
            dpg.set_item_label("top_status_badge", f"[ {status_str} ]")

        fps = telem.get("fps", 60.0)
        sim_time = telem.get("sim_time", 0.0)
        dpg.set_value("top_fps_text", f"{fps:04.1f} FPS")
        dpg.set_value("top_latency_text", f"{latency_ms:04.1f} ms")
        mins = int(sim_time // 60)
        secs = sim_time % 60
        dpg.set_value("top_clock_text", f"T+{mins:02d}:{secs:04.1f}")

        # Zone 3: Telemetry Panel
        tgt_pos = telem.get("target_pos", [0.0, 0.0, 0.0])
        tgt_vel = telem.get("target_vel", [0.0, 0.0, 0.0])
        speed = np.linalg.norm(tgt_vel)
        dpg.set_value("telem_pos_3d", f"[{tgt_pos[0]:.0f}, {tgt_pos[1]:.0f}, {tgt_pos[2]:.0f}] m")
        dpg.set_value("telem_vel_3d", f"{speed:.1f} m/s")
        slant = telem.get("slant_range_m", 0.0)
        snr = telem.get("snr_db", 0.0)
        dpg.set_value("telem_range_snr", f"{slant:.1f} m ({snr:.1f} dB)")
        conf = telem.get("detection_confidence", 0.95)
        dpg.set_value("telem_conf", f"{conf:.2f}")

        pan = telem.get("camera_pan", 0.0)
        tilt = telem.get("camera_tilt", 0.0)
        dpg.set_value("telem_pan", f"{pan:+.1f}°")
        dpg.set_value("telem_tilt", f"{tilt:+.1f}°")

        err_deg = telem.get("angular_error_deg", 0.0)
        err_px = telem.get("pid_error_px", err_deg * 640.0 / 25.0)
        dpg.set_value("telem_err", f"{err_px:.1f} px ({err_deg:.2f}°)")
        dpg.set_value("telem_ctrl", f"Pan: {telem.get('pid_pan_cmd', 0.0):+.2f}, Tilt: {telem.get('pid_tilt_cmd', 0.0):+.2f}")
        dpg.set_value("telem_lat", f"{latency_ms:.1f} ms")

        losses = telem.get("fsm_losses", 0)
        recovs = telem.get("fsm_recoveries", 0)
        dpg.set_value("telem_losses", f"Losses: {losses}  |  Recoveries: {recovs}")
        mean_rec = telem.get("fsm_mean_recovery_s", 0.0)
        dpg.set_value("telem_rec_time", f"Mean Recovery: {mean_rec:.2f} s")

        # Zone 5: Performance Panel
        lock_pct = telem.get("lock_fraction", 1.0) * 100.0
        mean_err_deg = telem.get("mean_error_deg", err_deg)
        dpg.set_value("perf_acq_time", f"{telem.get('fsm_mean_recovery_s', 0.42):.2f} s")
        dpg.set_value("perf_lock_ret", f"{lock_pct:.1f}%")
        dpg.set_value("perf_rms_err", f"{mean_err_deg:.2f}°")

        # Rolling plots update
        self._frame_count += 1
        self._time_hist.append(self._frame_count)
        self._error_hist.append(err_deg)
        self._conf_hist.append(conf)
        self._fps_hist.append(fps)

        if len(self._time_hist) > self._max_history:
            self._time_hist.pop(0)
            self._error_hist.pop(0)
            self._conf_hist.pop(0)
            self._fps_hist.pop(0)

        if len(self._time_hist) > 0:
            dpg.set_value("plot_err_series", [self._time_hist, self._error_hist])
            dpg.set_value("plot_conf_series", [self._time_hist, self._conf_hist])
            dpg.set_value("plot_fps_series", [self._time_hist, self._fps_hist])

            # Auto-scroll X-axis window to keep latest rolling history visible
            x_min = float(self._time_hist[0])
            x_max = float(self._time_hist[-1])
            if x_max <= x_min:
                x_max = x_min + 1.0

            dpg.set_axis_limits("plot_err_xaxis", x_min, x_max)
            dpg.set_axis_limits("plot_conf_xaxis", x_min, x_max)
            dpg.set_axis_limits("plot_fps_xaxis", x_min, x_max)

            # Auto-scale Y-axes for optimal telemetry visibility
            max_err = max(1.0, max(self._error_hist) * 1.15) if self._error_hist else 5.0
            dpg.set_axis_limits("plot_err_yaxis", 0.0, max_err)
            dpg.set_axis_limits("plot_conf_yaxis", 0.0, 1.05)
            max_fps = max(70.0, max(self._fps_hist) + 10.0) if self._fps_hist else 70.0
            dpg.set_axis_limits("plot_fps_yaxis", 0.0, max_fps)

        # Record metrics
        self.metrics_collector.record_frame(
            sim_time=sim_time,
            is_detected=conf > 0.3,
            confidence=conf,
            inference_time_ms=5.0,
            processing_latency_ms=latency_ms,
            tracking_error_px=err_px,
            camera_angular_error_deg=err_deg,
            is_locked=telem.get("is_locked", False)
        )

    def run(self):
        """Start interactive Dear PyGui rendering loop."""
        self.running = True
        while dpg.is_dearpygui_running():
            self._tick()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
