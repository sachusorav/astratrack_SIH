"""
ASTRATRACK — 3D FSOC Simulation Engine

Coordinates the target kinematics, tracking camera gimbal control loop,
disturbance perturbations, and metrics calculation in 3D space.

Supports:
- Deterministic mode (fixed seed & fixed time step)
- Real-time interactive mode
- Multi-scenario switching
- Full telemetry logging
- Optional full perception pipeline (IDetector → Kalman → FSM → PID)
"""

import time
import math
import numpy as np
from typing import Dict, List, Optional, Any

from simulator.target3d import Target3D, TargetConfig
from simulator.camera3d import TrackingCamera3D, CameraConfig3D
from simulator.scenarios import (
    Scenario, get_scenario_stationary, get_scenario_by_id_or_key,
    get_scenario_mode, list_scenarios_by_mode, MODE_LABELS, MODE_GROUPS
)
from simulator.renderer import Renderer3D


class Simulation3D:
    """
    Main 3D simulation coordinator for ASTRATRACK.

    Integrates:
    - Target trajectory kinematics
    - Pan/tilt tracking camera
    - Disturbance injection
    - Lock detection and metrics
    - Visual rendering pipeline
    """

    def __init__(self, scenario: Optional[Scenario] = None,
                 width: int = 1920, height: int = 1080,
                 deterministic: bool = False,
                 fixed_dt: float = 0.02,
                 use_pipeline: bool = False,
                 detector_type: str = "classical"):
        self.width = width
        self.height = height
        self.deterministic = deterministic
        self.fixed_dt = fixed_dt
        self.use_pipeline = use_pipeline
        self.detector_type = detector_type

        # Active scenario & mode
        self.scenario: Scenario = scenario or get_scenario_stationary()
        self.active_mode: str = get_scenario_mode(self.scenario.id)

        # Components
        self.target: Target3D = None
        self.camera: TrackingCamera3D = None
        self.renderer: Renderer3D = Renderer3D(width=width, height=height)

        # Optional full perception/estimation/control pipeline
        self._pipeline = None
        if use_pipeline:
            from simulator.pipeline3d import Pipeline3D
            self._pipeline = Pipeline3D(detector_type=detector_type)

        # Simulation state
        self.is_paused: bool = False
        self.sim_time: float = 0.0
        self.step_count: int = 0
        self.fps: float = 0.0

        # Switch flash (on-screen confirmation banner)
        self.switch_flash_timer: float = 0.0   # seconds remaining to show flash
        self.switch_flash_label: str = ""       # text shown in the flash banner

        # Metrics accumulators
        self._locked_steps: int = 0
        self._error_history: List[float] = []
        self._last_wall_time: float = time.perf_counter()
        self._last_pipeline_telem: Dict[str, Any] = {}

        # Initialize scenario
        self.reset()

    # ------------------------------------------------------------------ #
    # Lifecycle & Scenario Control
    # ------------------------------------------------------------------ #

    def load_scenario(self, scenario_or_key):
        """Switch active scenario by Scenario instance or key ('1'-'9')."""
        if isinstance(scenario_or_key, Scenario):
            self.scenario = scenario_or_key
        else:
            scen = get_scenario_by_id_or_key(str(scenario_or_key))
            if scen:
                self.scenario = scen
            else:
                return False
        self.active_mode = get_scenario_mode(self.scenario.id)
        self.reset()
        return True

    def switch_scene(self, mode: str, scenario_key: str) -> bool:
        """
        Hot-swap the active scenario and communication mode at runtime.

        Safe to call at any point — mid-tracking, mid-pause, any FSM state.
        Tears down current state and reinitializes cleanly into the new scenario.

        Args:
            mode: "ground_to_sat" or "sat_to_sat"
            scenario_key: Scenario ID string or numeric key ('1'-'9')

        Returns:
            True if switch succeeded, False if mode/key was invalid.
        """
        # Validate mode
        if mode not in MODE_GROUPS:
            print(f"[WARN] switch_scene: unknown mode '{mode}' — ignored.")
            return False

        # Validate scenario exists and belongs to the requested mode
        scen = get_scenario_by_id_or_key(scenario_key)
        if scen is None:
            print(f"[WARN] switch_scene: unknown scenario '{scenario_key}' — ignored.")
            return False
        if scen.id not in MODE_GROUPS[mode]:
            print(f"[WARN] switch_scene: scenario '{scen.id}' does not belong to mode '{mode}' — ignored.")
            return False

        # Apply switch
        self.scenario = scen
        self.active_mode = mode

        # Arm the on-screen confirmation banner
        mode_label = MODE_LABELS.get(mode, mode)
        self.switch_flash_label = f"\u27f6 {mode_label}  |  {scen.name}"
        self.switch_flash_timer = 1.5

        self.reset()
        print(f"[SWITCH] Mode: {mode_label}  |  Scenario: {scen.name}")
        return True

    def reset(self):
        """Reset the simulation back to t=0 with the active scenario."""
        target_cfg = self.scenario.target_config
        cam_cfg = self.scenario.camera_config

        # Deterministic seed override if requested
        if self.deterministic and target_cfg.random_seed is None:
            target_cfg.random_seed = 42

        self.target = Target3D(target_cfg)
        self.camera = TrackingCamera3D(cam_cfg)

        self.sim_time = 0.0
        self.step_count = 0
        self._locked_steps = 0
        self._error_history.clear()
        self._last_wall_time = time.perf_counter()
        self._last_pipeline_telem = {}
        # Don't clear switch_flash here — let it expire naturally so the banner shows
        if self._pipeline is not None:
            self._pipeline.reset()

        # Auto-scale initial camera orbit target and distance for clean scenario framing
        self.auto_frame_scenario()

    def auto_frame_scenario(self):
        """Auto-calculate target position/orbit center and scale camera orbit distance for current scenario."""
        if self.target is None or self.renderer is None:
            return
        tgt_pos = self.target.position
        # Orbit center focus
        self.renderer.orbit_target = np.array([tgt_pos[0] * 0.45, tgt_pos[1] * 0.75, tgt_pos[2] * 0.45], dtype=np.float64)
        dist = float(np.linalg.norm(tgt_pos - self.camera.position))
        self.renderer.orbit_distance = max(450.0, min(2200.0, dist * 0.85))

    @property
    def node_count(self) -> int:
        """Number of active nodes: 1 for ground_to_sat, 2 for sat_to_sat."""
        return 2 if self.active_mode == "sat_to_sat" else 1

    def set_deterministic(self, enable: bool, seed: Optional[int] = 42):
        """Toggle deterministic simulation mode."""
        self.deterministic = enable
        if enable:
            self.scenario.target_config.random_seed = seed
        else:
            self.scenario.target_config.random_seed = None
        self.reset()

    def toggle_pause(self):
        """Toggle simulation pause state."""
        self.is_paused = not self.is_paused

    # ------------------------------------------------------------------ #
    # Dynamic Runtime Pipeline Controls
    # ------------------------------------------------------------------ #

    def ensure_pipeline(self):
        """Ensure that perception pipeline is instantiated."""
        if self._pipeline is None:
            from simulator.pipeline3d import Pipeline3D
            self._pipeline = Pipeline3D(detector_type=self.detector_type)
            self.use_pipeline = True
        return self._pipeline

    def set_pid_gains(self, kp: float, ki: float, kd: float):
        """Update PID gains in the pipeline controller."""
        pipe = self.ensure_pipeline()
        pipe.controller.set_parameters(kp=kp, ki=ki, kd=kd)

    def set_detector_type(self, detector_type: str):
        """Switch detector backend ('classical' or 'ai')."""
        pipe = self.ensure_pipeline()
        from perception.factory import create_detector
        pipe.detector = create_detector(detector_type=detector_type, confidence_threshold=0.30)
        self.detector_type = detector_type

    def set_kalman_enabled(self, enabled: bool):
        """Toggle Kalman filter tracking."""
        pipe = self.ensure_pipeline()
        # Toggle Kalman filter operation
        pipe.kalman.prediction_enabled = enabled

    def set_prediction_enabled(self, enabled: bool):
        """Toggle trajectory prediction."""
        pipe = self.ensure_pipeline()
        pipe.kalman.prediction_enabled = enabled

    def set_controller_mode(self, mode):
        """Set controller mode (PID_CONTROL, P_CONTROL, DIRECT)."""
        pipe = self.ensure_pipeline()
        from control.camera_controller import ControllerMode
        if isinstance(mode, str):
            mode = ControllerMode[mode]
        pipe.controller.set_mode(mode)

    def set_target_speed(self, speed: float):
        """Update primary target speed."""
        if self.target is not None:
            self.target.config.speed = speed
            # Re-normalize velocity vector with new speed
            v_norm = np.linalg.norm(self.target.velocity)
            if v_norm > 1e-3:
                self.target.velocity = (self.target.velocity / v_norm) * speed

    def set_disturbance_level(self, level_name: str):
        """Set disturbance intensity preset dynamically."""
        mult_map = {
            "NORMAL": 1.0,
            "LIGHT DISTURBANCE": 1.5,
            "MODERATE DISTURBANCE": 2.5,
            "SEVERE DISTURBANCE": 4.0,
            "EXTREME STRESS TEST": 6.0,
        }
        mult = mult_map.get(level_name, 1.0)
        if self.target is not None:
            self.target.config.noise_sigma = 0.1 * mult
        if self.camera is not None:
            self.camera.config.vibration_amplitude = 0.05 * mult

    # ------------------------------------------------------------------ #
    # Stepping & Physics
    # ------------------------------------------------------------------ #

    def step(self, dt: Optional[float] = None):
        """
        Advance the simulation state by one time-step.

        Args:
            dt: Optional time delta. If None, uses fixed_dt or wall time.
        """
        if self.is_paused:
            return

        now = time.perf_counter()
        if dt is None:
            if self.deterministic:
                dt = self.fixed_dt
            else:
                wall_dt = now - self._last_wall_time
                dt = min(0.1, max(0.001, wall_dt))  # Clamp dt

        # FPS calculation
        if now > self._last_wall_time:
            instant_fps = 1.0 / max(1e-5, (now - self._last_wall_time))
            self.fps = 0.9 * self.fps + 0.1 * instant_fps
        self._last_wall_time = now

        # Decrement switch flash timer
        if self.switch_flash_timer > 0.0:
            self.switch_flash_timer = max(0.0, self.switch_flash_timer - dt)

        # 1. Update Target Kinematics
        self.target.update(dt)

        # 2. Update Ground Tracking Camera
        if self._pipeline is not None and self.target.visible:
            # Full perception → estimation → control pipeline
            self._last_pipeline_telem = self._pipeline.step(
                dt, self.target, self.camera, self.renderer
            )
            # Update angular error estimate for metrics (from Kalman pixel error)
            if self._last_pipeline_telem.get("pid_error_px", 0) > 0:
                err_deg = self._last_pipeline_telem["pid_error_px"] * self.camera.config.fov_deg / 640.0
                self.camera.angular_error_deg = err_deg
                self.camera.is_locked = (err_deg < self.camera.config.fov_deg * 0.4)
        else:
            # Direct proportional gimbal (default, existing behaviour)
            self.camera.update(dt, self.target.position, target_visible=self.target.visible)

        # 3. Accumulate tracking metrics
        self.sim_time += dt
        self.step_count += 1
        if self.camera.is_locked:
            self._locked_steps += 1
        self._error_history.append(self.camera.angular_error_deg)
        if len(self._error_history) > 1000:
            self._error_history.pop(0)

    # ------------------------------------------------------------------ #
    # Rendering & Telemetry
    # ------------------------------------------------------------------ #

    def get_sensor_frame(self, w: int = 640, h: int = 480) -> np.ndarray:
        """
        Return the rendered sensor-view frame (what the tracking camera sees).

        This is a (H, W, 3) BGR uint8 numpy array suitable for IDetector.detect().
        """
        return self.renderer._render_sensor_view(self.target, self.camera, w, h)

    def render(self) -> np.ndarray:
        """Render the current simulation state to an image frame."""
        telem = self.get_telemetry()
        return self.renderer.render(self.target, self.camera, telem)

    def get_telemetry(self) -> Dict:
        """Return real-time telemetry and tracking metrics."""
        lock_fraction = (self._locked_steps / max(1, self.step_count))
        mean_err = float(np.mean(self._error_history)) if self._error_history else 0.0
        slant_range = float(np.linalg.norm(self.target.position - self.camera.position))

        # Optical link budget estimation (Friis-like optical attenuation)
        # Higher distance = lower SNR, turbulence noise increases jitter
        base_snr = 30.0 - 5.0 * math.log10(max(1.0, slant_range / 300.0))
        if not self.target.visible:
            base_snr = 0.0

        telem = {
            "sim_time": self.sim_time,
            "step_count": self.step_count,
            "fps": self.fps,
            "mode": "Deterministic" if self.deterministic else "Real-Time",
            "active_mode": self.active_mode,
            "active_mode_label": MODE_LABELS.get(self.active_mode, self.active_mode),
            "node_count": self.node_count,
            "scenario_id": self.scenario.id,
            "scenario_name": self.scenario.name,
            "scenario_category": self.scenario.category,
            "target_pos": self.target.position.copy(),
            "target_vel": self.target.velocity.copy(),
            "camera_pan": self.camera.pan_deg,
            "camera_tilt": self.camera.tilt_deg,
            "angular_error_deg": self.camera.angular_error_deg,
            "mean_error_deg": mean_err,
            "is_locked": self.camera.is_locked,
            "lock_fraction": lock_fraction,
            "slant_range_m": slant_range,
            "snr_db": max(0.0, base_snr),
            "is_paused": self.is_paused,
            "pipeline_active": self._pipeline is not None,
            # Switch flash
            "switch_flash_active": self.switch_flash_timer > 0.0,
            "switch_flash_label": self.switch_flash_label,
        }
        # Merge pipeline telemetry (IDetector / Kalman / FSM / PID)
        if self._last_pipeline_telem:
            telem.update(self._last_pipeline_telem)
        return telem
