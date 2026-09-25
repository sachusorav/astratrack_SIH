# Implementation Plan - AstraTrack UI Bug Fixes & Aerospace Polish

Fix functional and visual bugs in the ASTRATRACK 3D Aerospace FSOC Tracking Simulator UI to ensure high fidelity, responsive layout, proper graph plotting, clear 3D viewport orientation, and full interactivity.

## Proposed Changes

### 1. Broken Characters ("?" Symbols)
#### [MODIFY] [gui3d.py](file:///c:/Users/heyim/Desktop/04_HACKATHONS/SIH/AstraaTrack/astratrack/simulator/gui3d.py)
#### [MODIFY] [scenarios.py](file:///c:/Users/heyim/Desktop/04_HACKATHONS/SIH/AstraaTrack/astratrack/simulator/scenarios.py)
#### [MODIFY] [renderer.py](file:///c:/Users/heyim/Desktop/04_HACKATHONS/SIH/AstraaTrack/astratrack/simulator/renderer.py)
#### [MODIFY] [dashboard.py](file:///c:/Users/heyim/Desktop/04_HACKATHONS/SIH/AstraaTrack/astratrack/ui/dashboard.py)
- Audit all string literals across all UI components, scenario presets, dropdown options, status badges, and tooltips.
- Replace non-standard unicode characters (`—`, `→`, `↔`, `★`) with safe ASCII/standard equivalents (` - `, `->`, `<->`, `*`).
- Update parsing functions (e.g. `scen_key = str(app_data).split("-")[0].strip()`) to match the clean separator format.

### 2. Live Performance Graphs Not Plotting (Zone 5)
#### [MODIFY] [gui3d.py](file:///c:/Users/heyim/Desktop/04_HACKATHONS/SIH/AstraaTrack/astratrack/simulator/gui3d.py)
- Update `_create_performance_panel` and `_update_telemetry_ui` to ensure `dpg.set_value()` correctly pushes updated arrays `[time_hist, error_hist]`, `[time_hist, conf_hist]`, `[time_hist, fps_hist]` every tick.
- Add dynamic X-axis scrolling bounds `dpg.set_axis_limits("plot_err_xaxis", min_x, max_x)` (or `dpg.fit_axis_data()`) so x-axis auto-scrolls with the progressing frame count/time history.
- Set appropriate Y-axis limits or auto-fitting for Boresight Tracking Error, Detector Confidence, and System FPS plots.

### 3. Layout Overflow / Clipped Content
#### [MODIFY] [gui3d.py](file:///c:/Users/heyim/Desktop/04_HACKATHONS/SIH/AstraaTrack/astratrack/simulator/gui3d.py)
- **Top Header Bar**: Refactor to be responsive with flexible element padding, scalable widths, and wrapping/min-width structure so "ASTRATRACK" title, rate/latency indicators, mission clock, and trigger buttons stay fully visible at all window sizes.
- **Zone 4 (Mission Control Panel)**: Enable text wrapping (`wrap=340`) and clean multiline formatting for "LIVE KEYBOARD CONTROLS GUIDE" so keyboard shortcut text is not clipped.
- **Scenario Preset Dropdown**: Restyle combo component with bounded height (`height=250`), internal scroll, dark elevation/shadow, proper pop-up z-index, and consistent dark theme styling.

### 4. 3D Viewport Orientation & Camera Framing
#### [MODIFY] [renderer.py](file:///c:/Users/heyim/Desktop/04_HACKATHONS/SIH/AstraaTrack/astratrack/simulator/renderer.py)
#### [MODIFY] [gui3d.py](file:///c:/Users/heyim/Desktop/04_HACKATHONS/SIH/AstraaTrack/astratrack/simulator/gui3d.py)
- Implement a persistent 3D Viewport Orientation Axis/Compass Indicator in the lower-left corner of the viewport canvas, fully synchronized with observer camera view orientation (X: East / Red, Y: Zenith / Green, Z: North / Blue).
- Fix scenario camera framing logic (`_on_scenario_select` / `load_scenario`): auto-calculate target position/orbit center for each scenario type (stationary, linear, orbital, ISL crosslink, etc.) and auto-scale initial camera orbit target and distance so the target is centered and properly proportioned.

### 5. Overall Visual Polish (Aerospace Control Station Theme)
#### [MODIFY] [gui3d.py](file:///c:/Users/heyim/Desktop/04_HACKATHONS/SIH/AstraaTrack/astratrack/simulator/gui3d.py)
- Upgrade Dear PyGui dark theme:
  - Curated aerospace color palette: Cyber Navy `#0B0E14` / Slate `#131924`, Glowing Cyan `#00E5FF`, Emerald `#00E676`, Amber `#FFB300`, Coral `#FF5252`.
  - Consistent corner rounding (`WindowRounding=6`, `ChildRounding=6`, `FrameRounding=4`).
  - Subtle panel elevation borders (`BorderColor=(35, 48, 70)`).
  - Clean typography hierarchy (contrast between section headers, telemetry metrics, and muted labels).
  - Styled sliders, buttons, combo menus, and checkboxes.

### 6. End-to-End Functionality Verification
#### [MODIFY] [gui3d.py](file:///c:/Users/heyim/Desktop/04_HACKATHONS/SIH/AstraaTrack/astratrack/simulator/gui3d.py)
#### [NEW] [tests/test_ui_interactive.py](file:///c:/Users/heyim/Desktop/04_HACKATHONS/SIH/AstraaTrack/astratrack/tests/test_ui_interactive.py)
- Verify scenario dropdown changes simulation state.
- Verify PID gain sliders update control loop in real time.
- Verify hotkeys (`1-9`, `X/D`, `K`, `P`, `C`, `V`, arrow keys, `Space`, `R`).
- Verify "START JUDGE DEMO", "Export Audit PDF", and "Pause" buttons.
- Create automated test verification for 15+ second simulation telemetry logging to guarantee graph arrays populate and plot axes scale.

## Verification Plan

### Automated Tests
- Run existing test suite: `pytest`
- Run new UI & telemetry test script: `pytest tests/test_ui_interactive.py`
- Run headless simulation script for 300 frames: `python run_3d_simulator.py --headless --frames 300`

### Manual Verification
- Run interactive 3D dashboard: `python run_3d_simulator.py`
- Test window resizing (from 1280x800 up to 1920x1080) to verify top header bar and Zone 4 text do not clip.
- Run scenario for 15+ seconds and observe Zone 5 live performance graphs (Boresight Error, Confidence, FPS) plotting smoothly with auto-scrolling x-axis.
- Verify 3D orientation compass indicator rotates smoothly with orbit camera rotation.
- Click "START JUDGE DEMO" and "Export Audit PDF" to confirm popup dialog and PDF output.
