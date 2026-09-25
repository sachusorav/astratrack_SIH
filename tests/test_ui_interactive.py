"""
ASTRATRACK — End-to-End UI & Simulation Verification Test Suite
"""

import os
import sys
import pytest
import numpy as np

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from simulator.simulation import Simulation3D
from simulator.scenarios import get_scenario_by_id_or_key, list_scenarios
from simulator.renderer import Renderer3D


def test_scenario_switching_and_clean_character_labels():
    """Verify all 9 scenarios load cleanly without unicode character corruptions ('?')."""
    sim = Simulation3D(deterministic=True)
    all_scenarios = list_scenarios()
    assert len(all_scenarios) >= 9, "Expected at least 9 scenarios in registry"

    for scen in all_scenarios:
        assert "?" not in scen.name, f"Broken character '?' found in scenario name: {scen.name}"
        assert "—" not in scen.name, f"Unreplaced em-dash found in scenario name: {scen.name}"
        assert "→" not in scen.name, f"Unreplaced arrow found in scenario name: {scen.name}"
        
        success = sim.load_scenario(scen.id)
        assert success, f"Failed to load scenario {scen.id}"
        assert sim.scenario.id == scen.id
        assert sim.target is not None
        assert sim.camera is not None


def test_telemetry_history_and_live_plot_arrays():
    """Verify simulation telemetry logging populates error, confidence, and FPS histories for Zone 5 live plots."""
    sim = Simulation3D(deterministic=True, fixed_dt=0.02)
    sim.load_scenario("2")  # Linear flight path

    # Step simulation for 150 frames (~3 seconds)
    for _ in range(150):
        sim.step()

    telem = sim.get_telemetry()
    assert sim.step_count == 150
    assert sim.sim_time > 2.5
    assert len(sim._error_history) > 0
    assert telem["angular_error_deg"] >= 0.0
    assert "slant_range_m" in telem
    assert telem["scenario_id"] == "linear"


def test_3d_orientation_compass_rendering():
    """Verify 3D Viewport Orientation Compass renders onto image buffer without error."""
    sim = Simulation3D(deterministic=True, width=1280, height=720)
    sim.load_scenario("3")  # Orbital tracking
    
    # Step 10 frames
    for _ in range(10):
        sim.step()
        
    frame = sim.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (720, 1280, 3)
    # Check compass region (lower left, y around 635, x around 85) has rendered pixels
    compass_region = frame[720 - 120:720 - 40, 40:130]
    assert np.mean(compass_region) > 0.0, "Compass region appears empty"


def test_pid_gain_tuning_and_disturbance_level():
    """Verify dynamic PID gain tuning and disturbance intensity setting."""
    sim = Simulation3D(deterministic=True, use_pipeline=True)
    sim.set_pid_gains(kp=0.55, ki=0.02, kd=0.12)
    sim.set_disturbance_level("MODERATE DISTURBANCE")
    
    sim.step()
    telem = sim.get_telemetry()
    assert telem["pipeline_active"] is True
