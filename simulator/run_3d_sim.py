"""
ASTRATRACK — Unified 3D Aerospace FSOC Tracking Simulator Launcher

Runs the single integrated 3D application combining high-fidelity 3D visual rendering,
full perception/estimation/control pipeline, and aerospace ground station telemetry dashboard.

Usage:
    python run_3d_simulator.py
    python run_3d_simulator.py --scenario 3
    python run_3d_simulator.py --pipeline --detector ai
    python run_3d_simulator.py --deterministic --frames 60 --headless
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np

# Ensure root workspace is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from simulator.simulation import Simulation3D
from simulator.scenarios import list_scenarios, get_scenario_by_id_or_key


def main():
    parser = argparse.ArgumentParser(
        description="ASTRATRACK Unified 3D FSOC Tracking Simulation Application"
    )
    parser.add_argument(
        "--scenario", "-s", type=str, default="2",
        help=(
            "Scenario number (1-9) or ID. "
            "Ground→Sat [1-6]: stationary/linear/orbital/turbulence/high_speed/evasive. "
            "Sat↔Sat    [7-9]: sat_to_ground/isl_same_plane/isl_crosslink."
        )
    )
    parser.add_argument(
        "--deterministic", "-d", action="store_true",
        help="Run in deterministic mode with fixed random seed"
    )
    parser.add_argument(
        "--width", type=int, default=1920,
        help="Viewport width (pixels)"
    )
    parser.add_argument(
        "--height", type=int, default=1080,
        help="Viewport height (pixels)"
    )
    parser.add_argument(
        "--record", type=str, default=None,
        help="Path to save output video (e.g. output.mp4)"
    )
    parser.add_argument(
        "--frames", type=int, default=0,
        help="Run for N frames then exit (0 = infinite)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without displaying GUI window (useful for automated tests/recording)"
    )
    parser.add_argument(
        "--save-screenshot", type=str, default=None,
        help="Save an initial screenshot to this path and continue"
    )
    parser.add_argument(
        "--pipeline", action="store_true",
        help="Enable full perception pipeline (IDetector → Kalman → FSM → PID) instead of direct gimbal"
    )
    parser.add_argument(
        "--detector", type=str, default="classical",
        help="Detector backend when --pipeline is active: 'classical' (default) or 'ai'"
    )

    args = parser.parse_args()

    # Load scenario
    scenario = get_scenario_by_id_or_key(args.scenario)
    if scenario is None:
        print(f"[ERROR] Unknown scenario '{args.scenario}'. Available scenarios:")
        for s in list_scenarios():
            print(f"  {s.id:20s}  {s.name}  ({s.category})")
        sys.exit(1)

    print(f"============================================================")
    print(f"  ASTRATRACK — UNIFIED 3D FSOC TRACKING APPLICATION")
    print(f"============================================================")
    print(f"  Loaded Scenario: [{scenario.id}] {scenario.name}")
    print(f"  Category:        {scenario.category}")
    print(f"  Description:     {scenario.description}")
    print(f"  Sim Mode:        {'DETERMINISTIC' if args.deterministic else 'REAL-TIME'}")
    print(f"  Resolution:      {args.width}x{args.height}")
    print(f"  Pipeline:        {'ENABLED [' + args.detector.upper() + ' detector]' if args.pipeline else 'DISABLED (direct gimbal)'}")
    print(f"  GUI Mode:        {'HEADLESS (CLI)' if args.headless else 'INTERACTIVE DEAR PYGUI 3D DASHBOARD'}")
    print(f"============================================================")

    # Initialize simulation coordinator
    sim = Simulation3D(
        scenario=scenario,
        width=args.width,
        height=args.height,
        deterministic=args.deterministic,
        use_pipeline=args.pipeline,
        detector_type=args.detector,
    )

    # HEADLESS MODE EXECUTION
    if args.headless:
        writer = None
        if args.record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.record, fourcc, 30.0, (args.width, args.height))
            print(f"[*] Recording enabled: saving to '{args.record}'")

        frame_idx = 0
        saved_screenshot = False

        try:
            while True:
                sim.step()
                frame = sim.render()

                if args.save_screenshot and not saved_screenshot and frame_idx >= 5:
                    os.makedirs(os.path.dirname(os.path.abspath(args.save_screenshot)), exist_ok=True)
                    cv2.imwrite(args.save_screenshot, frame)
                    print(f"[*] Saved screenshot to: {args.save_screenshot}")
                    saved_screenshot = True

                if writer is not None:
                    writer.write(frame)

                frame_idx += 1
                if args.frames > 0 and frame_idx >= args.frames:
                    break
        except KeyboardInterrupt:
            print("\n[!] Headless simulation interrupted by user.")
        finally:
            if writer is not None:
                writer.release()
                print(f"[*] Recording finalized: {args.record}")
            print(f"[*] Headless simulation ended after {frame_idx} frames. Final sim time: {sim.sim_time:.2f}s")
        return

    # INTERACTIVE DEAR PYGUI 3D DASHBOARD MODE
    from simulator.gui3d import Dashboard3D
    dashboard = Dashboard3D(sim)
    dashboard.setup()
    dashboard.run()


if __name__ == "__main__":
    main()
