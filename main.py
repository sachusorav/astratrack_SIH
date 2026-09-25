"""
ASTRATRACK — Main Entry Point (Deprecated)

[NOTICE] The 2D-only standalone entry point is deprecated.
ASTRATRACK has been merged into a unified 3D application with `run_3d_simulator.py`
as the sole entry point.

Running `python main.py` now automatically redirects to `run_3d_simulator.py`.
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator.run_3d_sim import main as run_3d_main


def main():
    print("=" * 60)
    print("  [NOTICE] ASTRATRACK 2D Standalone is deprecated.")
    print("  Redirecting to unified 3D application: run_3d_simulator.py")
    print("=" * 60)
    print()
    run_3d_main()


if __name__ == "__main__":
    main()
