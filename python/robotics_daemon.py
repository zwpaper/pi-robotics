#!/usr/bin/env python3
"""Entry point for the pi-robotics MuJoCo simulation daemon.

Run with a Python that has ``mujoco>=4.0.1`` (which bundles ``mujoco.studio``
and ``mujoco_studio_wasm``):

    python python/robotics_daemon.py --scene arm3 --runtime-dir .pi/robotics/runtime

The extension (``.pi/extensions/robotics``) spawns this automatically.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pi_robotics.daemon import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
