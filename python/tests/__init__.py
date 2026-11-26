"""Fast unit tests for the pi-robotics Python package.

These are pure-Python tests over compiled MuJoCo models — no daemon, no GL, no
browser. They cover the model builders, inverse kinematics, geometry/containment,
actuator math, and reel composition.

Run from the repo root with the project venv:

    .venv/bin/python -m pytest

(the ``pythonpath`` in ``pytest.ini`` puts ``python/`` on ``sys.path``).
"""
