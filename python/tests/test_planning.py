"""Joint-limit clamping, collision checks, and plan_path."""

import mujoco
import numpy as np
import pytest

from pi_robotics import ik

_RAIL_XML = """
<mujoco>
  <worldbody>
    <body name="arm">
      <joint name="j" type="slide" axis="1 0 0" range="0 0.6"/>
      <geom name="ee" type="sphere" size="0.05"/>
    </body>
    <body name="wall">
      <geom name="wall_geom" type="box" size="0.02 0.5 0.5" pos="0.4 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def rail():
    model = mujoco.MjModel.from_xml_string(_RAIL_XML)
    return model, np.zeros(model.nq)


def test_joint_limits_clamp_the_solution(rail):
    model, qpos0 = rail
    # Target is well past the joint's 0.6 m limit.
    solution = ik.solve_body(model, qpos0, "arm", [0.9, 0, 0])
    assert solution["joints"]["j"] <= 0.6 + 1e-6


def test_solve_within_limits_reaches_target(rail):
    model, qpos0 = rail
    solution = ik.solve_body(model, qpos0, "arm", [0.25, 0, 0])
    assert solution["joints"]["j"] == pytest.approx(0.25, abs=1e-3)


def test_collision_pairs_detects_overlap(rail):
    model, qpos0 = rail
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qpos[0] = 0.4  # into the wall
    pairs = ik.collision_pairs(model, data, ignore_geoms=None)
    names = {frozenset((p["geom1"], p["geom2"])) for p in pairs}
    assert frozenset(("ee", "wall_geom")) in names


def test_plan_path_stops_before_a_collision(rail):
    model, qpos0 = rail
    plan = ik.plan_path(
        model, qpos0, "arm", [0.55, 0, 0], steps=6, collision=True
    )
    assert plan["blocked"] is True
    # The colliding waypoint is the last one included, and it is flagged.
    assert plan["path"][-1]["collisions"]
    assert all(not entry["collisions"] for entry in plan["path"][:-1])


def test_plan_path_allows_ignored_geoms(rail):
    model, qpos0 = rail
    wall = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_geom")
    plan = ik.plan_path(
        model, qpos0, "arm", [0.25, 0, 0], steps=4,
        collision=True, ignore_geoms=[wall],
    )
    assert plan["blocked"] is False
    assert plan["steps"] == 4


def test_posture_target_reads_a_name_map(rail):
    model, _ = rail
    joints = [0]
    target = ik._posture_target(model, joints, {"j": 0.2}, np.array([0.0]))
    assert target is not None
    assert float(target[0]) == pytest.approx(0.2)
