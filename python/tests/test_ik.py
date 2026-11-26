"""Inverse kinematics on the arm3 scene (no simulation, no GL)."""

import mujoco
import numpy as np
import pytest

from pi_robotics import ik, scenes


@pytest.fixture
def arm3():
    model = mujoco.MjModel.from_xml_string(scenes.build_scene("arm3"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _site_pos(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return np.asarray(data.site_xpos[site_id], dtype=float).copy()


def test_solve_reaches_a_nearby_target(arm3):
    model, data = arm3
    start = _site_pos(model, data, "end_effector")
    target = start + np.array([0.05, 0.0, -0.05])
    solution = ik.solve(model, data.qpos, "end_effector", target)
    assert solution["error"]["norm"] < 0.02
    assert set(solution["joints"]) == {"joint0", "joint1", "joint2"}


def test_solve_reports_unknown_site(arm3):
    model, data = arm3
    with pytest.raises(ValueError, match="unknown site"):
        ik.solve(model, data.qpos, "not_a_site", [0, 0, 0])


def test_movable_joints_are_the_arm_not_the_world(arm3):
    model, _ = arm3
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    joints = ik._movable_joints(model, site_id)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
    assert names == ["joint0", "joint1", "joint2"]


def test_solve_body_reaches_a_nearby_target(arm3):
    model, data = arm3
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link3")
    start = np.asarray(data.xpos[body_id], dtype=float).copy()
    target = start + np.array([0.03, 0.0, -0.03])
    solution = ik.solve_body(model, data.qpos, "link3", target)
    assert solution["error"]["norm"] < 0.02


def test_solve_body_reports_unknown_body(arm3):
    model, data = arm3
    with pytest.raises(ValueError, match="unknown body"):
        ik.solve_body(model, data.qpos, "not_a_body", [0, 0, 0])


def test_solve_target_auto_detects_site_and_body(arm3):
    model, data = arm3
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    start = np.asarray(data.site_xpos[site_id], dtype=float).copy()
    # A site name resolves through the site path...
    sol = ik.solve_target(model, data.qpos, "end_effector", start + 0.01)
    assert sol["error"]["norm"] < 0.02
    # ...and a body name through the body path.
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link3")
    bstart = np.asarray(data.xpos[body_id], dtype=float).copy()
    sol = ik.solve_target(model, data.qpos, "link3", bstart + 0.01)
    assert sol["error"]["norm"] < 0.02
