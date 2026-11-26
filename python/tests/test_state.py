"""`/state` observability: named sensors, limits, body quats, contact pairs."""

import mujoco
import pytest

from pi_robotics import scenes
from pi_robotics.control import SharedState


def _state(tmp_path) -> SharedState:
    return SharedState(
        runtime_dir=tmp_path,
        scene_name="test",
        scene_params={},
        viewer_port=0,
        control_port=0,
    )


def test_model_metadata_exposes_limits_and_sensors(tmp_path):
    model = mujoco.MjModel.from_xml_string(scenes.build_scene("arm3"))
    state = _state(tmp_path)
    state.publish_model(model)
    meta = state.snapshot()["model"]
    assert meta["sensors"] == ["q0", "q1", "q2", "ee_pos"]
    assert meta["ctrlrange"]["joint0"] == [-3.0, 3.0]
    assert meta["joint_limits"]["joint0"] == [-3.0, 3.0]


def test_state_exposes_sensordata_and_body_quat(tmp_path):
    model = mujoco.MjModel.from_xml_string(scenes.build_scene("arm3"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = _state(tmp_path)
    state.publish_model(model)
    state.publish_data(model, data)
    snap = state.snapshot()
    assert set(snap["sensordata"]) >= {"q0", "q1", "q2", "ee_pos"}
    assert len(snap["sensordata"]["ee_pos"]) == 3  # vector sensor stays a list
    assert snap["body_quat"]["link3"][0] == pytest.approx(1.0)


def test_state_reports_contact_pairs_with_normal_force(tmp_path):
    model = mujoco.MjModel.from_xml_string(
        scenes.build_scene("box_drop", {"height": 0.09, "size": 0.1})
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    state = _state(tmp_path)
    state.publish_model(model)
    state.publish_data(model, data)
    snap = state.snapshot()
    assert snap["contacts"]
    top = snap["contacts"][0]
    assert {"geom1", "geom2", "normal", "dist"} <= set(top)
