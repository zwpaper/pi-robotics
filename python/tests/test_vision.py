"""Robot-agnostic end-effector / gripper resolution for the vision servo."""

import mujoco
import pytest

from pi_robotics import vision

_SITE_XML = """
<mujoco>
  <worldbody>
    <body name="arm">
      <geom type="box" size="0.02 0.02 0.1"/>
      <site name="attachment_site" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

_BODY_XML = """
<mujoco>
  <worldbody>
    <body name="base">
      <body name="link_gripper_slider" pos="0 0 0.1">
        <joint name="slide" type="slide" axis="0 0 1"/>
        <geom type="box" size="0.02 0.02 0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="grip" joint="slide"/>
    <position name="head_pan" joint="slide"/>
  </actuator>
</mujoco>
"""


def test_resolve_ee_prefers_a_named_site():
    model = mujoco.MjModel.from_xml_string(_SITE_XML)
    assert vision.resolve_ee(model) == ("site", "attachment_site")


def test_resolve_ee_falls_back_to_a_gripper_body():
    model = mujoco.MjModel.from_xml_string(_BODY_XML)
    assert vision.resolve_ee(model) == ("body", "link_gripper_slider")


def test_resolve_ee_honours_an_explicit_choice():
    model = mujoco.MjModel.from_xml_string(_SITE_XML)
    assert vision.resolve_ee(model, site="attachment_site") == ("site", "attachment_site")


def test_resolve_ee_rejects_unknown_names():
    model = mujoco.MjModel.from_xml_string(_SITE_XML)
    with pytest.raises(ValueError, match="unknown end-effector site"):
        vision.resolve_ee(model, site="ghost")
    with pytest.raises(ValueError, match="unknown end-effector body"):
        vision.resolve_ee(model, body="ghost")


def test_resolve_gripper_autodetects_and_validates():
    model = mujoco.MjModel.from_xml_string(_BODY_XML)
    assert vision.resolve_gripper(model) == "grip"
    assert vision.resolve_gripper(model, "grip") == "grip"
    assert vision.resolve_gripper(model, "ghost") is None


def test_head_ctrl_only_includes_present_actuators():
    model = mujoco.MjModel.from_xml_string(_BODY_XML)
    # `head_pan` exists but `head_tilt` does not -> only the present one.
    assert vision.head_ctrl(model) == {"head_pan": -1.5}
