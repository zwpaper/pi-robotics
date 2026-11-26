"""Actuator-target math for nudge / move_ee."""

import mujoco
import pytest

from pi_robotics import motion, scenes


@pytest.fixture
def cartpole():
    return mujoco.MjModel.from_xml_string(scenes.build_scene("cartpole"))


@pytest.fixture
def arm3():
    return mujoco.MjModel.from_xml_string(scenes.build_scene("arm3"))


def test_current_actuators_maps_names(cartpole):
    assert motion.current_actuators(cartpole, [5.0]) == {"slide_force": 5.0}
    # A short/absent ctrl vector defaults to zero.
    assert motion.current_actuators(cartpole, []) == {"slide_force": 0.0}


def test_generic_deltas_are_clamped_to_ctrlrange(cartpole):
    current = motion.current_actuators(cartpole, [0.0])
    motion.apply_manual_deltas(cartpole, current, actuators={"slide_force": 100.0})
    assert current["slide_force"] == pytest.approx(30.0)

    motion.apply_manual_deltas(cartpole, current, actuators={"slide_force": -100.0})
    assert current["slide_force"] == pytest.approx(-30.0)


def test_unknown_generic_actuator_is_ignored(cartpole):
    current = motion.current_actuators(cartpole, [0.0])
    motion.apply_manual_deltas(cartpole, current, actuators={"ghost": 1.0})
    assert "ghost" not in current


def test_stretch_shortcuts_clamp_arm3_joints(arm3):
    current = motion.current_actuators(arm3, [0.0, 0.0, 0.0])
    # joint0 has ctrlrange [-3, 3].
    motion.apply_manual_deltas(arm3, current, actuators={"joint0": 100.0})
    assert current["joint0"] == pytest.approx(3.0)


def test_solution_to_actuators_maps_joint_names(arm3):
    solution = {"joints": {"joint0": 0.5, "joint1": 0.2, "joint2": 0.1}}
    mapped = motion.solution_to_actuators(arm3, solution)
    assert mapped == {"joint0": 0.5, "joint1": 0.2, "joint2": 0.1}


_TENDON_XML = """
<mujoco>
  <worldbody>
    <body name="base">
      <body name="a" pos="0 0 0">
        <joint name="j0" type="slide" axis="1 0 0"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
      <body name="b" pos="0 0 0.2">
        <joint name="j1" type="slide" axis="1 0 0"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t">
      <joint joint="j0" coef="1"/>
      <joint joint="j1" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <general name="extend" tendon="t" gainprm="100" biasprm="0 -100 -10"/>
    <motor name="wheel" tendon="t"/>
  </actuator>
</mujoco>
"""


def test_solution_to_actuators_maps_fixed_tendon_servo():
    model = mujoco.MjModel.from_xml_string(_TENDON_XML)
    mapped = motion.solution_to_actuators(
        model, {"joints": {"j0": 0.1, "j1": 0.2}}
    )
    assert mapped["extend"] == pytest.approx(0.3)
    # A motor on the same tendon is a force actuator, not a position servo.
    assert "wheel" not in mapped
