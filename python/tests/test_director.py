"""The hidden `director` body that glides a chase camera along the action front."""

import mujoco
import pytest

from pi_robotics.director import Director


def _model(with_director: bool = True):
    director = (
        """
    <body name="director" pos="-1 0 0" gravcomp="1">
      <freejoint/>
      <geom type="sphere" size="0.01" rgba="0 0 0 0" contype="0" conaffinity="0"
            mass="0.001"/>
      <camera name="chase" pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
    </body>"""
        if with_director
        else ""
    )
    xml = f"""
<mujoco><option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="4 4 0.1"/>
{director}
    <body name="slider" pos="-1 0 0.05">
      <joint type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def test_inactive_without_a_director_body():
    director = Director(_model(with_director=False))
    assert not director.active


def test_director_glides_toward_a_moving_body():
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    director = Director(model, lead=0.0, max_speed=10.0)
    assert director.active
    x0 = float(data.xpos[director.body][0])

    slider = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "slider")
    for _ in range(500):
        data.qvel[int(model.body_dofadr[slider])] = 2.0  # drive the slider forward
        director.step(model, data)
        mujoco.mj_step(model, data)

    assert float(data.xpos[director.body][0]) > x0 + 0.5
    # The director should sit just ahead of the moving body (lead=0).
    assert float(director.front) == pytest.approx(
        float(data.xpos[slider][0]), abs=0.5
    )


def test_director_is_rate_limited():
    model = _model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # A very slow cap means one step can only move a tiny amount.
    director = Director(model, lead=0.0, max_speed=0.001)
    slider = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "slider")
    data.qpos[int(model.jnt_qposadr[int(model.body_jntadr[slider])])] = 1.0
    mujoco.mj_forward(model, data)
    before = float(data.qpos[director.qadr])
    director.step(model, data)
    moved = abs(float(data.qpos[director.qadr]) - before)
    assert moved <= 0.001 * model.opt.timestep + 1e-9
