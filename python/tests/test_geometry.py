"""Body AABB and VLABench-style containment checks."""

import mujoco
import numpy as np
import pytest

from pi_robotics import geometry

_TEMPLATE = """
<mujoco model="containment">
  <worldbody>
    <body name="bin" pos="0 0 0.5">
      <geom name="bin_geom" type="box" size="0.1 0.1 0.02"/>
    </body>
    <body name="object" pos="{x} 0 {z}">
      <geom name="object_geom" type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


def _model(x=0.0, z=0.5):
    xml = _TEMPLATE.format(x=x, z=z)
    model = mujoco.MjModel.from_xml_string(xml)
    scratch = mujoco.MjData(model)
    mujoco.mj_forward(model, scratch)
    return model, scratch


def test_body_aabb_is_exact_for_a_box():
    model, scratch = _model()
    bin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin")
    lo, hi = geometry.body_aabb(model, scratch, bin_id)
    assert np.allclose(lo, [-0.1, -0.1, 0.48], atol=1e-6)
    assert np.allclose(hi, [0.1, 0.1, 0.52], atol=1e-6)


def test_object_over_the_footprint_is_contained():
    model, scratch = _model(x=0.0)
    result = geometry.contains(model, scratch, "object", "bin")
    assert result["ok"] is True
    assert result["contained"] is True
    assert result["object_center"] == [0.0, 0.0, 0.5]


def test_object_outside_the_footprint_is_not_contained():
    model, scratch = _model(x=0.5)
    result = geometry.contains(model, scratch, "object", "bin")
    assert result["contained"] is False


def test_unknown_bodies_report_an_error():
    model, scratch = _model()
    result = geometry.contains(model, scratch, "ghost", "bin")
    assert result == {"ok": False, "error": "unknown object/container body"}
