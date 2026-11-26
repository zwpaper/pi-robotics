"""Headless physics audit: penetration, settle, limits, saturation, sanity."""

import mujoco

from pi_robotics import audit, scenes


def test_audit_good_scene_passes():
    model = mujoco.MjModel.from_xml_string(
        scenes.build_scene("box_drop", {"height": 0.5, "size": 0.1})
    )
    report = audit.audit_model(model, seconds=1.0)
    assert report["summary"]["errors"] == 0
    assert report["summary"]["status"] in ("pass", "warn")
    assert report["model"]["ngeom"] >= 2
    assert isinstance(report["findings"], list)
    assert "energy_start" in report["sim"]


def test_audit_flags_deep_penetration():
    xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <body name="a"><freejoint/><geom name="a_g" type="box" size="0.1 0.1 0.1" pos="0 0 0.1"/></body>
        <body name="b"><geom name="b_g" type="box" size="0.1 0.1 0.1" pos="0 0 0.15"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    report = audit.audit_model(model, settle=False)
    categories = {f["category"] for f in report["findings"]}
    assert "penetration" in categories
    assert report["summary"]["errors"] >= 1


def test_audit_reports_near_zero_mass():
    xml = """
    <mujoco>
      <worldbody>
        <body name="tiny"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="1e-12"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    report = audit.audit_model(model, settle=False)
    assert any(f["category"] == "mass" for f in report["findings"])


def test_settle_stabilizes_a_dropped_box():
    model = mujoco.MjModel.from_xml_string(
        scenes.build_scene("box_drop", {"height": 0.5, "size": 0.1})
    )
    report = audit.audit_model(model, seconds=1.5)
    assert report["sim"]["unstable"] is False
    assert report["sim"]["final_qvel_norm"] is not None
    assert report["sim"]["final_qvel_norm"] < 1.0


def test_joint_limit_hit_is_reported():
    xml = """
    <mujoco>
      <worldbody>
        <body name="arm">
          <joint name="j" type="hinge" axis="0 1 0" range="-1 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.02"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # Park the joint exactly at its upper limit (radians/degrees per the model).
    data.qpos[model.jnt_qposadr[0]] = float(model.jnt_range[0][1])
    report = audit.audit_model(model, data=data, settle=False)
    assert any(f["category"] == "joint_limit" for f in report["findings"])
