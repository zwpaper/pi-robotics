"""Parametric MuJoCo scenes for the pi-robotics simulation daemon.

Every scene is returned as an MJCF XML string so the agent can read, edit, and
reuse the exact model that the daemon compiled. Scene authors may also pass a
path to any MJCF / URDF / MJB / MJZ file instead of a named scene.
"""

from __future__ import annotations

from typing import Any, Callable


def box_drop(params: dict[str, Any] | None = None) -> str:
    """A free box falling onto a plane. The simplest smoke-test scene."""
    p = params or {}
    height = float(p.get("height", 1.0))
    size = float(p.get("size", 0.1))
    color = str(p.get("color", "0.2 0.6 0.9 1"))
    return f"""
<mujoco model="box_drop">
  <option timestep="0.005"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.05 0.07 0.1"
             width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.2 0.25 0.3"
             rgb2="0.1 0.12 0.15" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.15"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>
    <body name="box" pos="0 0 {height}">
      <freejoint name="box_free"/>
      <geom name="box_geom" type="box" size="{size} {size} {size}" rgba="{color}"
            condim="4" friction="1 0.05 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


def falling_objects(params: dict[str, Any] | None = None) -> str:
    """Several rigid bodies dropped over a plane, useful for contact tuning."""
    p = params or {}
    count = int(p.get("count", 6))
    bodies = []
    for i in range(count):
        x = -0.6 + 1.2 * (i / max(count - 1, 1))
        z = 1.0 + 0.12 * i
        kind = ["box", "sphere", "capsule"][i % 3]
        if kind == "box":
            geom = '<geom type="box" size="0.08 0.08 0.08" rgba="0.9 0.5 0.2 1"/>'
        elif kind == "sphere":
            geom = '<geom type="sphere" size="0.08" rgba="0.3 0.8 0.5 1"/>'
        else:
            geom = '<geom type="capsule" size="0.05 0.06" rgba="0.7 0.4 0.9 1"/>'
        bodies.append(
            f"""
    <body name="obj{i}" pos="{x:.3f} 0 {z:.3f}">
      <freejoint/>
      {geom}
    </body>"""
        )
    joined = "".join(bodies)
    return f"""
<mujoco model="falling_objects">
  <option timestep="0.005"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.75 0.75 0.78 1"/>
    {joined}
  </worldbody>
</mujoco>
"""


def cartpole(params: dict[str, Any] | None = None) -> str:
    """Classic cart-pole on a rail with a force actuator."""
    p = params or {}
    pole_length = float(p.get("pole_length", 1.0))
    half = pole_length / 2.0
    return f"""
<mujoco model="cartpole">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="4 4 0.1" rgba="0.8 0.8 0.82 1"/>
    <body name="rail" pos="0 0 0.1">
      <geom type="box" size="1.2 0.03 0.03" rgba="0.4 0.4 0.45 1"/>
      <body name="cart" pos="0 0 0.06">
        <joint name="slider" type="slide" axis="1 0 0" range="-1 1"/>
        <geom name="cart" type="box" size="0.1 0.06 0.06" rgba="0.2 0.5 0.9 1"/>
        <body name="pole" pos="0 0 0.06">
          <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
          <geom name="pole" type="capsule" fromto="0 0 {-half} 0 0 {half}"
                size="0.02" rgba="0.9 0.3 0.2 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="slide_force" joint="slider" gear="1" ctrlrange="-30 30"/>
  </actuator>
</mujoco>
"""


def arm3(params: dict[str, Any] | None = None) -> str:
    """A 3-DOF planar arm with position actuators and a target ball.

    Links rotate about the world's +y axis, so the arm moves in the x-z plane.
    Actuators are named ``joint0``..``joint2`` and take target angles.
    """
    p = params or {}
    l1 = float(p.get("l1", 0.35))
    l2 = float(p.get("l2", 0.3))
    l3 = float(p.get("l3", 0.18))
    tx = float(p.get("target_x", 0.45))
    tz = float(p.get("target_z", 0.55))
    return f"""
<mujoco model="arm3">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <compiler angle="radian"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.82 1"/>
    <site name="target" pos="{tx} 0 {tz}" size="0.02" rgba="0.1 0.9 0.2 1"
          type="sphere"/>
    <body name="base" pos="0 0 0.05">
      <geom type="cylinder" size="0.06 0.05" rgba="0.3 0.3 0.35 1"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint0" type="hinge" axis="0 1 0" range="-3.0 3.0"/>
        <geom name="cap1" type="capsule" fromto="0 0 0 0 0 {l1}" size="0.025"
              rgba="0.2 0.55 0.9 1"/>
        <body name="link2" pos="0 0 {l1}">
          <joint name="joint1" type="hinge" axis="0 1 0" range="-3.0 3.0"/>
          <geom name="cap2" type="capsule" fromto="0 0 0 0 0 {l2}" size="0.022"
                rgba="0.25 0.75 0.5 1"/>
          <body name="link3" pos="0 0 {l2}">
            <joint name="joint2" type="hinge" axis="0 1 0" range="-3.0 3.0"/>
            <geom name="cap3" type="capsule" fromto="0 0 0 0 0 {l3}" size="0.02"
                  rgba="0.9 0.6 0.2 1"/>
            <site name="end_effector" pos="0 0 {l3}" size="0.01"
                  rgba="0.9 0.1 0.1 1" type="sphere"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="joint0" joint="joint0" kp="80" kv="5" ctrlrange="-3.0 3.0"/>
    <position name="joint1" joint="joint1" kp="60" kv="5" ctrlrange="-3.0 3.0"/>
    <position name="joint2" joint="joint2" kp="40" kv="4" ctrlrange="-3.0 3.0"/>
  </actuator>
  <sensor>
    <jointpos name="q0" joint="joint0"/>
    <jointpos name="q1" joint="joint1"/>
    <jointpos name="q2" joint="joint2"/>
    <framepos name="ee_pos" objtype="site" objname="end_effector"/>
  </sensor>
</mujoco>
"""


def table_task(params: dict[str, Any] | None = None) -> str:
    """A tabletop with a few graspable objects and a target zone."""
    p = params or {}
    objects = int(p.get("objects", 4))
    rows = []
    for i in range(objects):
        x = -0.25 + 0.5 * (i / max(objects - 1, 1))
        rows.append(
            f"""
    <body name="object{i}" pos="{x:.3f} 0 0.83">
      <freejoint/>
      <geom type="box" size="0.035 0.035 0.035" rgba="0.85 0.4 0.25 1"
            condim="6" friction="1.2 0.1 0.01"/>
    </body>"""
        )
    joined = "".join(rows)
    return f"""
<mujoco model="table_task">
  <option timestep="0.004"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.78 0.78 0.8 1"/>
    <body name="table" pos="0 0 0.4">
      <geom name="top" type="box" size="0.5 0.4 0.02" rgba="0.55 0.4 0.28 1"/>
      <geom name="leg1" type="box" size="0.03 0.03 0.4" pos="0.45 0.35 -0.4"
            rgba="0.4 0.3 0.2 1"/>
      <geom name="leg2" type="box" size="0.03 0.03 0.4" pos="-0.45 0.35 -0.4"
            rgba="0.4 0.3 0.2 1"/>
      <geom name="leg3" type="box" size="0.03 0.03 0.4" pos="0.45 -0.35 -0.4"
            rgba="0.4 0.3 0.2 1"/>
      <geom name="leg4" type="box" size="0.03 0.03 0.4" pos="-0.45 -0.35 -0.4"
            rgba="0.4 0.3 0.2 1"/>
    </body>
    <body name="bin" pos="0.0 -0.32 0.44">
      <geom type="box" size="0.12 0.06 0.02" rgba="0.2 0.7 0.3 0.6"/>
    </body>
    {joined}
  </worldbody>
</mujoco>
"""


def empty(_params: dict[str, Any] | None = None) -> str:
    """An empty world with lighting and a ground plane."""
    return """
<mujoco model="empty">
  <option timestep="0.005"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.82 1"/>
  </worldbody>
</mujoco>
"""


SCENES: dict[str, Callable[[dict[str, Any] | None], str]] = {
    "box_drop": box_drop,
    "falling_objects": falling_objects,
    "cartpole": cartpole,
    "arm3": arm3,
    "table_task": table_task,
    "empty": empty,
}

DEFAULT_SCENE = "box_drop"


def list_scenes() -> list[str]:
    return sorted(SCENES)


def build_scene(name: str | None, params: dict[str, Any] | None = None) -> str:
    """Returns MJCF XML for a named built-in scene."""
    key = (name or DEFAULT_SCENE).strip()
    if key not in SCENES:
        raise KeyError(
            f"Unknown scene '{key}'. Available: {', '.join(list_scenes())}"
        )
    return SCENES[key](params or {})
