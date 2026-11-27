"""Smooth `qpos` moves: interpolate over time instead of teleporting (no sim, no GL)."""

import mujoco
import numpy as np
import pytest

from pi_robotics.daemon import RoboticsSimPlugin


def _plugin() -> RoboticsSimPlugin:
    # The tween logic is self-contained: only `_tween` state is needed.
    instance = object.__new__(RoboticsSimPlugin)
    instance._tween = None
    return instance


def _free_block_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    body = spec.worldbody.add_body(name="block", pos=[0, 0, 0.2])
    body.add_freejoint(name="block_free")
    body.add_geom(
        name="block_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05]
    )
    return spec.compile()


def test_qpos_with_duration_tweens_only_the_changed_entries():
    model = _free_block_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    plugin = _plugin()
    target = np.array(data.qpos, dtype=float)
    target[0] += 0.4  # move the block in x; nothing else changes

    applied: list[str] = []
    plugin._apply_command(
        model, data, {"cmd": "qpos", "values": target, "seconds": 2.0}, applied
    )

    assert applied[0].startswith("qpos_tween")
    assert data.qpos[0] == 0.0  # the move does not jump
    assert list(plugin._tween["indices"]) == [0]
    # The moved joint is frozen: its 6 dofs have their velocities zeroed so
    # `mj_step` cannot accumulate contact forces and fling the body.
    assert list(plugin._tween["dofs"]) == [0, 1, 2, 3, 4, 5]
    data.qvel[:] = 5.0
    data.time = 0.5
    plugin._advance_tween(data)
    assert not data.qvel[:6].any()

    # Half way through the eased move the block is halfway there.
    data.time = 1.0
    plugin._advance_tween(data)
    assert 0.15 < float(data.qpos[0]) < 0.25

    # It lands exactly on the target and the tween clears.
    data.time = 2.0
    plugin._advance_tween(data)
    assert float(data.qpos[0]) == pytest.approx(0.4)
    assert plugin._tween is None


def test_qpos_without_duration_still_teleports():
    model = _free_block_model()
    data = mujoco.MjData(model)
    plugin = _plugin()
    target = np.array(data.qpos, dtype=float)
    target[0] = 0.3

    applied: list[str] = []
    plugin._apply_command(model, data, {"cmd": "qpos", "values": target}, applied)

    assert applied == ["qpos[7]"]
    assert float(data.qpos[0]) == pytest.approx(0.3)
    assert plugin._tween is None
