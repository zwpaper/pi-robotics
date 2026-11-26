"""Actuator-target math for the manual ``nudge`` and ``move_ee`` servos.

These are pure functions over a compiled ``MjModel`` plus a name->value control
map, so the clamping and joint->actuator mapping can be unit-tested without a
running simulation. The control server owns the side effects (applying the
servo, sleeping, rendering the frame).
"""

from __future__ import annotations

from typing import Any

import mujoco


def actuator_limits(model: mujoco.MjModel) -> dict[str, tuple[float, float]]:
    """``{actuator_name: (lo, hi)}`` from the model's ctrlrange."""
    ranges: dict[str, tuple[float, float]] = {}
    for i in range(model.nu):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            or f"act{i}"
        )
        lo, hi = model.actuator_ctrlrange[i]
        ranges[name] = (float(lo), float(hi))
    return ranges


def current_actuators(
    model: mujoco.MjModel, ctrl_values: Any
) -> dict[str, float]:
    """``{actuator_name: value}`` from a published ``ctrl`` vector."""
    values = list(ctrl_values or [])
    current: dict[str, float] = {}
    for i in range(model.nu):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            or f"act{i}"
        )
        current[name] = float(values[i]) if i < len(values) else 0.0
    return current


def _clamp(
    ranges: dict[str, tuple[float, float]], name: str, value: float
) -> float:
    lo, hi = ranges.get(name, (0.0, 0.0))
    if hi > lo:
        return max(lo, min(hi, value))
    return value


def apply_manual_deltas(
    model: mujoco.MjModel,
    current: dict[str, float],
    *,
    extend: float | None = None,
    lift: float | None = None,
    wrist: float | None = None,
    grip: float | None = None,
    actuators: dict[str, float] | None = None,
) -> dict[str, float]:
    """Applies relative Stretch deltas (and an absolute ``grip``) in place.

    ``current`` is ``{actuator_name: value}`` (see :func:`current_actuators`);
    the Stretch shortcuts map to ``arm_extend`` / ``lift`` / ``wrist_yaw`` /
    ``grip``. ``actuators`` is the generic ``{name: delta}`` path for any robot
    (only known actuators are touched). Values are clamped to ``ctrlrange``.
    """
    ranges = actuator_limits(model)
    if extend is not None:
        current["arm_extend"] = _clamp(
            ranges, "arm_extend", current.get("arm_extend", 0.0) + float(extend)
        )
    if lift is not None:
        current["lift"] = _clamp(
            ranges, "lift", current.get("lift", 0.0) + float(lift)
        )
    if wrist is not None:
        current["wrist_yaw"] = _clamp(
            ranges, "wrist_yaw", current.get("wrist_yaw", 0.0) + float(wrist)
        )
    if grip is not None:
        current["grip"] = _clamp(ranges, "grip", float(grip))
    for name, delta in (actuators or {}).items():
        if name in current:
            current[name] = _clamp(ranges, name, current[name] + float(delta))
    return current


def solution_to_actuators(
    model: mujoco.MjModel, solution: dict[str, Any]
) -> dict[str, float]:
    """Maps an ``ik.solve`` joint solution to ``{actuator_name: target}``.

    Handles both direct joint actuators and **fixed-tendon** position actuators
    (e.g. Stretch's ``arm_extend``, which ties four telescoping joints into one
    actuator). Only actuators driven by joints on the IK chain are touched; a
    tendon actuator is included only when it looks like a position servo
    (``gainprm[0] == -biasprm[1]``), so wheel/torque tendons are skipped.
    """
    joints = solution.get("joints", {})
    out: dict[str, float] = {}
    for actuator in range(model.nu):
        aname = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
            or f"act{actuator}"
        )
        trntype = int(model.actuator_trntype[actuator])
        if trntype == int(mujoco.mjtTrn.mjTRN_JOINT):
            joint = int(model.actuator_trnid[actuator, 0])
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if jname in joints:
                out[aname] = float(joints[jname])
            continue
        if trntype != int(mujoco.mjtTrn.mjTRN_TENDON):
            continue
        tendon = int(model.actuator_trnid[actuator, 0])
        if int(model.tendon_type[tendon]) != int(
            mujoco.mjtTendon.mjTENDON_FIXED
        ):
            continue
        gain = float(model.actuator_gainprm[actuator, 0])
        bias = float(model.actuator_biasprm[actuator, 1])
        if not gain or abs(gain + bias) > 1e-6:
            continue  # not a tendon-length position servo
        length = 0.0
        used = False
        for k in range(int(model.tendon_num[tendon])):
            wrap = int(model.tendon_adr[tendon]) + k
            if int(model.wrap_type[wrap]) != int(
                mujoco.mjtWrap.mjWRAP_JOINT
            ):
                continue
            jname = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, int(model.wrap_objid[wrap])
            )
            if jname in joints:
                length += float(model.wrap_prm[wrap]) * float(joints[jname])
                used = True
        if used:
            out[aname] = length
    return out
