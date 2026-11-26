"""Pure geometry helpers: body world AABBs and containment checks.

Kept free of the control server so they can be unit-tested directly against a
compiled ``MjModel``. Both functions read a caller-supplied scratch ``MjData``
so they never touch the live simulation state.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import mujoco


def body_aabb(
    model: mujoco.MjModel, scratch: mujoco.MjData, body_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """World AABB of a body from its geoms (exact for box/sphere/cylinder)."""
    lo = np.full(3, 1e9)
    hi = np.full(3, -1e9)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body_id:
            continue
        pos = scratch.geom_xpos[g]
        mat = scratch.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        gtype = int(model.geom_type[g])
        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            half = np.abs(mat) @ size
        elif gtype in (
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            int(mujoco.mjtGeom.mjGEOM_MESH),
        ):
            half = np.full(3, float(model.geom_rbound[g]))
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            half = np.array([size[0], size[0], size[0] + size[1]])
        else:  # cylinder / ellipsoid / other
            half = np.array([size[0], size[0], max(size[0], size[1])])
        lo = np.minimum(lo, pos - half)
        hi = np.maximum(hi, pos + half)
    return lo, hi


def contains(
    model: mujoco.MjModel,
    scratch: mujoco.MjData,
    object_name: str,
    container_name: str,
) -> dict[str, Any]:
    """VLABench-style ``contain`` condition: is ``object`` inside ``container``?

    Reports whether the object body's centre lies within the container body's
    world AABB, so a task can be judged done without guessing.

    x/y: the object's centre must lie over the container's footprint.
    z:   the container's span extended upwards by the object's own height, so a
         tall object standing in a shallow tray counts as contained (its centre
         is above the tray walls).
    """
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, container_name)
    if obj_id < 0 or box_id < 0:
        return {"ok": False, "error": "unknown object/container body"}
    mujoco.mj_forward(model, scratch)
    center = np.asarray(scratch.xpos[obj_id], dtype=float)
    lo_o, hi_o = body_aabb(model, scratch, obj_id)
    obj_height = float(max(hi_o[2] - lo_o[2], 0.0))
    lo, hi = body_aabb(model, scratch, box_id)
    contained = bool(
        lo[0] <= center[0] <= hi[0]
        and lo[1] <= center[1] <= hi[1]
        and (lo[2] - 0.02) <= center[2] <= (hi[2] + obj_height)
    )
    return {
        "ok": True,
        "contained": contained,
        "object": object_name,
        "container": container_name,
        "object_center": [round(float(v), 4) for v in center],
        "container_aabb": {
            "min": [round(float(v), 4) for v in lo],
            "max": [round(float(v), 4) for v in hi],
        },
    }
