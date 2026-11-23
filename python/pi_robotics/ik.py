"""Damped-least-squares inverse kinematics for a named site.

Ported from the approach in jimmyyhwu/tidybot2's ``ik_solver.py`` (which follows
mjctrl / dm_control): iteratively move a site to a target pose, using only the
joints on the chain from the site's body up to (but excluding) the root body —
so a mobile base is left alone and only the arm moves.

Pure MuJoCo + NumPy, no extra dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import mujoco


def _chain_joints(model: mujoco.MjModel, body_id: int) -> list[int]:
    """Joint ids on the chain from the root body down to ``body_id``."""
    joints: list[int] = []
    body = body_id
    while body != 0:
        for k in range(model.body_jntnum[body]):
            joints.append(model.body_jntadr[body] + k)
        body = model.body_parentid[body]
    return joints


def _movable_joints(
    model: mujoco.MjModel, site_id: int, exclude_root: bool = True
) -> list[int]:
    """Hinge/slide joints between the site and the root (the arm, not the base)."""
    body = model.site_bodyid[site_id]
    chain: list[int] = []
    cursor = body
    while cursor != 0:
        chain.append(cursor)
        cursor = model.body_parentid[cursor]
    root = chain[-1] if chain else 0
    joints: list[int] = []
    for b in chain:
        if exclude_root and b == root:
            continue
        for k in range(model.body_jntnum[b]):
            jid = int(model.body_jntadr[b]) + k
            jtype = int(model.jnt_type[jid])
            if jtype in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                joints.append(jid)
    return sorted(set(joints))


def solve(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    site_name: str,
    target_pos: np.ndarray | list[float],
    target_quat: np.ndarray | list[float] | None = None,
    *,
    max_iters: int = 80,
    tol: float = 2e-4,
    damping: float = 1e-3,
    max_step: float = 0.3,
) -> dict[str, Any]:
    """Solves IK for ``site_name`` and returns target joint angles.

    ``target_quat`` is ``(w, x, y, z)`` (MuJoCo convention); when omitted only the
    position is solved for. Returns ``{"joints": {name: angle}, "error": ...}``.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"unknown site '{site_name}'")
    joints = _movable_joints(model, site_id)
    if not joints:
        raise ValueError(f"no movable joints for site '{site_name}'")
    dofs = np.array([int(model.jnt_dofadr[j]) for j in joints], dtype=int)

    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    target = np.asarray(target_pos, dtype=float)
    want_rot = target_quat is not None
    tquat = np.asarray(target_quat, dtype=float) if want_rot else np.array([1.0, 0, 0, 0])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err = np.zeros(6)
    final = np.zeros(6)
    for _ in range(max_iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        final[:3] = target - data.site_xpos[site_id]
        if want_rot:
            cquat = np.zeros(4)
            mujoco.mju_mat2Quat(cquat, data.site_xmat[site_id].reshape(3, 3).flatten())
            inv = np.zeros(4)
            mujoco.mju_negQuat(inv, cquat)
            qerr = np.zeros(4)
            mujoco.mju_mulQuat(qerr, tquat, inv)
            vel = np.zeros(3)
            mujoco.mju_quat2Vel(vel, qerr, 1.0)
            final[3:] = vel
            err[:] = final
        else:
            err[:3] = final[:3]
        n = 6 if want_rot else 3
        if np.linalg.norm(err[:n]) < tol:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack([jacp, jacr]) if want_rot else jacp
        jac = jac[:, dofs]
        dq = jac.T @ np.linalg.solve(
            jac @ jac.T + damping * np.eye(n), err[:n]
        )
        biggest = float(np.abs(dq).max()) if dq.size else 0.0
        if biggest > max_step:
            dq *= max_step / biggest
        full = np.zeros(model.nv)
        full[dofs] = dq
        mujoco.mj_integratePos(model, data.qpos, full, 1.0)

    result: dict[str, float] = {}
    for jid in joints:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint{jid}"
        result[name] = float(data.qpos[model.jnt_qposadr[jid]])
    return {
        "joints": result,
        "error": {
            "pos": [round(float(v), 4) for v in final[:3]],
            "rot": [round(float(v), 4) for v in final[3:]] if want_rot else None,
            "norm": round(float(np.linalg.norm(final[: 6 if want_rot else 3])), 4),
        },
    }
