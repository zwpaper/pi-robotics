"""Damped-least-squares inverse kinematics with motion-planning guard-rails.

Ported from the approach in jimmyyhwu/tidybot2's ``ik_solver.py`` (which follows
mjctrl / dm_control): iteratively move a site (or body frame) to a target pose,
using only the joints on the chain from the target's body up to (but excluding)
the root body — so a mobile base is left alone and only the arm moves.

On top of the base solver this module adds:

* **joint-limit clamping** — limited hinge/slide joints never leave ``jnt_range``;
* **null-space posture bias** — a secondary posture is projected into the
  Jacobian null space, so redundant joints keep a preferred (e.g. home) pose;
* **collision checking** — ``mj_forward`` (which calls ``mj_collision``) on a
  scratch ``MjData`` reports contact pairs, and :func:`plan_path` checks every
  waypoint so motion stops before driving an arm through a table or itself.

Entry points: :func:`solve` (site), :func:`solve_body` (body), and
:func:`solve_target` (auto-detect). :func:`plan_path` returns a checked path.

Pure MuJoCo + NumPy, no extra dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import mujoco


def _movable_joints_from_body(
    model: mujoco.MjModel, body_id: int, exclude_root: bool = True
) -> list[int]:
    """Hinge/slide joints between a body and the root (the arm, not the base)."""
    chain: list[int] = []
    cursor = body_id
    while cursor != 0:
        chain.append(cursor)
        cursor = model.body_parentid[cursor]
    root = chain[-1] if chain else 0
    joints: list[int] = []
    for b in chain:
        # Skip the topmost body's joints (a mobile base) — unless it *is* the
        # target body (a single-link arm bolted straight to the world).
        if exclude_root and b == root and b != body_id:
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


def _movable_joints(
    model: mujoco.MjModel, site_id: int, exclude_root: bool = True
) -> list[int]:
    """Hinge/slide joints between a site's body and the root."""
    return _movable_joints_from_body(
        model, int(model.site_bodyid[site_id]), exclude_root=exclude_root
    )


def _joint_name(model: mujoco.MjModel, jid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint{jid}"


def _resolve(
    model: mujoco.MjModel, name: str, kind: str | None
) -> tuple[str, int, list[int]]:
    """Resolves ``name`` to ``(kind, obj_id, movable_joints)``."""
    if kind in (None, "site") and (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
    ):
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
        joints = _movable_joints(model, site_id)
        if not joints:
            raise ValueError(f"no movable joints for site '{name}'")
        return "site", site_id, joints
    if kind in (None, "body") and (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    ):
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        joints = _movable_joints_from_body(model, body_id)
        if not joints:
            raise ValueError(f"no movable joints for body '{name}'")
        return "body", body_id, joints
    raise ValueError(f"unknown site or body '{name}'")


def _obj_position(
    model: mujoco.MjModel, qpos: np.ndarray, kind: str, obj_id: int
) -> np.ndarray:
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = qpos
    mujoco.mj_forward(model, scratch)
    if kind == "site":
        return np.asarray(scratch.site_xpos[obj_id], dtype=float).copy()
    return np.asarray(scratch.xpos[obj_id], dtype=float).copy()


def collision_pairs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ignore_geoms: Any = None,
) -> list[dict[str, Any]]:
    """Contact pairs at the current state as ``{geom1, geom2, dist}``.

    ``mj_forward`` recomputes kinematics and collision (``mj_collision``), so the
    contacts always match ``data``. Geom ids in ``ignore_geoms`` are dropped
    (e.g. the object being grasped).
    """
    ignore = {int(g) for g in (ignore_geoms or ())}
    mujoco.mj_forward(model, data)
    pairs: list[dict[str, Any]] = []
    for i in range(data.ncon):
        con = data.contact[i]
        if int(con.geom1) in ignore or int(con.geom2) in ignore:
            continue
        pairs.append(
            {
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
                or f"geom{con.geom1}",
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
                or f"geom{con.geom2}",
                "dist": round(float(con.dist), 5),
            }
        )
    return pairs


def _pair_key(pair: dict[str, Any]) -> tuple[str, str]:
    return tuple(sorted((pair["geom1"], pair["geom2"])))


def _posture_target(
    model: mujoco.MjModel,
    joints: list[int],
    posture: Any,
    current: np.ndarray,
) -> np.ndarray | None:
    """Per-joint reference angles from a name->angle map or a full qpos array."""
    if posture is None:
        return None
    adr = [int(model.jnt_qposadr[j]) for j in joints]
    if isinstance(posture, dict):
        return np.array(
            [
                float(posture.get(_joint_name(model, j), current[i]))
                for i, j in enumerate(joints)
            ],
            dtype=float,
        )
    arr = np.asarray(posture, dtype=float)
    if arr.size != model.nq:
        raise ValueError("posture array must have length model.nq")
    return arr[adr]


def _clamp_limits(model: mujoco.MjModel, data: mujoco.MjData, joints: list[int]) -> None:
    for jid in joints:
        if not model.jnt_limited[jid]:
            continue
        adr = int(model.jnt_qposadr[jid])
        lo, hi = (float(v) for v in model.jnt_range[jid])
        data.qpos[adr] = min(max(data.qpos[adr], lo), hi)


def _solve(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    joints: list[int],
    obj_kind: str,
    obj_id: int,
    target_pos: np.ndarray | list[float],
    target_quat: np.ndarray | list[float] | None,
    *,
    max_iters: int,
    tol: float,
    damping: float,
    max_step: float,
    limits: bool = True,
    posture: Any = None,
    nullspace_gain: float = 0.05,
    collision: bool = False,
    ignore_geoms: Any = None,
) -> dict[str, Any]:
    """Shared DLS solver; ``obj_kind`` is ``"site"`` or ``"body"``."""
    dofs = np.array([int(model.jnt_dofadr[j]) for j in joints], dtype=int)
    qadr = np.array([int(model.jnt_qposadr[j]) for j in joints], dtype=int)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    target = np.asarray(target_pos, dtype=float)
    want_rot = target_quat is not None
    tquat = np.asarray(target_quat, dtype=float) if want_rot else np.array([1.0, 0, 0, 0])
    bias = posture is not None

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err = np.zeros(6)
    final = np.zeros(6)
    for _ in range(max_iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        if obj_kind == "site":
            cur_pos = data.site_xpos[obj_id]
            cur_mat = data.site_xmat[obj_id].reshape(3, 3)
        else:
            cur_pos = data.xpos[obj_id]
            cur_mat = data.xmat[obj_id].reshape(3, 3)
        final[:3] = target - cur_pos
        if want_rot:
            cquat = np.zeros(4)
            mujoco.mju_mat2Quat(cquat, cur_mat.flatten())
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
        if obj_kind == "site":
            mujoco.mj_jacSite(model, data, jacp, jacr, obj_id)
        else:
            mujoco.mj_jacBody(model, data, jacp, jacr, obj_id)
        jac = (np.vstack([jacp, jacr]) if want_rot else jacp)[:, dofs]
        gram = jac @ jac.T + damping * np.eye(n)
        if bias:
            jinv = jac.T @ np.linalg.solve(gram, np.eye(n))
            dq = jinv @ err[:n]
            target_q = _posture_target(model, joints, posture, data.qpos[qadr])
            assert target_q is not None
            null = np.eye(len(joints)) - jinv @ jac
            dq = dq + null @ (nullspace_gain * (target_q - data.qpos[qadr]))
        else:
            dq = jac.T @ np.linalg.solve(gram, err[:n])
        biggest = float(np.abs(dq).max()) if dq.size else 0.0
        if biggest > max_step:
            dq *= max_step / biggest
        full = np.zeros(model.nv)
        full[dofs] = dq
        mujoco.mj_integratePos(model, data.qpos, full, 1.0)
        if limits:
            _clamp_limits(model, data, joints)

    result: dict[str, float] = {}
    for jid in joints:
        result[_joint_name(model, jid)] = float(data.qpos[model.jnt_qposadr[jid]])
    collisions = collision_pairs(model, data, ignore_geoms) if collision else []
    return {
        "joints": result,
        "error": {
            "pos": [round(float(v), 4) for v in final[:3]],
            "rot": [round(float(v), 4) for v in final[3:]] if want_rot else None,
            "norm": round(float(np.linalg.norm(final[: 6 if want_rot else 3])), 4),
        },
        "collisions": collisions,
    }


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
    limits: bool = True,
    posture: Any = None,
    nullspace_gain: float = 0.05,
    collision: bool = False,
    ignore_geoms: Any = None,
) -> dict[str, Any]:
    """Solves IK for ``site_name`` and returns target joint angles.

    ``target_quat`` is ``(w, x, y, z)`` (MuJoCo convention); when omitted only the
    position is solved for. Returns
    ``{"joints": {name: angle}, "error": ..., "collisions": [...]}``.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"unknown site '{site_name}'")
    joints = _movable_joints(model, site_id)
    if not joints:
        raise ValueError(f"no movable joints for site '{site_name}'")
    return _solve(
        model, qpos0, joints, "site", site_id, target_pos, target_quat,
        max_iters=max_iters, tol=tol, damping=damping, max_step=max_step,
        limits=limits, posture=posture, nullspace_gain=nullspace_gain,
        collision=collision, ignore_geoms=ignore_geoms,
    )


def solve_body(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    body_name: str,
    target_pos: np.ndarray | list[float],
    target_quat: np.ndarray | list[float] | None = None,
    *,
    max_iters: int = 80,
    tol: float = 2e-4,
    damping: float = 1e-3,
    max_step: float = 0.3,
    limits: bool = True,
    posture: Any = None,
    nullspace_gain: float = 0.05,
    collision: bool = False,
    ignore_geoms: Any = None,
) -> dict[str, Any]:
    """Solves IK for a named body's frame (for models with no EE site)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"unknown body '{body_name}'")
    joints = _movable_joints_from_body(model, body_id)
    if not joints:
        raise ValueError(f"no movable joints for body '{body_name}'")
    return _solve(
        model, qpos0, joints, "body", body_id, target_pos, target_quat,
        max_iters=max_iters, tol=tol, damping=damping, max_step=max_step,
        limits=limits, posture=posture, nullspace_gain=nullspace_gain,
        collision=collision, ignore_geoms=ignore_geoms,
    )


def solve_target(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    name: str,
    target_pos: np.ndarray | list[float],
    target_quat: np.ndarray | list[float] | None = None,
    *,
    kind: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Solves for a site or body, auto-detecting which ``name`` refers to.

    ``kind`` may be ``"site"`` or ``"body"`` to skip detection.
    """
    if kind in (None, "site") and (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
    ):
        return solve(model, qpos0, name, target_pos, target_quat, **kwargs)
    if kind in (None, "body") and (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    ):
        return solve_body(model, qpos0, name, target_pos, target_quat, **kwargs)
    raise ValueError(f"unknown site or body '{name}'")


def plan_path(
    model: mujoco.MjModel,
    qpos0: np.ndarray,
    name: str,
    target_pos: np.ndarray | list[float],
    target_quat: np.ndarray | list[float] | None = None,
    *,
    steps: int = 6,
    kind: str | None = None,
    posture: Any = None,
    nullspace_gain: float = 0.05,
    collision: bool = True,
    ignore_geoms: Any = None,
    max_iters: int = 80,
    tol: float = 2e-4,
    damping: float = 1e-3,
    max_step: float = 0.3,
) -> dict[str, Any]:
    """Plans checked IK waypoints from the current pose to ``target_pos``.

    Each waypoint is solved from the previous one (a smooth joint-space walk).
    When ``collision`` is set, every waypoint runs a collision check and any
    contact pair **new** relative to the start pose marks the path ``blocked``;
    colliding-in-at-rest pairs and ``ignore_geoms`` are excluded. Returns
    ``{"path": [...], "blocked": bool, ...}``.
    """
    obj_kind, obj_id, joints = _resolve(model, name, kind)
    q0 = np.asarray(qpos0, dtype=float).copy()
    if q0.size != model.nq:
        raise ValueError("qpos0 must have length model.nq")
    start = _obj_position(model, q0, obj_kind, obj_id)
    target = np.asarray(target_pos, dtype=float)
    steps = max(1, int(steps))

    baseline: set[tuple[str, str]] = set()
    if collision:
        start_data = mujoco.MjData(model)
        start_data.qpos[:] = q0
        baseline = {_pair_key(p) for p in collision_pairs(model, start_data, ignore_geoms)}

    qpos = q0.copy()
    path: list[dict[str, Any]] = []
    blocked = False
    for i in range(1, steps + 1):
        waypoint = start + (target - start) * (i / steps)
        solution = _solve(
            model, qpos, joints, obj_kind, obj_id, waypoint, target_quat,
            max_iters=max_iters, tol=tol, damping=damping, max_step=max_step,
            limits=True, posture=posture, nullspace_gain=nullspace_gain,
            collision=False, ignore_geoms=ignore_geoms,
        )
        for jid in joints:
            qpos[int(model.jnt_qposadr[jid])] = solution["joints"][_joint_name(model, jid)]
        collisions: list[dict[str, Any]] = []
        if collision:
            scratch = mujoco.MjData(model)
            scratch.qpos[:] = qpos
            collisions = [
                p
                for p in collision_pairs(model, scratch, ignore_geoms)
                if _pair_key(p) not in baseline
            ]
        path.append(
            {
                "joints": solution["joints"],
                "error": solution["error"],
                "waypoint": [round(float(v), 4) for v in waypoint],
                "collisions": collisions,
            }
        )
        if collisions:
            blocked = True
            break

    return {
        "path": path,
        "blocked": blocked,
        "steps": len(path),
        "end_effector": name,
        "target": [round(float(v), 4) for v in target],
    }
