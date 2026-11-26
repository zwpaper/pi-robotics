"""Headless physics audit for a compiled MuJoCo model.

Runs a battery of cheap, deterministic checks plus a short **headless** settle
simulation, and returns a structured report the agent can fix iteratively:

* initial penetration (deep overlaps at the spawn state),
* drop / settle stability (does it come to rest, or NaN/blow up),
* mass / inertia sanity (zero or absurd masses, zero inertia, mass ratios),
* joint-limit hits (a joint pinned at its range),
* actuator saturation (control or force at its limit),
* energy blow-up (non-finite or runaway energy).

Nothing here touches the live simulation: it builds its own ``MjData`` from a
reset state, so it is safe to run while the viewer is live, and it needs no GL.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import mujoco

# Finding severities, weakest to strongest.
SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}

# Tunables (SI units unless noted).
PENETRATION_ERROR = -0.01  # 1 cm overlap -> error
PENETRATION_WARN = -0.003  # 3 mm overlap -> warning
MASS_MIN = 1e-9
MASS_RATIO_WARN = 1e4
INERTIA_MIN = 1e-9
DEFAULT_SECONDS = 2.0
MAX_STEPS = 6000
SETTLE_QVEL_WARN = 1.0  # m/s (or rad/s) residual speed after settling
JOINT_LIMIT_TOL = 1e-4
SATURATION_TOL = 1e-4
ENERGY_GROWTH_WARN = 20.0


def _finding(severity: str, category: str, message: str, **detail: Any) -> dict[str, Any]:
    return {
        "severity": severity,
        "category": category,
        "message": message,
        "detail": detail,
    }


def _name(model: mujoco.MjModel, objtype: mujoco.mjtObj, i: int, fallback: str) -> str:
    return mujoco.mj_id2name(model, objtype, i) or f"{fallback}{i}"


def check_model_sanity(model: mujoco.MjModel) -> list[dict[str, Any]]:
    """Static checks that need no simulation: masses, inertia, limits, ranges."""
    findings: list[dict[str, Any]] = []
    if model.nq == 0:
        findings.append(
            _finding("warning", "model", "model has no degrees of freedom")
        )
    if model.nu == 0:
        findings.append(
            _finding("info", "model", "model has no actuators (passive)")
        )

    masses: list[tuple[str, float]] = []
    for bid in range(1, model.nbody):
        name = _name(model, mujoco.mjtObj.mjOBJ_BODY, bid, "body")
        mass = float(model.body_mass[bid])
        if not np.isfinite(mass) or mass <= 0:
            findings.append(
                _finding(
                    "error", "mass", f"body '{name}' has non-positive mass",
                    body=name, mass=mass,
                )
            )
            continue
        if mass < MASS_MIN:
            findings.append(
                _finding(
                    "warning", "mass", f"body '{name}' has a near-zero mass",
                    body=name, mass=mass,
                )
            )
        masses.append((name, mass))
        # Inertia sanity only matters for bodies that can move (have a joint).
        movable = model.body_jntnum[bid] > 0
        inertia = np.asarray(model.body_inertia[bid], dtype=float)
        if movable and mass > 0 and float(inertia.max()) <= INERTIA_MIN:
            findings.append(
                _finding(
                    "error", "inertia", f"movable body '{name}' has zero inertia",
                    body=name,
                )
            )
    if len(masses) >= 2:
        values = [m for _, m in masses]
        ratio = max(values) / max(min(values), MASS_MIN)
        if ratio > MASS_RATIO_WARN:
            findings.append(
                _finding(
                    "warning", "mass", "extreme body mass ratio",
                    ratio=round(float(ratio), 3),
                )
            )

    for gid in range(model.ngeom):
        size = np.asarray(model.geom_size[gid], dtype=float)
        if size.max() <= 0:
            name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, gid, "geom")
            findings.append(
                _finding("error", "geom", f"geom '{name}' has zero size", geom=name)
            )

    for jid in range(model.njnt):
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, jid, "joint")
        if model.jnt_limited[jid]:
            lo, hi = (float(v) for v in model.jnt_range[jid])
            if lo > hi:
                findings.append(
                    _finding(
                        "error", "joint_limit", f"joint '{name}' has an inverted range",
                        joint=name, low=lo, high=hi,
                    )
                )
    for aid in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid, "actuator")
        if model.actuator_ctrllimited[aid]:
            lo, hi = (float(v) for v in model.actuator_ctrlrange[aid])
            if lo > hi:
                findings.append(
                    _finding(
                        "error", "actuator",
                        f"actuator '{name}' has an inverted ctrlrange",
                        actuator=name,
                    )
                )
        if model.actuator_forcelimited[aid]:
            lo, hi = (float(v) for v in model.actuator_forcerange[aid])
            if lo > hi:
                findings.append(
                    _finding(
                        "error", "actuator",
                        f"actuator '{name}' has an inverted forcerange",
                        actuator=name,
                    )
                )
    return findings


def _contact_rows(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(data.ncon):
        con = data.contact[i]
        rows.append(
            {
                "geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1, "geom"),
                "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2, "geom"),
                "dist": round(float(con.dist), 5),
            }
        )
    return rows


def check_penetration(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[list[dict[str, Any]], float]:
    """Depth-overlap check at the current state (after ``mj_forward``)."""
    findings: list[dict[str, Any]] = []
    max_depth = 0.0
    for row in _contact_rows(model, data):
        max_depth = min(max_depth, row["dist"])
        if row["dist"] < PENETRATION_ERROR:
            findings.append(
                _finding(
                    "error", "penetration",
                    f"deep initial penetration {row['dist'] * 1000:.1f} mm "
                    f"({row['geom1']} / {row['geom2']})",
                    **row,
                )
            )
        elif row["dist"] < PENETRATION_WARN:
            findings.append(
                _finding(
                    "warning", "penetration",
                    f"initial penetration {row['dist'] * 1000:.1f} mm "
                    f"({row['geom1']} / {row['geom2']})",
                    **row,
                )
            )
    return findings, round(max_depth, 5)


def _reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)


def check_joint_limits(
    model: mujoco.MjModel, data: mujoco.MjData
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for jid in range(model.njnt):
        if not model.jnt_limited[jid]:
            continue
        name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, jid, "joint")
        lo, hi = (float(v) for v in model.jnt_range[jid])
        value = float(data.qpos[model.jnt_qposadr[jid]])
        if abs(value - lo) <= JOINT_LIMIT_TOL or abs(value - hi) <= JOINT_LIMIT_TOL:
            findings.append(
                _finding(
                    "warning", "joint_limit",
                    f"joint '{name}' rests at its limit",
                    joint=name, value=round(value, 4), low=lo, high=hi,
                )
            )
    return findings


def check_actuators(
    model: mujoco.MjModel, data: mujoco.MjData
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for aid in range(model.nu):
        name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid, "actuator")
        force = float(data.actuator_force[aid])
        if not np.isfinite(force):
            findings.append(
                _finding(
                    "error", "actuator", f"actuator '{name}' force is not finite",
                    actuator=name,
                )
            )
            continue
        if model.actuator_forcelimited[aid]:
            lo, hi = (float(v) for v in model.actuator_forcerange[aid])
            span = max(abs(lo), abs(hi), 1e-9)
            if abs(force) >= span - SATURATION_TOL * span:
                findings.append(
                    _finding(
                        "warning", "actuator",
                        f"actuator '{name}' is force-saturated",
                        actuator=name, force=round(force, 4),
                    )
                )
        ctrl = float(data.ctrl[aid])
        if model.actuator_ctrllimited[aid]:
            lo, hi = (float(v) for v in model.actuator_ctrlrange[aid])
            at_edge = abs(ctrl - lo) <= SATURATION_TOL or abs(ctrl - hi) <= SATURATION_TOL
            if at_edge and abs(force) > 0:
                findings.append(
                    _finding(
                        "info", "actuator",
                        f"actuator '{name}' control is at its range edge",
                        actuator=name, ctrl=round(ctrl, 4),
                    )
                )
    return findings


def simulate_settle(
    model: mujoco.MjModel, data: mujoco.MjData, *, seconds: float = DEFAULT_SECONDS
) -> dict[str, Any]:
    """Steps a fresh reset state with zero control and returns stability stats."""
    dt = float(model.opt.timestep) or 0.002
    steps = int(max(50, min(MAX_STEPS, round(seconds / dt))))
    max_depth = 0.0
    energy_start = energy_end = 0.0
    unstable = False
    energy = np.zeros(2)
    for i in range(steps):
        mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            unstable = True
            break
        for row in _contact_rows(model, data):
            max_depth = min(max_depth, row["dist"])
        if i == 0:
            mujoco.mj_energyPos(model, data)
            mujoco.mj_energyVel(model, data)
            energy_start = float(abs(data.energy[0]) + data.energy[1])
        if i == steps - 1 or (steps > 10 and i % max(1, steps // 10) == 0):
            mujoco.mj_energyPos(model, data)
            mujoco.mj_energyVel(model, data)
            energy_end = float(abs(data.energy[0]) + data.energy[1])
            energy = np.asarray(data.energy, dtype=float)
    qvel_norm = float(np.linalg.norm(data.qvel)) if not unstable else float("inf")
    return {
        "steps": steps,
        "seconds": round(steps * dt, 3),
        "unstable": unstable,
        "max_penetration": round(max_depth, 5),
        "final_qvel_norm": round(qvel_norm, 4) if np.isfinite(qvel_norm) else None,
        "energy_start": round(energy_start, 4),
        "energy_end": round(energy_end, 4),
        "energy": [round(float(v), 4) for v in energy],
    }


def audit_model(
    model: mujoco.MjModel,
    data: mujoco.MjData | None = None,
    *,
    seconds: float = DEFAULT_SECONDS,
    settle: bool = True,
) -> dict[str, Any]:
    """Runs all checks and returns a structured, agent-readable report."""
    scratch = data if data is not None else mujoco.MjData(model)
    if data is None:
        _reset(model, scratch)
    mujoco.mj_forward(model, scratch)

    findings: list[dict[str, Any]] = list(check_model_sanity(model))
    pen_findings, initial_depth = check_penetration(model, scratch)
    findings.extend(pen_findings)
    initial_contacts = int(scratch.ncon)

    settle_stats: dict[str, Any] = {}
    if settle:
        settle_stats = simulate_settle(model, scratch, seconds=seconds)
        if settle_stats["unstable"]:
            findings.append(
                _finding("error", "stability", "simulation diverged (NaN/Inf state)")
            )
        else:
            if settle_stats["final_qvel_norm"] is None:
                pass
            elif settle_stats["final_qvel_norm"] > SETTLE_QVEL_WARN:
                findings.append(
                    _finding(
                        "warning", "stability",
                        "object did not settle (residual velocity "
                        f"{settle_stats['final_qvel_norm']:.3g})",
                        qvel=settle_stats["final_qvel_norm"],
                    )
                )
            e0, e1 = settle_stats["energy_start"], settle_stats["energy_end"]
            if not np.isfinite(e1):
                findings.append(
                    _finding("error", "energy", "energy became non-finite")
                )
            elif e0 > 0 and e1 > e0 * ENERGY_GROWTH_WARN:
                findings.append(
                    _finding(
                        "warning", "energy",
                        f"energy grew {e1 / e0:.1f}x during settle",
                        energy_start=e0, energy_end=e1,
                    )
                )
        if settle_stats.get("max_penetration", 0.0) < PENETRATION_ERROR:
            findings.append(
                _finding(
                    "warning", "penetration",
                    "transient penetration during settle "
                    f"({settle_stats['max_penetration'] * 1000:.1f} mm)",
                    max_penetration=settle_stats["max_penetration"],
                )
            )
        # Resting penetration at the settled state (the initial check was spawn).
        for finding in check_penetration(model, scratch)[0]:
            finding["message"] = finding["message"].replace("initial ", "resting ")
            findings.append(finding)
    # Limit / saturation checks run on the final state (settled or initial).
    findings.extend(check_joint_limits(model, scratch))
    findings.extend(check_actuators(model, scratch))

    counts = {"error": 0, "warning": 0, "info": 0}
    for finding in findings:
        counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
    return {
        "ok": counts["error"] == 0,
        "source": "model",
        "summary": {
            "errors": counts["error"],
            "warnings": counts["warning"],
            "infos": counts["info"],
            "status": "fail" if counts["error"] else ("warn" if counts["warning"] else "pass"),
        },
        "model": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "njnt": int(model.njnt),
            "timestep": round(float(model.opt.timestep), 6),
        },
        "sim": {
            "initial_contacts": initial_contacts,
            "initial_penetration": initial_depth,
            **settle_stats,
        },
        "findings": findings,
    }
