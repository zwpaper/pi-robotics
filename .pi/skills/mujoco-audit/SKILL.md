---
name: mujoco-audit
description: >-
  Run a headless physics audit on a MuJoCo model or scene with the `robotics`
  tool's `audit` action: initial penetration, drop/settle stability,
  mass/inertia sanity, joint-limit hits, actuator saturation, and energy
  blow-up. Use when a scene behaves strangely (explodes, sinks, jitters, drifts,
  does not settle), before trusting a new MJCF, or when iterating on a model and
  wanting a structured pass/fail report to fix. Don't use for rendering, for
  non-MuJoCo simulators, or for editing a model (use the mujoco-spec-editing
  skill) — this skill diagnoses and reports.
metadata:
  author: pi-robotics
  version: "0.1.0"
---

# MuJoCo physics audit

The `audit` action compiles a model, runs cheap deterministic checks plus a
short **headless** settle simulation (no GL, no viewer), and returns a
structured report. The live simulation is untouched, so you can audit while the
panel is running.

## Usage

```jsonc
{ "action": "audit" }                                  // the loaded model
{ "action": "audit", "scene": "cartpole" }             // a built-in scene
{ "action": "audit", "scene": "stretch", "seconds": 3 }// a Menagerie robot
{ "action": "audit", "path": "scenes/my_scene.xml" }   // an MJCF/URDF/MJB file
```

`scene` and `path` compile a throwaway model, so you can audit several
candidates without loading them.

## What it checks

| Category | Meaning |
|---|---|
| `penetration` | Deep overlap at the spawn state, or **resting** overlap after settling. |
| `stability` | NaN/Inf divergence, or a body still moving after the settle window. |
| `mass` / `inertia` | Non-positive/near-zero mass, zero inertia on a movable body, extreme mass ratio. |
| `joint_limit` | A limited joint resting at (or clamped to) its range. |
| `actuator` | Non-finite force, force saturation, or control pinned at a range edge. |
| `energy` | Energy became non-finite or grew disproportionately during settle. |
| `geom` / `model` | Zero-size geom, no DOF, no actuators, inverted ranges. |

Report shape: `summary` (`status` = `pass`/`warn`/`fail`, counts), `model`
(dimensions), `sim` (penetration, `final_qvel_norm`, energies, `unstable`), and
`findings` (each with `severity`, `category`, `message`, `detail`).

## Interpreting and fixing

- **Penetration at spawn** → move the body/geom apart, or give the colliding
  pair a `contype`/`conaffinity` filter if they are meant to be cosmetic.
- **Resting penetration** → increase contact stiffness (`solref`) or reduce the
  drop height; check `margin`/`gap`.
- **Does not settle** → friction too low, actuator pushing forever, or an
  unstable integrator; try `implicitfast` and smaller `timestep`.
- **NaN / energy blow-up** → a bad `solref`/`solimp`, huge stiffness, zero
  inertia, or a contact explosion; find the offending body in `findings`.
- **Joint at limit** → the spawn pose or actuator target exceeds the range;
  retune `range`, the keyframe, or the target.
- **Actuator saturation** → raise `ctrlrange`/`forcerange` or lower the demand;
  saturation hides control authority.

Fix the model, `load` it again (the viewer hot-swaps), and re-`audit` until
`summary.errors == 0`. Warnings are often acceptable transients (e.g. an impact
penetration when a box is dropped) — judge them against the task.

## Pairing with motion

Collision-aware `move_ee` uses the same contact machinery: it plans
joint-limit-clamped IK waypoints and **stops before a new collision**, returning
`blocked: true` and the offending `{geom1, geom2}` pairs. If an audit shows a
table/robot overlap, `move_ee` will refuse paths through it — fix the scene or
pass `allow_collision: true` deliberately.
