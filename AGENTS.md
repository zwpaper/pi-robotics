# AGENTS.md

This repository is a **pi extension** that brings MuJoCo robotics simulation and
a live 3D panel to the pi coding agent. It mirrors the simulation surface of
`google-deepmind/robotics-antigravity-plugin` on top of `google-deepmind/mujoco`.

## Layout

- `.pi/extensions/robotics/` — the pi extension (TypeScript, loaded by jiti).
  - `index.ts` — the `robotics` tool + `/robotics` command.
  - `daemon.ts` — spawns/supervises the Python daemon.
  - `client.ts` — loopback control HTTP client.
  - `paths.ts` — interpreter discovery, runtime paths, session file.
- `python/pi_robotics/` — daemon, scenes, control server, and focused helpers.
  - `daemon.py` — launches `mujoco.studio.launch_web` and the sim plugin.
  - `scenes.py` — parametric MJCF scenes.
  - `menagerie.py` — fetches Menagerie robot models (e.g. Stretch) on demand.
  - `control.py` — `ControlServer`: command orchestration + `/state` publisher.
  - `server.py` — loopback HTTP request handler + CORS allowlist.
  - `rendering.py` — offscreen render helpers (panel/camera, PNG/JPEG, stream page).
  - `geometry.py` — body world AABBs and the VLABench `contains` check.
  - `motion.py` — actuator-target math for `nudge` / `move_ee`.
  - `ik.py` — damped-least-squares IK: site/body targets, joint-limit clamping,
    null-space posture bias, collision checks, and `plan_path`.
  - `audit.py` — headless physics audit (penetration, settle, mass/inertia,
    limits, saturation, energy) returning a structured report.
  - `reel.py` — state recorder + multi-view video composition.
  - `vision.py` — camera -> detect -> control grasp servo.
- `python/tests/` — fast unit tests (no daemon, no GL) for the modules above.
- `python/robotics_daemon.py` — daemon entry point.
- `.pi/skills/mujoco/` — upstream MuJoCo agent skills (Apache-2.0).
- `.pi/skills/mujoco-audit/` — the physics-audit workflow skill.
- `.pi/skills/robotics-sim/` — the live-simulation workflow skill.

## Conventions

- The extension never edits project files. All runtime state lives under
  `<cwd>/.pi/robotics/` (git-ignored).
- Control and viewer ports are loopback only. Never bind them to a public
  interface, and never add authentication bypasses.
- Keep the `robotics` tool a single action-based tool; the action set is the
  public contract. Update `index.ts` help text, the README, and the skill
  together when it changes.
- The live viewer is shown by opening its loopback URL in the user's browser
  (`openInBrowser`); do not reintroduce a dashboard/canvas dependency — the
  MuJoCo WASM viewer cannot run in a sandboxed opaque-origin preview iframe.
- Keep `scenes.py` self-contained (XML strings, no `MjSpec` internals) so the
  exact model the agent reads is the model the daemon compiles.
- Never vendor large MuJoCo Menagerie meshes into the repo. `menagerie.py`
  fetches them on demand into `<cwd>/.pi/robotics/models/` (git-ignored) with a
  sparse, blobless `git` checkout. Add new robots to its `MODELS` map.

## Checks

```bash
npm run typecheck                       # tsc over the extension
node scripts/test-extension.mjs         # loads the extension and exercises the tool
.venv/bin/python -m pytest              # fast Python unit tests (no daemon, no GL)
```

`test-extension.mjs` needs `PI_ROBOTICS_PYTHON` to point at a Python with
`mujoco.studio` (or a project `.venv` discovered automatically).

The Python unit tests are pure: they compile MJCF and exercise `scenes`,
`ik` (site + body + tendon mapping + limits/collision/planning), `geometry`,
`motion`, `vision` (EE/gripper resolution), `audit` (penetration/settle/limits),
`control` (`/state` fields + CORS), and `reel` with no viewer.
Install the dev dependencies once with
`uv pip install --python .venv/bin/python -r python/requirements-dev.txt` (or
`pip install -r python/requirements-dev.txt`). `pytest.ini` puts `python/` on
`sys.path`, so run them from the repo root.

## Python environment

`mujoco.studio` and `mujoco_studio_wasm` require `mujoco>=4.0.1` (Python 3.11+).
The extension discovers a project `.venv`, `~/.venvs/pi-robotics`, or
`PI_ROBOTICS_PYTHON`; the `robotics` tool's `setup` action installs it.
