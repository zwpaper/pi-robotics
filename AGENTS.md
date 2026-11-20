# AGENTS.md

This repository is a **pi extension** that brings MuJoCo robotics simulation and
a live 3D panel to the pi coding agent. It mirrors the simulation surface of
`google-deepmind/robotics-antigravity-plugin` on top of `google-deepmind/mujoco`.

## Layout

- `.pi/extensions/robotics/` — the pi extension (TypeScript, loaded by jiti).
  - `index.ts` — the `robotics` tool + `/robotics` command.
  - `daemon.ts` — spawns/supervises the Python daemon.
  - `panel-shim.ts` — loopback proxy that makes the viewer embeddable in pi-dashboard.
  - `client.ts` — loopback control HTTP client.
  - `paths.ts` — interpreter discovery, runtime paths, session file.
- `python/pi_robotics/` — daemon, scenes, control server.
  - `daemon.py` — launches `mujoco.studio.launch_web` and the sim plugin.
  - `scenes.py` — parametric MJCF scenes.
  - `control.py` — loopback control API + shared state.
- `python/robotics_daemon.py` — daemon entry point.
- `.pi/skills/mujoco/` — upstream MuJoCo agent skills (Apache-2.0).
- `.pi/skills/robotics-sim/` — the live-simulation workflow skill.

## Conventions

- The extension never edits project files. All runtime state lives under
  `<cwd>/.pi/robotics/` (git-ignored).
- Control and viewer ports are loopback only. Never bind them to a public
  interface, and never add authentication bypasses.
- Keep the `robotics` tool a single action-based tool; the action set is the
  public contract. Update `index.ts` help text, the README, and the skill
  together when it changes.
- The panel is declared to pi-dashboard through the shim proxy port, never the
  raw viewer port: the dashboard embeds servers under a path prefix that the
  viewer's root-absolute URLs do not survive. Keep `panel-shim.ts` prefix-aware
  and header-stripping; it must stay transparent when opened top-level.
- Keep `scenes.py` self-contained (XML strings, no `MjSpec` internals) so the
  exact model the agent reads is the model the daemon compiles.

## Checks

```bash
npm run typecheck                       # tsc over the extension
node scripts/test-extension.mjs         # loads the extension and exercises the tool
node scripts/test-panel-proxy.mjs       # verifies the shim proxy against a live viewer
```

Both scripts need a live viewer (the tool script starts one). `test-extension.mjs`
needs `PI_ROBOTICS_PYTHON` to point at a Python with `mujoco.studio` (or a
project `.venv` discovered automatically).

## Python environment

`mujoco.studio` and `mujoco_studio_wasm` require `mujoco>=4.0.1` (Python 3.11+).
The extension discovers a project `.venv`, `~/.venvs/pi-robotics`, or
`PI_ROBOTICS_PYTHON`; the `robotics` tool's `setup` action installs it.
