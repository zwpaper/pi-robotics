# pi-robotics

MuJoCo robotics simulation with a **live 3D panel**, packaged as a
[pi](https://github.com/earendil-works/pi) extension.

Your agent writes and runs a controller against a real physics engine, and you
watch the same scene update live in your **browser** — no desktop window, no GPU
on the agent host (the browser renders with MuJoCo Studio's WASM Filament
client).

This is a pi-native re-implementation of the simulation surface of
[`google-deepmind/robotics-antigravity-plugin`](https://github.com/google-deepmind/robotics-antigravity-plugin),
built directly on [`google-deepmind/mujoco`](https://github.com/google-deepmind/mujoco).

<p align="center">
  <em>pi agent ─▶ robotics tool ─▶ control API ─▶ MuJoCo daemon ─▶ Studio web viewer ─▶ browser tab</em>
</p>

## What you get

- **`robotics` tool** — start/stop a simulation, switch or load scenes, drive
  actuators and joints, step/reset physics, read `qpos`/`qvel`/sites/contacts,
  and capture frames.
- **Live 3D panel** — the daemon runs MuJoCo Studio in headless web mode and
  the extension opens it in your default browser (`/robotics start`). The same
  physics state the agent drives is what you see.
- **Bundled MuJoCo skills** — the upstream `mujoco-python`,
  `mujoco-spec-editing`, `mujoco-rendering`, `mujoco-gui`, `mujoco-studio`,
  `mujoco-accelerated`, `mujoco-parity`, and `mujoco-deformables` skills, so the
  agent authors correct MuJoCo code.
- **Parametric scenes** — `arm3`, `box_drop`, `cartpole`, `falling_objects`,
  `table_task`, `empty`, plus any MJCF / URDF / MJB file.

## How it maps to the two upstream projects

| Concern | `robotics-antigravity-plugin` | this repo |
|---|---|---|
| Agent surface | Antigravity skills + sidecar | pi `robotics` tool + pi skills |
| Live 3D viewer | `mujoco_studio` sidecar (Node) + `studiod.py` | MuJoCo Studio web viewer in the browser |
| Physics | `mujoco` (Python) | `mujoco` (Python) |
| Scene authoring | `sim_plugin.py` / `scene.xml` | `python/pi_robotics/scenes.py` + `load` |
| Real hardware | SO-101 / YAM / FR3 skills | out of scope (sim first) |

The MuJoCo skills are the same files the Antigravity plugin bundles, taken from
`mujoco/doc/skills` (Apache-2.0).

## Requirements

- pi and Node 18+ (any browser for the 3D panel)
- Python 3.11+ with `mujoco>=4.0.1` (ships `mujoco.studio` and
  `mujoco_studio_wasm`)
- optional: `uv` (faster env setup)

## Setup

From the project root:

```bash
# 1. install the pi extension + skills (auto-discovered from .pi/)
#    nothing to do for local use — pi loads .pi/extensions and .pi/skills.

# 2. create the Python environment with MuJoCo Studio
uv venv .venv
uv pip install --python .venv/bin/python -r python/requirements.txt
```

Or let the agent do it: call the `robotics` tool with `action: "setup"`.

Interpreter discovery order: `$PI_ROBOTICS_PYTHON`, `./.venv`, `~/.venvs/pi-robotics`,
then `python3`/`python`.

## Usage

In a pi session in this project:

```text
/robotics start arm3          # boot the sim and open the live 3D panel
/robotics open                # (re)show the live 3D panel
/robotics scenes              # list built-in scenes
/robotics scene cartpole      # switch scene
/robotics capture             # save a frame
/robotics status
/robotics stop
```

Or via the tool (the agent does this automatically):

```jsonc
{ "action": "start", "scene": "arm3" }
{ "action": "ctrl", "name": "joint0", "value": 0.8 }
{ "action": "state" }                       // time, qpos, qvel, ctrl, sites, ncon
{ "action": "step", "n": 50 }
{ "action": "scene", "name": "cartpole", "params": { "pole_length": 1.2 } }
{ "action": "load", "path": "scenes/my_scene.xml" }
{ "action": "capture", "width": 640, "height": 480 }
{ "action": "open_panel" }                  // alias: "panel"
```

### Viewing the 3D panel

`start` and `open_panel` open the live scene in your **default browser** at
`http://127.0.0.1:<viewer_port>/` (macOS `open`, Windows `start`, Linux
`xdg-open`). You get the full interactive MuJoCo Studio viewer (orbit, pan,
inspect, play/pause) at 60 FPS, driven by the same daemon the agent controls.
Keep the tab open while the agent drives `ctrl`/`step`/`scene` — the view updates
live.

Only the browser needs to reach the loopback port. To view the panel from
another machine, forward it, e.g. `ssh -L 8080:127.0.0.1:8080 <agent-host>` and
open `http://localhost:8080/`.

## Architecture

```text
pi session
 ├─ .pi/extensions/robotics/index.ts     robotics tool + /robotics command
 │     ├─ daemon.ts ──spawn/stop──┐
 │     ├─ client.ts ──HTTP────────┼──────────────┐
 │     └─ openInBrowser ──▶ browser tab          │
 │                                ▼              ▼
 └─ python/robotics_daemon.py ── mujoco.studio.launch_web
       ├─ RoboticsSimPlugin   builds scenes, applies ctrl/qpos, publishes state
       ├─ step_control / sim_history / studio App (interactive UI)
       ├─ PortCapturePlugin   records the bound web-viewer port
       └─ control.py          loopback JSON API + shared state
             ├─ GET  /status /state /scenes /scene
             └─ POST /command /scene /load /capture /shutdown
```

The viewer serves the page, `/ui`, `/state`, `/model`, and the WASM client on a
single loopback port; the agent talks to the Python daemon over the separate
control port.

## Notes and limitations

- Loopback only, **no auth or encryption**. Never expose the ports.
- This plugin drives MuJoCo **simulation**, not real hardware. Real-robot
  bring-up needs a separate safety review.
- `capture` needs `imageio` + `pillow` and a working GL context; the live viewer
  does not.
- Offscreen/native rendering (`mujoco.Renderer`, `mujoco.viewer`) is not used by
  the panel path.

## Layout

```text
pi-robotics/
├── package.json                     # pi manifest (extension + skills)
├── .pi/
│   ├── extensions/robotics/         # the pi extension
│   └── skills/                      # mujoco-* + robotics-sim skills
├── python/
│   ├── robotics_daemon.py           # daemon entry point
│   ├── requirements.txt
│   └── pi_robotics/                 # daemon, scenes, control server
├── examples/scenes/                 # example MJCF
└── scripts/test-extension.mjs       # headless extension smoke test
```

## License

Apache-2.0. MuJoCo skills are © Google LLC (Apache-2.0), copied from
`google-deepmind/mujoco/doc/skills`.
