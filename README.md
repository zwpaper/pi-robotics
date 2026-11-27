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

## Showcase

### Removing a marker cap: the agent fails three times, then finds the trick

<!--
  Placeholder link. Replace `src` with the uploaded video URL (a GitHub Release
  asset works well, e.g. https://github.com/<owner>/<repo>/releases/download/<tag>/<file>.mp4).
-->
<p align="center">
  <video src="https://github.com/zwpaper/pi-robotics/releases/download/demos/marker_remove_lid.mp4"
         controls muted playsinline width="760"></video>
</p>

A bimanual ALOHA 2 (`examples/aloha_sim`) has to pull the cap off a marker that
is standing on the table. The cap is a tight friction fit, so this is a
**grip-and-pull** problem, not a reach problem — and an LLM agent driving the
`robotics` tool (`look` → `move_ee` / `nudge` → `contains`) needs **four
attempts** to solve it.

| # | what it tried | result |
|---|---|---|
| 1 | low-tension **axial pull** | the pull never beats the cap–body friction; the cap just slides in the fingers |
| 2 | lifted the marker **mid-air**, both arms pulling in **opposing** directions | nothing to brace against — the marker squirts out of the opposite gripper |
| 3 | **table-level pull + twist** | right idea, but no recovery once the fingers slip: the cap rotates on the spot and stays on |
| 4 | **high-tension opposing pull + peel + slip recovery** | grip hard, brace the marker on the table, **peel** sideways, and **re-seat the cap instead of pulling harder** when it slips — the cap pops off |

The video is the **full session, failed attempts included**: one QuickTime chapter
per attempt, with the MuJoCo 3D panel plus all six ALOHA cameras arranged in a
grid, rendered from the recorded state timeline. See the
[full write-up](examples/aloha_sim/videos/marker_remove_lid.md).

More demos: [TidyBot++ water-the-plant / fill-a-cup](examples/tidybot2/README.md),
[VLABench mahjong selection](examples/vlabench/README.md),
[joinery (榫卯) interactive workspace](examples/joinery_workspace/README.md).

https://github.com/user-attachments/assets/8f9a91fe-3751-4b96-bca0-a3ed46a3efa1

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
- **Hello Robot Stretch** — load the real MuJoCo Menagerie Stretch
  (`scene stretch`, fetched on first use) and run **`pick`** to grasp and lift the
  tabletop object.
- **Live Python Studio workspace** — `scaffold_workspace` writes editable
  `messages.py` / `sim_plugin.py` / `viewer_plugin.py`; MuJoCo Studio hot-reloads
  them on save and shows a **Dear ImGui panel in the browser viewer** so a human
  and the agent drive the same scene. `bake` freezes it to `scene.xml`.

## How it maps to the two upstream projects

| Concern | `robotics-antigravity-plugin` | this repo |
|---|---|---|
| Agent surface | Antigravity skills + sidecar | pi `robotics` tool + pi skills |
| Live 3D viewer | `mujoco_studio` sidecar (Node) + `studiod.py` | MuJoCo Studio web viewer in the browser |
| Physics | `mujoco` (Python) | `mujoco` (Python) |
| Scene authoring | `sim_plugin.py` / `scene.xml` | `python/pi_robotics/scenes.py` + `load` + the Live Python Studio workspace |
| Real hardware | SO-101 / YAM / FR3 skills | out of scope (sim first) |

The MuJoCo skills are the same files the Antigravity plugin bundles, taken from
`mujoco/doc/skills` (Apache-2.0).

## Requirements

- pi and Node 18+ (any browser for the 3D panel)
- Python 3.11+ with `mujoco>=4.0.1` (ships `mujoco.studio` and
  `mujoco_studio_wasm`)
- `git` (only to fetch the optional MuJoCo Menagerie robot models on first use)
- [`git-lfs`](https://git-lfs.com) — only for the example scenes' meshes and
  textures; the core tool works without it
- optional: `uv` (faster env setup)

## Setup

From the project root:

```bash
# 1. install the pi extension + skills (auto-discovered from .pi/)
#    nothing to do for local use — pi loads .pi/extensions and .pi/skills.

# 2. (example scenes only) pull the meshes/textures stored in Git LFS
#    a plain `git clone` already does this when git-lfs is on PATH
# git lfs install
# git lfs pull

# 3. create the Python environment with MuJoCo Studio
uv venv .venv
uv pip install --python .venv/bin/python -r python/requirements.txt
```

Or let the agent do it: call the `robotics` tool with `action: "setup"`.

Interpreter discovery order: `$PI_ROBOTICS_PYTHON`, `./.venv`, `~/.venvs/pi-robotics`,
then `python3`/`python`.

## Tests

Fast Python unit tests cover the model builders, IK, geometry/containment,
actuator math, and reel composition — no daemon, viewer, or GL context:

```bash
uv pip install --python .venv/bin/python -r python/requirements-dev.txt  # once
.venv/bin/python -m pytest                                              # ~0.2 s
npm run typecheck                                                       # extension
node scripts/test-extension.mjs                                         # live E2E smoke
```

`pytest.ini` puts `python/` on `sys.path`, so run the tests from the repo root.
The `test-extension.mjs` / `test-vision-loop.mjs` scripts are the slower live
end-to-end checks and need a working MuJoCo Studio Python.

## Usage

In a pi session in this project:

```text
/robotics start arm3          # boot the sim and open the live 3D panel
/robotics open                # (re)show the live 3D panel
/robotics scenes              # list built-in + Menagerie scenes
/robotics scene cartpole      # switch scene
/robotics scene stretch       # load the Hello Robot Stretch (fetches ~75 MB first time)
/robotics pick                # Stretch grasps and lifts the tabletop object (scripted)
/robotics vision              # camera -> detect -> control closed-loop grasp
/robotics camera              # play the robot camera as live video in the browser
/robotics cameras             # list the model's cameras
/robotics capture             # save a frame
/robotics workspace           # status of the live Python Studio workspace
/robotics bake                # freeze the current scene to scene.xml
/robotics status
/robotics stop
```

Or via the tool (the agent does this automatically):

```jsonc
{ "action": "start", "scene": "arm3" }
{ "action": "ctrl", "name": "joint0", "value": 0.8 }
{ "action": "qpos", "values": [0.0, 0.5, -0.5] }  // set joint positions directly
{ "action": "qvel", "values": [0.0, 0.0, 0.0] }     // set joint velocities directly
{ "action": "state" }                       // time, qpos, qvel, ctrl, ncon, sites, bodies, body_quat, sensordata, contacts
{ "action": "step", "n": 50 }
{ "action": "scene", "name": "cartpole", "params": { "pole_length": 1.2 } }
{ "action": "load", "path": "scenes/my_scene.xml" }
{ "action": "capture", "width": 640, "height": 480 }
{ "action": "open_panel" }                  // alias: "panel"
{ "action": "scene", "scene": "stretch" }     // load Hello Robot Stretch
{ "action": "pick", "object": "object" }     // scripted grasp + lift
{ "action": "vision_pick", "object": "object" } // camera->detect->control grasp
{ "action": "move_ee", "pos": [0.3, 0.0, 0.6], "posture": { "joint1": 0.2 } } // IK move, collision-checked
{ "action": "audit", "scene": "cartpole" }  // headless physics audit
{ "action": "stream", "camera": "camera_rgb" } // live robot-camera video
{ "action": "fetch", "scene": "stretch" }    // pre-download a Menagerie model
```

Menagerie scene names (fetched on first use into `<cwd>/.pi/robotics/models/`):
`stretch`, `stretch3`, `panda`, `franka_fr3`, `ur5e`, `aloha`, `so101`, `go2`,
`g1`. Switch to any of them with `scene { name: "panda" }`; `fetch { scene }`
pre-downloads one without loading it.

`state` is the physics debugger: besides `qpos`/`qvel`/`ctrl` it returns named
`sensordata` (touch, jointpos, …), `bodies` + `body_quat` (world pose of every
named body), the top `contacts` as `{geom1, geom2, normal, dist}` pairs (strongest
first), and model metadata with `ctrlrange`, `joint_limits`, and sensor names —
enough to see a slip happen directly in the numbers.

### Picking up an object with the Hello Robot Stretch

The `stretch` scene is the real [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
Hello Robot Stretch 2 with a table and an object in front (Menagerie
`scene.xml`). It is **not vendored** (the meshes are ~75 MB): on first use the
daemon does a sparse, blobless `git` checkout of just that directory into
`<cwd>/.pi/robotics/models/` and reuses it afterwards. `fetch` pre-downloads it.

`pick` runs a calibrated, closed-scene scripted grasp on the sim thread:
settle → approach the object → close the gripper → lift → raise. It uses the
model's real actuators (`lift`, `arm_extend`, `wrist_yaw`, `grip`), resumes the
sim if it was paused, and reports the measured lift:

```jsonc
// robotics { action: "pick", object: "object" }
{ "done": true, "object": "object", "start_z": 0.5199, "end_z": 0.9162, "lifted": 0.3963 }
```

The object starts at z≈0.52 on the table and ends held ≈0.92 by the gripper
(~0.40 m lift). `reset` returns it to the table. For a different object or
placement, drive the arm yourself with `ctrl` on `lift` / `arm_extend` /
`wrist_yaw` / `grip` and read `state`.

### Driving the grasp from the camera (LLM in the loop)

This is the *model-looks-and-acts* path. `look` renders the robot camera and
returns it as an **image the model actually sees**; `nudge` applies relative
actuator changes and returns a **fresh image**. The model runs the loop itself:

```jsonc
{ "action": "look", "camera": "camera_rgb" }                         // the model sees the scene
{ "action": "nudge", "extend": 0.15 }                                // move a little, get a new frame
{ "action": "nudge", "extend": 0.13, "lift": -0.07, "wrist": 0.15 }
{ "action": "nudge", "grip": -0.005 }                                // close the gripper
{ "action": "nudge", "lift": 0.3 }                                   // raise
{ "action": "clear" }                                                  // release manual control
```

Each `nudge` returns the new frame plus the current actuator values, so the
model can decide the next step from pixels. (`look`/`nudge` need `imageio` +
`pillow`.) This is genuinely the model in the control loop: slower and coarser
than `vision_pick`, which is why both exist — use `vision_pick` for a fast,
reliable grasp, and `look`/`nudge` when you want the model to perceive and
decide each step.

#### Operation video (3D panel + every camera, as a grid)

`record_start { fps }` captures the **3D-panel overview plus every camera in the
model**, continuously. `chapter { title }` marks each **attempt** (record every
try, including failed ones); `reel { height, cols }` composes them into a video
with the tiles in a **grid** (3D panel first, then every camera) and a
**QuickTime chapter track** for the attempts.

The **LLM never sees the 3D panel image** — `look`/`nudge`/`move_ee` return only
the robot **camera** image(s) (add `panel: true` to override), but the panel is
always captured for the video. Camera-only input is deliberate: it matches real
hardware, where no free-camera 3D view exists.

### Vision-based grasping (camera → detect → control)

`pick` is a fixed, calibrated sequence — it only works for the object at the
built-in position. `vision_pick` is a **closed loop** driven by the robot's own
camera:

1. **Camera** — renders `camera_rgb` offscreen with MuJoCo's **segmentation**
   renderer (per-geom ids, so detection is robust to colour and lighting),
   reading only the published state snapshot.
2. **Detect / localize** — takes the target body's image centroid and
   back-projects it through the camera model onto the tabletop plane
   (`plane_z`, default 0.52 m), giving the object's world position.
3. **Control** — uses MuJoCo inverse kinematics (`ik.py`) to drive a
   configurable end-effector — a named **site** (Panda/UR5e/FR3 `attachment_site`)
   or gripper **body** (Stretch ships no EE site, so it uses
   `link_gripper_slider`) — onto the object in waypoints, closes the auto-detected
   gripper actuator, then lifts with a second IK target. Fixed-tendon actuators
   (Stretch's `arm_extend`) are mapped too, so no actuator names are hard-coded.

```jsonc
// robotics { action: "vision_pick", object: "object", camera: "camera_rgb" }
{ "done": true, "aligned": true, "iterations": 4,
  "target": [-0.019, -0.449, 0.52], "lifted": 0.2614 }
```

The loop runs in the daemon (fast, no LLM in the control path); the tool action
reports the resolved `end_effector` (site or body) and `gripper`, `target`
(camera-localized object), the per-iteration error, and the measured lift.
`vision_stop` cancels it. Override the defaults with `site`, `gripper`, or the
other `vision_pick` parameters. It is robust to the object being moved, as long
as it stays on the tabletop plane and in the camera's view.

### Physics audit (`audit`)

`audit` compiles a model and runs a **headless** battery of checks — no GL, no
viewer, and it never touches the live sim, so it works while the panel is
running. It returns a structured report (`summary`, `model`, `sim`, `findings`)
the agent can fix iteratively:

- **initial penetration** at spawn and **resting** penetration after settling,
- **drop / settle stability** (NaN/Inf, residual velocity, energy growth),
- **mass / inertia sanity** (zero/negative mass, zero inertia, extreme ratio),
- **joint-limit hits**, **actuator force/control saturation**, and **energy
  blow-up**.

```jsonc
{ "action": "audit" }                          // the loaded model
{ "action": "audit", "scene": "cartpole" }     // a built-in/Menagerie scene
{ "action": "audit", "path": "scenes/my.xml" } // an MJCF/URDF/MJB file
```

`scene`/`path` compile a throwaway model, so candidates can be audited without
loading them. See the bundled **`mujoco-audit`** skill for how to interpret and
fix each finding.

### Collision-aware `move_ee`

`move_ee` plans a path of joint-limit-clamped IK waypoints (with an optional
null-space `posture` bias) and **stops before any new contact**, so it will not
drive an arm through a table or into itself. It returns `blocked: true` and the
`{geom1, geom2}` pairs it hit. Pass `allow_collision: true` to override, or
`site`/`body` to choose the end-effector (auto-detected otherwise).

### Playing the robot camera as live video

`cameras` lists the current model's cameras; `stream { camera, width, height,
fps }` serves the chosen camera as an **MJPEG video** and opens a player page in
the browser (`http://127.0.0.1:<control_port>/camera`). The Stretch head
RealSense provides `camera_rgb` and `camera_depth`; the daemon points the head at
the workspace so the stream shows the gripper and the object.

The stream is a plain `multipart/x-mixed-replace` response, so it plays in any
browser or `<img>` tag:

```html
<img src="http://127.0.0.1:<control_port>/stream.mjpg?camera=camera_rgb&width=480&height=360&fps=15">
```

The daemon renders each frame offscreen with `mujoco.Renderer` **from the
published state snapshot**, so it never reads live `MjData` while the sim thread
is stepping. Offscreen rendering needs `imageio` + `pillow` (installed by
`setup`) and a working GL context.

### Live Python Studio workspace (interactive authoring + hot-reload)

Every simulation above is *driven* by the agent. A **workspace** is the other
half: a small directory of Python files that build the scene and expose a
**Dear ImGui control panel rendered inside the browser 3D viewer** — sliders and
buttons a human can drag while the agent still drives the same simulation
through the control API.

```jsonc
{ "action": "scaffold_workspace" }                       // write + start it
{ "action": "workspace" }                                // status
{ "action": "bake" }                                     // freeze to scene.xml
```

The scaffold writes three files into `<cwd>/studio_workspace`
(override with `{ "dir": "..." }`, or the `PI_ROBOTICS_WORKSPACE_DIR` env var):

```text
studio_workspace/
├── messages.py       # shared Snapshot / Event classes (sim <-> UI)
├── sim_plugin.py     # plugin.SimPlugin: builds the scene with mujoco.MjSpec
└── viewer_plugin.py  # plugin.ViewerPlugin: the Dear ImGui panel
```

MuJoCo Studio loads these as plugins and **watches them**: save any of the three
and the running simulation is rebuilt in place — the same in-place class swap
Studio uses for its own plugins, with the camera where it was. The panel appears
in the viewer tab already open in your browser (no extra window or port).

- **Panel → sim** — a slider change sends a `SceneConfig` snapshot; the sim
  plugin rebuilds its `MjSpec` and calls `handle.reload()`.
- **Bake** — the panel's *Bake to scene.xml* button (or the `bake` action) writes
  `spec.to_xml()` to `<workspace>/scene.xml` and shows that file from then on.
  The daemon-level `bake` also works for the built-in Python scenes, serializing
  the compiled model with `mj_saveLastXML`.
- **Hot-reload** — enabled for the workspace plugin classes plus the shown model
  file, so editing `scene.xml` or a loaded MJCF also reloads live.

While a workspace owns the scene, built-in `scene` switching is refused (the
workspace built the model first); `bake` + `load`, or starting without the
workspace, switch models instead. `workspace` reports the directory, files,
loaded plugin classes, and any import errors.

```bash
/robotics workspace     # status of the live workspace
/robotics bake          # freeze the current scene to scene.xml
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
- `capture` and `stream` need `imageio` + `pillow` and a working GL context; the
  live 3D viewer does not.
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
│   ├── requirements.txt             # runtime deps (mujoco, imageio, pillow)
│   ├── requirements-dev.txt         # pytest (tests only)
│   ├── tests/                       # fast unit tests (no daemon, no GL)
│   └── pi_robotics/                 # daemon, scenes, control server + helpers
│       ├── control.py               # ControlServer: commands + `/state`
│       ├── server.py                # loopback HTTP handler + CORS allowlist
│       ├── rendering.py             # offscreen render / camera stream page
│       ├── geometry.py              # body AABBs + `contains` check
│       ├── motion.py                # nudge / move_ee actuator math
│       ├── ik.py · motion.py · audit.py · reel.py · vision.py · scenes.py · menagerie.py
├── examples/
│   ├── scenes/                      # example MJCF (peg_and_hole.xml)
│   ├── tidybot2/                    # TidyBot++ models + water_plant / fill_cup scenes
│   ├── aloha_sim/                   # ALOHA 2 task scenes (+ exporter) and demo write-ups
│   ├── vlabench/                    # VLABench Franka scenes (mahjong assets vendored)
│   └── joinery_workspace/           # 榫卯 joinery Live Python Studio workspace
└── scripts/test-extension.mjs       # headless extension smoke test
```

Example-scene meshes and textures are stored in **Git LFS**
(`.gitattributes` → `examples/**/*.{obj,stl,mtl,msh,png,jpg,jpeg}`), so a fresh
clone gets them automatically when `git-lfs` is installed; the `aloha_sim` and
`vlabench` models are committed this way because their assets are tens of MB.
Fetched Menagerie models live in `<cwd>/.pi/robotics/models/` (git-ignored).

## License

Apache-2.0. MuJoCo skills are © Google LLC (Apache-2.0), copied from
`google-deepmind/mujoco/doc/skills`.
