---
name: robotics-sim
description: >-
  Drives the pi-robotics MuJoCo simulation through the `robotics` tool: start
  and stop the daemon, load or build scenes, drive actuators and joints, step
  and reset physics, read qpos/qvel/sensors/contacts, capture frames, and show
  the live 3D scene in the browser. Use when the user asks to
  simulate a robot in MuJoCo, test a controller in simulation, build a physics
  scene, or show a live 3D robotics panel. Don't use for pure
  rendering-only requests, non-MuJoCo simulators, or real-hardware wiring.
metadata:
  author: pi-robotics
  version: "0.1.0"
---

# Live MuJoCo simulation with the `robotics` tool

This skill covers the simulation loop and the live 3D panel. For MuJoCo model
authoring and Python/C++ API details, load the bundled `mujoco-*` skills
(`mujoco-python`, `mujoco-spec-editing`, `mujoco-rendering`, `mujoco-gui`,
`mujoco-studio`, `mujoco-accelerated`, `mujoco-parity`, `mujoco-deformables`).

## Mental model

One background Python process (the *daemon*) owns the simulation and the 3D
viewer. It exposes two loopback ports:

- a **web-viewer port** — MuJoCo Studio, rendering the live scene in the browser
  via a WASM Filament client; the extension opens it in the user's browser.
- a **control port** — the JSON API the `robotics` tool calls.

Only one daemon runs per project. It is discovered through
`<cwd>/.pi/robotics/session.json`; runtime data (captures, logs) stays under
`<cwd>/.pi/robotics/`.

## Workflow

1. **Start** — `robotics` action `start`, optionally `{ scene: "arm3" }`. This
   boots the daemon, waits for the viewer, and opens the live 3D scene in the
   user's default browser.
2. **Author** — switch scenes with `scene { name, params }` (built-ins:
   `arm3`, `box_drop`, `cartpole`, `falling_objects`, `table_task`, `empty`;
   Menagerie robots: `stretch`, `stretch3`) or load a file with `load { path }`.
   To iterate on a scene, edit an MJCF file and `load` it again; the viewer
   hot-swaps the model.
3. **Act** — `ctrl { name, value }` or `ctrl { values: [...] }` to drive
   actuators, `qpos { values }` to set joints directly, `pause { paused }`,
   `speed { value }`, `step { n }`, `reset`.
4. **Observe** — `state` returns `time`, `qpos`, `qvel`, `ctrl`, `ncon`, site
   positions, and model metadata (joint/actuator names). `capture { width,
   height }` saves an offscreen PNG.
5. **Vision grasp loop** — to grasp by looking, run the `look`/`nudge` loop in
   "Vision grasp loop" below (prefer a subagent — see that section). Every
   `look`/`nudge` is also recorded for the operation reel.
6. **Operation video** — `reel` stitches the recorded 3D-panel + camera frames
   into a video (see "Operation video").
7. **Show the panel** — `open_panel` reopens the live scene in the browser if it
   was closed.

Always pair a physical-sounding claim with a measured number: call `state`
after `step`/`ctrl` and report the returned `time`/`qpos`, never an estimated
one.

## Showing the live panel

`start` and `open_panel` open the live scene in the user's **default browser** at
`http://127.0.0.1:<viewer_port>/`. Tell the user to keep that tab open and watch
it; it updates live as the agent drives `ctrl`/`step`/`scene`. To view from
another machine, port-forward (e.g. `ssh -L 8080:127.0.0.1:8080 <agent-host>`)
and open `http://localhost:8080/`.

Commands: `/robotics start [scene]`, `/robotics open`, `/robotics scenes`,
`/robotics scene <name>`, `/robotics pick [object]`, `/robotics vision [object]`,
`/robotics camera [name]`, `/robotics cameras`, `/robotics capture`,
`/robotics stop`, `/robotics status`.

## Hello Robot Stretch and `pick`

`scene { name: "stretch" }` loads the real MuJoCo Menagerie Hello Robot
Stretch 2 (`scene.xml`: robot + table + object). It is fetched on first use into
`<cwd>/.pi/robotics/models/` (sparse, blobless `git` checkout, ~75 MB);
`fetch { scene: "stretch" }` pre-downloads without loading.

`pick { object }` (default `object`) runs a calibrated scripted grasp on the sim
thread — settle, approach, close gripper, lift, raise — using the model's real
actuators (`lift`, `arm_extend`, `wrist_yaw`, `grip`), resuming the sim if it was
paused. It reports the measured lift, e.g.
`{ done: true, start_z: 0.52, end_z: 0.92, lifted: 0.40 }`. `reset` puts the
object back. For a different object or placement, drive the arm yourself with
`ctrl` on those actuators and read `state`.

## Vision grasp loop (`look` → decide → `nudge`) — the model sees the camera

`look { camera }` renders the robot camera and returns it as an **image in the
tool result**, so the model actually sees the scene. `nudge { extend, lift,
wrist, grip, settle }` applies **relative** actuator changes, lets the arm
settle, and returns a **fresh image** plus the current actuator values. `clear`
releases the manual hold; `reset` returns the object.

The loop the model runs:

1. `look` — see where the gripper and the object are.
2. `nudge` — move a little (`extend` toward the object, `lift` to the object's
   height, `wrist` to aim; the image and `ctrl` come back).
3. repeat 1–2 until the gripper is over the object.
4. `nudge { grip: -0.005 }` to close; `nudge { lift: 0.3 }` to raise.
5. `clear` to release the manual hold.
6. `reel` to build the operation video (see "Operation video").

**Stop as soon as the task succeeds.** Once it succeeds, do NOT keep operating
or retrying: confirm with one `look` (or the `contains` check below), then
`record_stop` and `reel`. The successful attempt is the **last** chapter; keep
the earlier failed attempts before it. Only run more attempts if the task failed,
or if the user explicitly asked for several candidate strategies.

For a VLABench-style placement task, `contains { object, container }` (POST
`/command {"cmd":"contains",...}`) reports `contained: true|false` — the
VLABench `contain` condition (the object's centre over the container's
footprint) — so success does not have to be guessed from pixels.

### Prefer a subagent for this loop

This loop is long (many image-bearing tool calls) and self-contained. **If
subagents are available, delegate it to one** — keep the image-heavy iteration
out of the main conversation (a real run here used ~8 frames and several
retries):

- Spawn a subagent (e.g. the pi-subagents tool) with a task like: *"In this
  project the pi-robotics daemon is running the Stretch scene. Run a
  `look`/`nudge` vision loop to pick up the tabletop object, save every frame,
  then report the sequence and the final result."*
- Give the subagent the cwd and the goal; it owns `look`/`nudge`/`grip`/`lift`
  and returns a short summary (plus the frame files it saved).
- A subagent may not have this project's `robotics` tool loaded. Give it the
  fallback: drive the daemon over loopback HTTP — read `control_port` from
  `<cwd>/.pi/robotics/session.json`, then `POST /command`
  `{"cmd":"look","camera":"camera_rgb"}` and
  `{"cmd":"nudge","extend":…,"lift":…,"wrist":…,"grip":…}`; decode
  `png_base64` into a file and inspect it. `capture`/`/capture` saves a PNG too.
- Only run the loop inline if no subagent tool is available, or the user asks to
  watch each step in the main thread.

## Operation video (3D panel + all cameras, as a grid) — independent capability

Whenever recording is active, the daemon captures the **3D-panel overview plus
every camera in the model**, one step per frame. This is independent of the task:
any run can be turned into a video.

- `record_start { fps, height, cols }` — **continuous** capture of the 3D panel
  + **all** model cameras (it does not restrict to one camera) at `fps`; the reel
  is then a real video of the whole operation.
- `chapter { title }` — marks a new **attempt**; the reel gets a QuickTime
  **chapter track** so each attempt is a chapter (use it for every try, not just
  the best one — keep the failed attempts too).
- `record_stop`, then `reel { height, cols, reel_name, hold }` — composes the
  frames into an MP4. Tiles are laid out in a **grid** (3D panel first, then
  every camera; `cols` defaults to a square-ish arrangement), encoded
  `libx264 -crf 18`, and the chapter track is attached. `reel_clear` starts a new
  take.
- The **LLM never sees the 3D panel image**: `look`/`nudge`/`move_ee` return only
  the robot **camera** image(s) unless `panel: true` is passed, but the panel is
  always captured for the video. (Camera-only input matches real hardware; the
  panel is a free-camera render that no real robot has.)

Layout rule: the 3D panel is tile 0, then the cameras in first-seen order; one
camera → one tile, several → one tile each. A camera that first appears later
gets a blank tile in earlier frames.

With the reel, a subagent's run is self-documenting: the daemon records while
the child drives the robot; afterwards either the child or the parent can `reel`
(the recorder lives in the daemon). Typical finish: `record_stop` →
`reel { fps: 10, height: 480 }` → report the video path.

Rules for the loop: never claim a pose without a fresh `look` (or `state`);
`nudge` values are deltas, `grip` is absolute (0.04 open, -0.005 closed);
needs `imageio` + `pillow`. Slower/coarser than `vision_pick` — use this when the
model must perceive and decide, and `vision_pick` for a fast autonomous grasp.

## Vision-based grasping (closed loop)

`vision_pick { object, camera }` is the perception-driven version, for when the
object has been moved. It runs in the daemon (no LLM in the control path):

1. render the robot camera with MuJoCo's segmentation renderer;
2. detect the target body's image centroid and back-project it onto the tabletop
   plane (`plane_z`, default 0.52 m) to get its world position;
3. measure the gripper fingertip by forward kinematics and servo `arm_extend`
   (y) and `lift` (z) until the fingertip reaches the object;
4. close the gripper and lift.

It reports `target` (camera-localized object), `aligned`, `iterations`, and
`lifted`. `vision_stop` cancels a running servo. It assumes the target is on the
table plane and in the camera's field of view.

## Robot camera stream

`cameras` lists the model's cameras; `stream { camera, width, height, fps }`
serves the chosen camera as a live MJPEG video and opens the player page
`http://127.0.0.1:<control_port>/camera` in the browser. On Stretch the head
RealSense gives `camera_rgb` and `camera_depth`; the daemon points the head at the
workspace so the video shows the gripper and object. The raw stream URL is
`/stream.mjpg?camera=<name>&width=&height=&fps=` and plays in any `<img>`. Frames
are rendered offscreen from the published state snapshot; `capture` saves a
single frame to PNG.

## Authoring scenes

Write scenes as MJCF XML in the project (for example `scenes/my_scene.xml`) and
`load` them, or use `scene` params for the built-ins. Keep one file per scene so
edits are reviewable and hot-reloadable. For programmatic scene construction use
`mujoco.MjSpec` per the `mujoco-spec-editing` skill.

## Safety

The daemon binds loopback only and has no authentication or encryption. Do not
expose its ports on a network. The MuJoCo *web* viewer ships no safety envelope
for real hardware; this plugin drives simulation only. Real-robot bring-up needs
separate, explicit safety review (limits, e-stop, watched workspace) before any
torque is enabled.
