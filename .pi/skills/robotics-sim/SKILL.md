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
5. **Show the panel** — `open_panel` reopens the live scene in the browser if it
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

## LLM-in-the-loop vision (model sees the camera and drives)

`look { camera }` renders the robot camera and returns it as an **image in the
tool result**, so the model actually sees the scene. `nudge { extend, lift,
wrist, grip, settle }` applies relative actuator changes, lets the arm settle, and
returns a **fresh image** plus the current actuator values. The model runs the
loop itself:

1. `look` — see where the gripper and object are.
2. `nudge` — move a little.
3. repeat until the gripper is over the object.
4. `nudge { grip: -0.005 }` to close, `nudge { lift: 0.3 }` to raise.
5. `clear` to release the manual hold.

This is the model perceiving and deciding every step; it needs `imageio` +
`pillow`. Slower and coarser than `vision_pick`; use it when you want the model
in the loop, and `vision_pick` for a fast autonomous grasp.

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
