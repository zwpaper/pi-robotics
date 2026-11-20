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
   `arm3`, `box_drop`, `cartpole`, `falling_objects`, `table_task`, `empty`) or
   load a file with `load { path }`. To iterate on a scene, edit an MJCF file
   and `load` it again; the viewer hot-swaps the model.
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
`/robotics scene <name>`, `/robotics capture`, `/robotics stop`, `/robotics
status`.

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
