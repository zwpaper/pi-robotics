---
name: robotics-sim
description: >-
  Drives the pi-robotics MuJoCo simulation through the `robotics` tool: start
  and stop the daemon, load or build scenes, drive actuators and joints, step
  and reset physics, read qpos/qvel/sensors/contacts, capture frames, and show
  the live 3D scene on the pi-dashboard canvas. Use when the user asks to
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
  via a WASM Filament client; this is what the pi-dashboard canvas embeds.
- a **control port** — the JSON API the `robotics` tool calls.

Only one daemon runs per project. It is discovered through
`<cwd>/.pi/robotics/session.json`; runtime data (captures, logs) stays under
`<cwd>/.pi/robotics/`.

## Workflow

1. **Start** — `robotics` action `start`, optionally `{ scene: "arm3" }`. This
   boots the daemon, waits for the viewer, and declares the live 3D scene on the
   pi-dashboard canvas (a server chip the user taps to embed). The declaration is
   made lazily too: the first `state`/`step`/`ctrl`/`scene`/`load` of a session
   also declares it, so the panel is always one tap away.
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
5. **Open the panel** — `open_panel` re-declares the live scene on the canvas if
   the chip was dismissed.

Always pair a physical-sounding claim with a measured number: call `state`
after `step`/`ctrl` and report the returned `time`/`qpos`, never an estimated
one.

## Reading and writing the live panel

`start` and `open_panel` call the pi-dashboard `canvas` tool with a loopback
port. The dashboard shows a **server chip**; the user taps it to embed the 3D
view. The port is the extension's **shim proxy** (`panel-shim.ts`), not the raw
viewer: the proxy injects a shim so the viewer's root-absolute `/model`,
`/state`, `/ui`, and `/drop` requests survive pi-dashboard's `/live/<id>/` path
proxy. If the canvas tool is unavailable (pi running outside the dashboard), the
tool returns the URL so the user can open it directly.

Commands: `/robotics start [scene]`, `/robotics open`, `/robotics scenes`,
`/robotics scene <name>`, `/robotics capture`, `/robotics stop`, `/robotics
status`. `/robotics open` (and `scene`, and `start`) ask the agent to call the
`open_panel` tool, since command handlers cannot call the canvas tool directly.

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
