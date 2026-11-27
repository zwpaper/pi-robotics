# Joinery (榫卯) workspace example

A **Live Python Studio workspace** — the authoring mode where the scene is
editable Python that hot-reloads, rather than a built-in `scenes.py` scene or a
hand-written MJCF. It builds a small 榫卯 (Chinese joinery) sandbox with two
classic joints:

- **直榫 (straight tenon)** — a shoulder block whose square tenon drops into a
  socket cut into a fixed beam; the shoulder rests on the beam top.
- **十字搭接 (cross-lap)** — two beams, each notched to half depth, interlocking
  flush when the upper one is lowered into the lower one.

Every loose part is a real rigid body (gravity, contact, friction, press fit), so
it can be dragged, dropped, jammed and re-seated. A kinematic **mocap hand**
grasps by an equality weld: the part is held rigidly but keeps its own mass, so
it still collides — drag the hand near wood and it closes on it.

Videos are **not** committed: `record_start` + `reel` write to the git-ignored
`.pi/robotics/runtime*/reels/`.

## Layout

```text
examples/joinery_workspace/
├── messages.py        # shared Snapshot / Event classes (sim <-> UI)
├── sim_plugin.py      # plugin.SimPlugin: builds the scene with mujoco.MjSpec
└── viewer_plugin.py   # plugin.ViewerPlugin: the Dear ImGui panel
```

`sim_plugin.py` also adds four cameras, which is what makes `reel` multi-angle
(the grid is the 3D panel plus every camera):

| camera | view |
|---|---|
| `cam_iso` | overview of both joints |
| `cam_tenon` | straight-tenon socket, front |
| `cam_lap` | cross-lap, front |
| `cam_top` | top-down |

## Running it

Point the daemon at this directory (or copy the three files into
`<cwd>/studio_workspace`):

```bash
PI_ROBOTICS_WORKSPACE_DIR=examples/joinery_workspace /robotics start
```

Then the ImGui panel (`pi-robotics joinery`) appears inside the 3D viewer at the
reported loopback URL, next to the scene.

## The panel

- `Gravity (m/s²)` and `Wood friction` — applied live to the running model.
- `Hand X / Y / Z` — move the gripper with left-clicks only (the viewer's own
  left-drag **rotates** the hand; translation is Ctrl/Cmd + **right**-drag).
- `Grab nearest` / `Open hand` — snap the hand onto the nearest part and close,
  or let go. The hand also closes by itself when it comes near wood.
- `Reset / drop everything`, `Drop tenon`, `Bake to scene.xml`.

## Recording a multi-angle clip

```text
/robotics record_start {fps:15}
... drive the joints (see below) ...
/robotics reel {reel_name:"joinery", cols:3}
```

`reel` condenses runs of still frames longer than `idle_cap` (default 1 s) and
also writes the untouched timeline as `<name>_raw.mp4`.

## Driving a joint from the agent

`nq = 21`; `qpos` is three `[x,y,z,qw,qx,qy,qz]` triples — `tenon` (0..6),
`lap_a` (7..13), `lap_b` (14..20). Pass `duration` to move smoothly instead of
teleporting:

```jsonc
// separate lap_b, then bring it back so the beams re-interlock
{ "action": "qpos", "duration": 1.6,
  "values": [-0.34,0,0.05,1,0,0,0, 0.36,0,0,1,0,0,0, 0.36,0.30,0.16,1,0,0,0] }
{ "action": "qpos", "duration": 1.6,
  "values": [-0.34,0,0.05,1,0,0,0, 0.36,0,0,1,0,0,0, 0.36,0.0,0.14,1,0,0,0] }
```

Only the entries that change are interpolated, so the rest of the scene keeps
simulating. `bake` freezes the result to `scene.xml` for a plain-MJCF hand-off.
