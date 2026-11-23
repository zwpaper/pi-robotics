# Aloha Sim task scenes

Task scenes from [google-deepmind/aloha_sim](https://github.com/google-deepmind/aloha_sim)
(Apache-2.0): the ALOHA 2 bimanual manipulator with three household tasks.

Aloha Sim tasks are **dm_control `composer` environments**, not standalone MJCF
files, so they can't be loaded directly. `export_scenes.py` creates each task,
takes its composed `mjcf_model`, and writes it out with all meshes/textures
(dm_control's `export_with_assets`), deduplicating the shared assets into one
directory so the scenes load offline with plain `mujoco`.

## Tasks

| task | objects |
|---|---|
| `DiningPlaceBananaInBowl` | put the banana in the bowl |
| `ToolsPlaceCanOpenerInLeftCompartment` | put the can opener in the caddy's left compartment |
| `MarkerRemoveLid` | remove the cap from the marker |

Every scene is a bimanual ALOHA 2: 14 actuators (left/right × waist, shoulder,
elbow, forearm_roll, wrist_angle, wrist_rotate, gripper) and 6 cameras
(`overhead_cam`, `worms_eye_cam`, `wrist_cam_left`, `wrist_cam_right`, …).

## Regenerate the scenes

The scenes are built from the upstream assets by the exporter (no meshes are
vendorized here — the output is ~70 MB):

```bash
git clone https://github.com/google-deepmind/aloha_sim /tmp/aloha_sim
uv pip install --python .venv/bin/python "dm-control>=1.0.31" dm-env immutabledict mediapy
PYTHONPATH=/tmp/aloha_sim MUJOCO_GL=disable \
  .venv/bin/python examples/aloha_sim/export_scenes.py \
  --aloha-sim /tmp/aloha_sim \
  --out .pi/robotics/models/aloha_sim
```

Then load one (the daemon must **start with** the scene — hot-swapping from a
different model can abort the sim loop on this model's very high contact count):

```bash
# stop any running daemon first
python python/robotics_daemon.py \
  --model "$PWD/.pi/robotics/models/aloha_sim/MarkerRemoveLid/scene.xml" \
  --runtime-dir .pi/robotics/runtime --session-file .pi/robotics/session.json
# or, from an agent session, /robotics start <scene> once added to the scene list
```

## Notes

- Aloha Sim ships **no policy**. Its own `viewer.py`/`run_eval.py` drive the robot
  with Gemini Robotics (Trusted Testers) via the `gdm-robotics`/`safari-sdk`
  stack, which is not part of this repo. Driving the tasks yourself means
  low-level control (the `robotics` tool's `move_ee` IK / `nudge`), which is what
  the `google/gemini-4-hydrogen` subagent attempts.
- Fluids/no-policy caveats aside, these scenes are good for visualization, task
  setup replay, and as a target for control experiments.
