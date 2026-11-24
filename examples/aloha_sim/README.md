# Aloha Sim task scenes

Task scenes from [google-deepmind/aloha_sim](https://github.com/google-deepmind/aloha_sim)
(Apache-2.0): the ALOHA 2 bimanual manipulator with three household tasks.

Aloha Sim tasks are **dm_control `composer` environments**, not standalone MJCF
files, so they can't be loaded directly. `export_scenes.py` creates each task,
takes its composed `mjcf_model`, and writes it out with all meshes/textures
(dm_control's `export_with_assets`), deduplicating the shared assets into one
directory so the scenes load offline with plain `mujoco`.

The exported scenes are committed under `models/` — the meshes and textures are
stored in **Git LFS**, so a fresh clone gets them automatically when `git-lfs`
is installed (`git lfs install && git lfs pull` otherwise).

## Tasks

| task | objects |
|---|---|
| `DiningPlaceBananaInBowl` | put the banana in the bowl |
| `ToolsPlaceCanOpenerInLeftCompartment` | put the can opener in the caddy's left compartment |
| `MarkerRemoveLid` | remove the cap from the marker |

Every scene is a bimanual ALOHA 2: 14 actuators (left/right × waist, shoulder,
elbow, forearm_roll, wrist_angle, wrist_rotate, gripper) and 6 cameras
(`overhead_cam`, `worms_eye_cam`, `wrist_cam_left`, `wrist_cam_right`, …).

## Demos

- [`videos/marker_remove_lid.md`](videos/marker_remove_lid.md) — the agent fails
  three times on `MarkerRemoveLid`, then finds the trick (high-tension opposing
  pull + peel + slip recovery) and pulls the cap off. Full operation video, one
  QuickTime chapter per attempt. (The video file itself is not committed: reels
  live in the git-ignored `<cwd>/.pi/robotics/runtime*/reels/`.)

## Regenerate the scenes

The scenes under `models/` are **checked in** (via Git LFS); this exporter
reproduces them from the upstream package if you want to refresh or extend them:

```bash
git clone https://github.com/google-deepmind/aloha_sim /tmp/aloha_sim
uv pip install --python .venv/bin/python "dm-control>=1.0.31" dm-env immutabledict mediapy
PYTHONPATH=/tmp/aloha_sim MUJOCO_GL=disable \
  .venv/bin/python examples/aloha_sim/export_scenes.py \
  --aloha-sim /tmp/aloha_sim \
  --out examples/aloha_sim/models
```

Then load one (the daemon must **start with** the scene — hot-swapping from a
different model can abort the sim loop on this model's very high contact count):

```bash
# stop any running daemon first
python python/robotics_daemon.py \
  --model "$PWD/examples/aloha_sim/models/MarkerRemoveLid/scene.xml" \
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
