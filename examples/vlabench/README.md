# VLABench example

A `select_mahjong` scene from [OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench)
(ICCV 2025), the language-conditioned manipulation benchmark.

## Where VLABench keeps tasks and scenes

- **Tasks** — `VLABench/tasks/`, registered by name. Primitive tasks live in
  `hierarchical_tasks/primitive/` (e.g. `select_mahjong_series.py`), composite
  ones in `hierarchical_tasks/composite/` (e.g. `play_mahjong.py`). Each task
  subclasses `PrimitiveTask` and registers a `BenchTaskConfigManager` with
  `@register.add_config_manager("<name>")`; the manager builds a config whose
  `components` are entities (object xml + pose) and whose `conditions` define
  success. `VLABench/configs/task_config.json` lists the task families and
  `configs/constant.py`'s `name2class_xml` maps an entity (e.g. `mahjong`) to its
  asset xml (`obj/meshes/mahjong/mahjong.xml`).
- **Scenes/robots** — `VLABench/envs/dm_env.py` (`LM4ManipDMEnv`, a
  `dm_control.composer.Environment`) assembles the robot + objects. Robot mounts
  live in `VLABench/configs/robot_config.json`; the **franka** entry is
  `position [0,-0.4,0.78]`, `euler [0,0,1.57]` and
  `default_qpos [0,-1.04,0,-2.56,0,1.55,-2.32,0.04,0.04]`. The robot assets ship
  in git (`VLABench/assets/robots/franka_emika_panda/`); the **object** assets
  (incl. the mahjong mesh) come from `scripts/download_assets.py`, which pulls
  `obj.zip` + `scene.zip` from Google Drive into `VLABench/assets/` (set
  `VLABENCH_ROOT` first). Simulator: **MuJoCo + dm_control** (single/dual arm,
  humanoid).
- **Demonstrations** — the Hugging Face dataset
  [VLABench/vlabench_primitive_ft_lerobot_video](https://huggingface.co/datasets/VLABench/vlabench_primitive_ft_lerobot_video)
  is *data*, not scenes: LeRobot **v3.0**, `robot_type: panda`, **10 fps**,
  5000 episodes / 575k frames, 128 language tasks, features `image`,
  `second_image`, `wrist_image` (480x480 video), `state` (7), `actions` (7).
  `meta/tasks.parquet` maps `task_index` -> instruction; `meta/info.json` gives
  fps/chunks; the top-level `eval.yaml` lists the 10 primitive tasks
  (`select_mahjong`, `select_poker`, `select_fruit`, `add_condiment`, ...).

So: **scenes/tasks = the VLABench code repo (+ its asset download); demonstrations
= the HF LeRobot datasets**. This example reproduces the `select_mahjong` scene
as a standalone MJCF for the pi-robotics daemon.

## This example

`models/franka_emika_panda/` is vendored from the VLABench repo (Apache-2.0):
`panda.xml` + `hand.xml` + meshes. `panda_mounted.xml` is `panda.xml` with the
robot mounted on the table like `robot_config.json`. `scene_select_mahjong.xml`
adds the work table (top `z = 0.78`), a row of 7 mahjong tiles (`tile_<value>_<suite>`,
suites man/pin/sou, faces coloured by suit), a tray, and cameras
(`top_cam`, `front_cam`, plus the panda's `Franka_wrist_cam`). The VLABench
instruction is `"Pick up the mahjong of 3 sou"` -> put `tile_3_sou` in the tray.

```text
/robotics load examples/vlabench/models/franka_emika_panda/scene_select_mahjong.xml
```

(mahjong tile meshes are not in the VLABench git repo — they need the Google
Drive `obj.zip` — so the tiles here are modelled as boxes with suit-coloured
faces, keeping the `value_suite` naming and layout.)
