# TidyBot++ example scenes

MuJoCo models from [jimmyyhwu/tidybot2](https://github.com/jimmyyhwu/tidybot2)
(TidyBot++, CoRL 2024), MIT licensed — see [LICENSE](LICENSE) and
[UPSTREAM_README.md](UPSTREAM_README.md). Vendored here so the pi-robotics
daemon can load them offline.

The robot is a **holonomic mobile manipulator**: a 3-DOF powered-caster base
(`joint_x`, `joint_y`, `joint_th`) + a 7-DOF **Kinova Gen3** arm (`joint_1..7`)
+ a **Robotiq 2F85** gripper (`fingers_actuator`), with `base` and `wrist`
cameras and a `pinch_site` at the gripper. 11 actuators total.

## Layout

```text
examples/tidybot2/
├── LICENSE, UPSTREAM_README.md         # from the upstream repo
└── models/
    ├── assets/                         # gen3 / 2f85 / base meshes
    ├── kinova_gen3/                    # bare Gen3 arm scenes
    └── stanford_tidybot/
        ├── base.xml, tidybot.xml, scene.xml   # upstream robot + cube scene
        ├── scene_water_plant.xml              # NEW: water the plant
        └── scene_fill_cup.xml                 # NEW: fill the cup with water
```

## Loading a scene

```text
/robotics start stretch            # any scene boots the daemon
/robotics load examples/tidybot2/models/stanford_tidybot/scene_water_plant.xml
```

or with the tool:

```jsonc
{ "action": "load", "path": "examples/tidybot2/models/stanford_tidybot/scene_water_plant.xml" }
```

`scene.xml` is the upstream "pick the cube" sanity scene. The two task scenes
below are authored for this repo.

## Task scenes

### `scene_water_plant.xml` — water the plant

Workbench in front of the robot with a **potted plant** (fixed target) and a
**watering can** (free body). Drive the arm to grasp the can and pour over the
plant. Reachable keypoints (`state` gives `qpos`; use `ctrl` on the actuators):

- watering can starts at ~`(0.62, -0.12, 0.40)`, plant at `(0.62, 0.15, 0.38)`.

### `scene_fill_cup.xml` — fill the cup

Workbench with a **dispenser** (fixed: base + post + arm + spout at
`(0.62, 0.0, 0.61)`) and a **cup** (free body, starts under the spout). Drive the
arm to place/hold the cup under the spout.

## A note on water

MuJoCo does not simulate fluid. The water inside the watering can, the column in
the cup, and the stream from the spout are **translucent visual placeholders**
(`contype="0" conaffinity="0" mass="0"`, so they do not affect physics). The
scenes model the **manipulation** (reach, grasp, carry, pour/place); an actual
fill level would be driven by task code (e.g. scaling the water geom when the cup
is under the spout).

## Coordinate conventions

Robot base at the origin facing **+x**; the arm base is at `z = 0.335`. The
upstream sim uses end-effector targets around `x ≈ 0.55, z ≈ 0.4`, which is
where the workbench top (`z = 0.38`) sits.
