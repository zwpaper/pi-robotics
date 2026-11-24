# MarkerRemoveLid — failing three times, then finding the trick

Demo description for the operation video `marker_remove_lid.mp4` (not committed
to the repository — the reel lives in `<cwd>/.pi/robotics/runtime*/reels/`,
which is git-ignored).

**Task** (`google-deepmind/aloha_sim`, Apache-2.0): a bimanual ALOHA 2 stands a
marker upright on the table and must **pull the cap off the marker**. The red
cap is a tight friction fit on the marker body, so the task is really a
dexterous *grip-and-pull* problem: get both fingers fully around a small,
smooth, round cap, hold it against a strong extraction force, and break the fit
without losing the grip.

**In this video**: an LLM agent (driven through the `robotics` tool's
`look` / `move_ee` / `nudge` / `contains` loop) needs **four attempts**. The
first three fail; on the fourth it finds the trick and the cap comes off.

## Video

| | |
|---|---|
| grid | MuJoCo 3D panel + all 6 ALOHA cameras (`overhead_cam`, `worms_eye_cam`, `wrist_cam_left`, `wrist_cam_right`, …) |
| resolution | 1600 × 1440 |
| duration | 45.3 s, 30 fps |
| chapters | 4 — one per **attempt**, including the failed ones |

| chapter | attempt | start |
|---|---|---|
| 1 | attempt 1: low-tension axial pull | 0:00 |
| 2 | attempt 2: lifted mid-air opposing pull | 0:10 |
| 3 | attempt 3: table-level pull + twist (no slip recovery) | 0:21 |
| 4 | attempt 4: high-tension opposing pull + peel + slip recovery | 0:31 |

## What went wrong, and what finally worked

| # | technique | outcome |
|---|---|---|
| 1 | one arm, **axial pull** along the marker axis, gentle grip | the pull never exceeds the cap–body friction, so the cap just slides in the fingers. Failed. |
| 2 | lift the marker **mid-air** and pull with both arms in **opposing** directions | there is no table reaction force to brace against; the marker squirts out of the opposite gripper and the grasp is lost. Failed. |
| 3 | pull **at table level** and add a **twist** | the right idea (the table as a brace, the twist to shear the fit), but once the fingers slip there is no recovery — the cap rotates on the spot and stays on. Failed. |
| 4 | **high-tension opposing pull + peel + slip recovery** | the trick: (a) close hard on the cap — high normal force means high friction; (b) hold the marker body down on the table with the other arm so the pull is braced; (c) **peel** the cap sideways instead of dragging it straight along the axis; and (d) when the fingers start to slip, **recover** — re-seat and re-grip instead of pushing harder. The cap pops off and stays in the gripper. **Succeeded.** |

The actionable lesson for an agent: for a friction-fit cap, the *friction
budget* matters more than raw force. Grip force, a braced (not airborne)
posture, a peeling direction, and a recovery move each buy more than simply
pulling harder — which is exactly what the first three attempts kept doing.

## Reproducing

```bash
# 1. the scene (vendored via Git LFS)
git lfs install

# 2. start the daemon on this scene (it must be the start-up model: this model's
#    contact count is high enough that hot-swapping can abort the sim loop)
python python/robotics_daemon.py \
  --model "$PWD/examples/aloha_sim/models/MarkerRemoveLid/scene.xml" \
  --runtime-dir .pi/robotics/runtime --session-file .pi/robotics/session.json
```

Then, from a pi session, drive the arms with `look` / `move_ee` / `nudge`,
wrap each try in `record_start` … `chapter { title }` … `record_stop`, and finally
render the grid video with `reel`.
