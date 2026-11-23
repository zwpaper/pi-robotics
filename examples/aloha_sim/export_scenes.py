#!/usr/bin/env python3
"""Export Aloha Sim task scenes to self-contained MJCF for the pi-robotics daemon.

The google-deepmind/aloha_sim tasks are dm_control `composer` environments, not
standalone MJCF files. This script creates each task, takes its composed
`mjcf_model`, and writes it out with all meshes/textures (dm_control's
`export_with_assets`), deduplicating the shared assets into one directory so the
three scenes load offline with plain `mujoco`.

Usage:
  python examples/aloha_sim/export_scenes.py \
      --aloha-sim /path/to/aloha_sim \
      --out <project>/.pi/robotics/models/aloha_sim

Requires an environment with `dm_control` and `aloha_sim` importable
(e.g. `pip install dm-control` and `pip install -e /path/to/aloha_sim`).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil

TASKS = [
    "DiningPlaceBananaInBowl",
    "ToolsPlaceCanOpenerInLeftCompartment",
    "MarkerRemoveLid",
]


def _rewrite_asset_dirs(scene_xml: pathlib.Path, asset_dir: str) -> None:
    """Points the scene at the shared asset directory."""
    text = scene_xml.read_text()
    compiler = f'<compiler angle="radian" meshdir="{asset_dir}" texturedir="{asset_dir}"/>'
    if re.search(r"<compiler\b[^>]*/>", text):
        text = re.sub(r"<compiler\b[^>]*/>", compiler, text, count=1)
    else:
        text = re.sub(r"(<mujoco\b[^>]*>)", r"\1\n  " + compiler, text, count=1)
    scene_xml.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aloha-sim", default=os.environ.get("ALOHA_SIM_PATH", ""))
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", nargs="*", default=TASKS)
    args = parser.parse_args()

    if args.aloha_sim:
        import sys

        sys.path.insert(0, str(pathlib.Path(args.aloha_sim).resolve()))

    os.environ.setdefault("MUJOCO_GL", "disable")  # no rendering needed to compile
    from aloha_sim import task_suite
    from dm_control import mjcf as dm_mjcf

    out = pathlib.Path(args.out).resolve()
    assets = out / "assets"
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(exist_ok=True)

    for name in args.tasks:
        env = task_suite.create_task_env(name, time_limit=10.0, cameras=())
        root = env.task.root_entity.mjcf_model
        staging = out / f".{name}.staging"
        shutil.rmtree(staging, ignore_errors=True)
        dm_mjcf.export_with_assets(root, str(staging), out_file_name="scene.xml")
        scene_dir = out / name
        scene_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for item in staging.iterdir():
            if item.name == "scene.xml":
                shutil.move(str(item), str(scene_dir / "scene.xml"))
            else:
                target = assets / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
                    moved += 1
        shutil.rmtree(staging, ignore_errors=True)
        _rewrite_asset_dirs(scene_dir / "scene.xml", "../assets")
        print(f"{name}: {moved} new assets -> {scene_dir / 'scene.xml'}")

    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
