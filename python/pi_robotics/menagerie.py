"""MuJoCo Menagerie model fetching for pi-robotics.

The real robot models (Hello Robot Stretch and friends) live in
``google-deepmind/mujoco_menagerie`` as MJCF + meshes and are ~70 MB each, so
they are **not** vendored into this repo. Instead they are fetched on demand
into ``<cwd>/.pi/robotics/models/`` with a sparse, blobless ``git`` checkout of
just the requested model directory, and reused afterwards.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import uuid

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"

# Scene name -> Menagerie directory. Fetched on demand; never vendored.
MODELS: dict[str, str] = {
    # Mobile manipulators.
    "stretch": "hello_robot_stretch",
    "stretch3": "hello_robot_stretch_3",
    # Arms.
    "panda": "franka_emika_panda",
    "franka_fr3": "franka_fr3",
    "ur5e": "universal_robots_ur5e",
    "aloha": "aloha",
    "so101": "robotstudio_so101",
    # Quadrupeds / humanoids.
    "go2": "unitree_go2",
    "g1": "unitree_g1",
}

# MJCF file to open inside each model directory (most ship `scene.xml`).
SCENE_FILES: dict[str, str] = {}
DEFAULT_SCENE_FILE = "scene.xml"


def is_menagerie_scene(name: str | None) -> bool:
    return bool(name) and str(name) in MODELS


def model_dir(models_root: pathlib.Path, name: str) -> pathlib.Path:
    return models_root / MODELS[name]


def scene_path(models_root: pathlib.Path, name: str) -> pathlib.Path:
    return model_dir(models_root, name) / SCENE_FILES.get(name, DEFAULT_SCENE_FILE)


def ensure_model(
    models_root: pathlib.Path,
    name: str,
    *,
    ref: str = "main",
    timeout: float = 600.0,
) -> pathlib.Path:
    """Returns the model directory, fetching it on first use.

    Raises RuntimeError with an actionable message when the model is missing
    and cannot be fetched (no git, no network).
    """
    if name not in MODELS:
        raise KeyError(f"Unknown Menagerie model '{name}'. Known: {', '.join(MODELS)}")

    destination = model_dir(models_root, name)
    scene_file = SCENE_FILES.get(name, DEFAULT_SCENE_FILE)
    if (destination / scene_file).is_file():
        return destination

    models_root.mkdir(parents=True, exist_ok=True)
    scratch = models_root / f".fetch-{uuid.uuid4().hex[:8]}"
    try:
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                ref,
                MENAGERIE_URL,
                str(scratch),
            ],
            timeout=timeout,
        )
        _run(
            ["git", "-C", str(scratch), "sparse-checkout", "set", MODELS[name]],
            timeout=timeout,
        )
        source = scratch / MODELS[name]
        if not (source / scene_file).is_file():
            raise RuntimeError(
                f"fetched '{name}' but {MODELS[name]}/{scene_file} is missing"
            )
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        shutil.move(str(source), str(destination))
        return destination
    except FileNotFoundError as exc:  # git missing
        raise RuntimeError(
            "`git` is required to fetch MuJoCo Menagerie models. Install git or "
            f"place the model at {destination} manually."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"failed to fetch Menagerie model '{name}': {exc.stderr or exc}. "
            f"Fetch it manually into {destination} (e.g. a sparse clone of "
            f"{MENAGERIE_URL})."
        ) from exc
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _run(command: list[str], timeout: float) -> None:
    subprocess.run(
        command,
        check=True,
        timeout=timeout,
        capture_output=True,
        text=True,
    )
