"""Operation reel: record a state timeline, then compose a multi-view video.

Recording is decoupled from rendering:

* `record_start` samples the simulation **state** (``qpos``) at a frame rate —
  cheap, no OpenGL, so it can run fast and smooth, and it never starves the
  control requests.
* `reel` renders the grid (3D-panel overview + **every** camera) from that stored
  timeline, once, and encodes an MP4 with the system ``ffmpeg`` (GIF fallback).
* `chapter` marks the attempts, written as a **QuickTime chapter track**.

Because the camera views and the panel are all functions of the state, rendering
them after the fact gives a smooth, complete video without slowing the run.
"""

from __future__ import annotations

import math
import pathlib
import shutil
import subprocess
import threading
import time
from typing import Any

import numpy as np
from PIL import Image

PANEL_COLUMN = "panel"
_PLACEHOLDER = (24, 24, 28)


class Recorder:
    """Thread-safe timeline of (timestamp, qpos) plus attempt chapters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: list[tuple[float, np.ndarray]] = []
        self._chapters: list[tuple[int, str]] = []

    def clear(self) -> None:
        with self._lock:
            self._states = []
            self._chapters = []

    def add_state(self, qpos: np.ndarray) -> int:
        with self._lock:
            self._states.append(
                (time.time(), np.asarray(qpos, dtype=float).copy())
            )
            return len(self._states)

    def mark_chapter(self, title: str) -> int:
        with self._lock:
            self._chapters.append((len(self._states), str(title or "attempt")))
            return len(self._chapters)

    def count(self) -> int:
        with self._lock:
            return len(self._states)

    def timeline(self) -> tuple[list[tuple[float, np.ndarray]], list[tuple[int, str]]]:
        with self._lock:
            states = [(t, q.copy()) for t, q in self._states]
            return states, list(self._chapters)


def compose_video(
    out_dir: pathlib.Path,
    frames: list[dict[str, Any]],
    stamps: list[float],
    chapters: list[tuple[int, str]],
    *,
    fps: float = 10.0,
    hold: float | None = None,
    min_hold: float = 0.0,
    max_hold: float = 6.0,
    tail: float = 0.8,
    height: int = 480,
    cols: int | None = None,
    name: str = "operation",
) -> dict[str, Any]:
    """Lays out ``frames`` (panel + cameras) in a grid and encodes a video.

    ``frames`` is a list of ``{"panel": ndarray, "cameras": {name: ndarray}}`` in
    execution order; ``stamps`` gives their real times (used for pacing when
    ``hold`` is None); ``chapters`` is ``(frame index, title)``.
    """
    if not frames:
        raise ValueError("no frames to compose")

    # Camera set, in first-seen order.
    cameras: list[str] = []
    for frame in frames:
        for cam in frame.get("cameras", {}):
            if cam not in cameras:
                cameras.append(cam)
    columns = [PANEL_COLUMN, *cameras]
    n_tiles = len(columns)
    grid_cols = int(cols) if cols else max(1, math.ceil(math.sqrt(n_tiles)))
    grid_cols = max(1, min(grid_cols, n_tiles))
    grid_rows = max(1, math.ceil(n_tiles / grid_cols))

    # Per-frame display durations (real pacing, or a fixed `hold`).
    count = len(frames)
    if hold is not None:
        durations = [float(hold)] * count
    elif len(stamps) == count and count >= 2:
        durations = [stamps[i + 1] - stamps[i] for i in range(count - 1)]
        durations.append(durations[-1] if durations else 1.0)
    else:
        durations = [max(min_hold, 1.0 / max(fps, 1.0))] * count
    durations = [max(min_hold, min(max_hold, d)) for d in durations]
    if tail and durations:
        durations[-1] = min(max_hold, durations[-1] + float(tail))

    starts: list[float] = []
    acc = 0.0
    for d in durations:
        starts.append(acc)
        acc += d
    total_seconds = acc

    def chapter_spans() -> list[tuple[float, float, str]]:
        spans: list[tuple[float, float, str]] = []
        for i, (idx, title) in enumerate(chapters):
            start = starts[idx] if 0 <= idx < len(starts) else (starts[-1] if starts else 0.0)
            nxt = chapters[i + 1][0] if i + 1 < len(chapters) else len(starts)
            end = starts[nxt] if 0 <= nxt < len(starts) else total_seconds
            spans.append((start, max(end, start + 0.2), title))
        return spans

    def load(array: Any) -> Image.Image:
        image = Image.fromarray(np.asarray(array, dtype=np.uint8))
        width = max(1, round(image.width * height / image.height))
        return image.resize((width, height))

    tiles: list[dict[str, Image.Image]] = []
    widths = {col: 0 for col in columns}
    for frame in frames:
        row: dict[str, Image.Image] = {}
        panel = frame.get("panel")
        if panel is not None:
            row[PANEL_COLUMN] = load(panel)
        for cam, array in frame.get("cameras", {}).items():
            if array is not None:
                row[cam] = load(array)
        for cam in cameras:
            if cam not in row:
                row[cam] = Image.new(
                    "RGB", (max(1, round(height * 4 / 3)), height), _PLACEHOLDER
                )
        for col, image in row.items():
            widths[col] = max(widths[col], image.width)
        tiles.append(row)

    grid_col_widths = [0] * grid_cols
    for i, col in enumerate(columns):
        c = i % grid_cols
        grid_col_widths[c] = max(grid_col_widths[c], widths.get(col, 0))
    frame_width = sum(grid_col_widths)
    frame_height = grid_rows * height

    def cell_x(c: int) -> int:
        return sum(grid_col_widths[:c])

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / f"{name}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    encoded = 0
    for i, row in enumerate(tiles):
        canvas = Image.new("RGB", (frame_width, frame_height), _PLACEHOLDER)
        for j, col in enumerate(columns):
            image = row.get(col)
            if image is None:
                continue
            r, c = divmod(j, grid_cols)
            x = cell_x(c) + (grid_col_widths[c] - image.width) // 2
            canvas.paste(image, (x, r * height))
        repeats = max(1, round(durations[i] * fps))
        for _ in range(repeats):
            canvas.save(frame_dir / f"frame_{encoded:04d}.png")
            encoded += 1

    spans = chapter_spans()
    result: dict[str, Any] = {
        "format": "gif",
        "steps": len(frames),
        "frames": encoded,
        "seconds": round(sum(durations), 2),
        "layout": "grid",
        "rows": grid_rows,
        "cols": grid_cols,
        "columns": columns,
        "chapters": [
            {"start": round(s, 2), "end": round(e, 2), "title": t}
            for s, e, t in spans
        ],
        "fps": fps,
    }

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        output = out_dir / f"{name}.mp4"
        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%04d.png"),
        ]
        meta_path = out_dir / f".{name}.chapters.txt"
        if spans:
            lines = [";FFMETADATA1"]
            for start, end, title in spans:
                lines += [
                    "[CHAPTER]",
                    "TIMEBASE=1/1000",
                    f"START={int(round(start * 1000))}",
                    f"END={int(round(end * 1000))}",
                    f"title={title}",
                ]
            meta_path.write_text("\n".join(lines) + "\n")
            command += ["-i", str(meta_path), "-map", "0:v:0", "-map_metadata", "1"]
        command += [
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(output),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            shutil.rmtree(frame_dir, ignore_errors=True)
            meta_path.unlink(missing_ok=True)
            result["path"] = str(output)
            result["format"] = "mp4"
            return result
        except subprocess.CalledProcessError:
            pass  # fall through to GIF

    output = out_dir / f"{name}.gif"
    images = [
        Image.open(frame_dir / f"frame_{i:04d}.png").convert("RGB")
        for i in range(encoded)
    ]
    images[0].save(
        output, save_all=True, append_images=images[1:],
        duration=max(1, round(1000 / fps)), loop=0,
    )
    shutil.rmtree(frame_dir, ignore_errors=True)
    result["path"] = str(output)
    return result
