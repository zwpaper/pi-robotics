"""Operation reel: compose recorded frames (3D panel + all cameras) into a video.

A recording captures one *step* per frame: a 3D overview frame (what the 3D
panel shows) plus one frame per camera. ``build()`` lays the tiles out in a
**grid** (the 3D panel first, then every camera), writes an MP4 with the system
``ffmpeg`` (falling back to an animated GIF via Pillow) and attaches a
**QuickTime chapter track** for the recorded attempts.
"""

from __future__ import annotations

import io
import math
import pathlib
import shutil
import subprocess
import threading
import time
from typing import Any

from PIL import Image

PANEL_COLUMN = "panel"
_PLACEHOLDER = (24, 24, 28)


class Recorder:
    """Ordered timeline of (panel, {camera: png}) steps."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: list[dict[str, Any]] = []
        self._cameras: list[str] = []
        self._stamps: list[float] = []
        # (recorded-step index, title) marks used to build a chapter track.
        self._chapters: list[tuple[int, str]] = []

    def clear(self) -> None:
        with self._lock:
            self._steps = []
            self._cameras = []
            self._stamps = []
            self._chapters = []

    def mark_chapter(self, title: str) -> int:
        """Marks the start of a new attempt/chapter at the current step."""
        with self._lock:
            self._chapters.append((len(self._steps), str(title or "attempt")))
            return len(self._chapters)

    def chapters(self) -> list[tuple[int, str]]:
        with self._lock:
            return list(self._chapters)

    def add(self, panel_png: bytes, cameras: dict[str, bytes]) -> int:
        with self._lock:
            for name in cameras:
                if name not in self._cameras:
                    self._cameras.append(name)
            self._steps.append({"panel": panel_png, "cameras": dict(cameras)})
            self._stamps.append(time.time())
            return len(self._steps)

    def count(self) -> int:
        with self._lock:
            return len(self._steps)

    def columns(self) -> list[str]:
        with self._lock:
            return [PANEL_COLUMN, *self._cameras]

    def build(
        self,
        out_dir: pathlib.Path,
        *,
        fps: float = 10.0,
        hold: float | None = None,
        min_hold: float = 0.6,
        max_hold: float = 6.0,
        tail: float = 0.8,
        height: int = 360,
        cols: int | None = None,
        name: str = "operation",
    ) -> dict[str, Any]:
        with self._lock:
            steps = list(self._steps)
            cameras = list(self._cameras)
            stamps = list(self._stamps)
            chapters = list(self._chapters)
        if not steps:
            raise ValueError("no frames recorded; run 'look' or 'nudge' first")

        # How long each recorded frame is shown. By default this is the real
        # elapsed time between recordings, so the video plays at the pace the
        # operation actually happened (each `nudge` waits for the arm to
        # settle). `hold` forces a fixed per-image time instead.
        count = len(steps)
        if hold is not None:
            durations = [float(hold)] * count
        elif len(stamps) == count and count >= 2:
            durations = [
                stamps[i + 1] - stamps[i] for i in range(count - 1)
            ]
            durations.append(max(min_hold, durations[-1] if durations else min_hold))
        else:
            durations = [max(min_hold, 1.2)] * count
        durations = [max(min_hold, min(max_hold, d)) for d in durations]
        # Hold the final pose a moment so the finished state is clearly visible.
        if tail and durations:
            durations[-1] = min(max_hold, durations[-1] + float(tail))
        # Swallow tiny jitter: clamp per-frame hold so a dense continuous take
        # plays smoothly rather than as a slow slide show.
        if hold is None and min(durations) < 0.08 and max(durations) < 0.25:
            durations = [max(0.08, min(max_hold, d)) for d in durations]
        # Chapter boundaries (start seconds per recorded step) for the attempts.
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
        # Grid layout: the 3D panel + all cameras, arranged in a grid.
        columns = [PANEL_COLUMN, *cameras]
        columns_cams = cameras
        n_tiles = len(columns)
        grid_cols = int(cols) if cols else max(1, math.ceil(math.sqrt(n_tiles)))
        grid_cols = max(1, min(grid_cols, n_tiles))
        grid_rows = max(1, math.ceil(n_tiles / grid_cols))

        def load(png: bytes) -> Image.Image:
            image = Image.open(io.BytesIO(png)).convert("RGB")
            width = max(1, round(image.width * height / image.height))
            return image.resize((width, height))

        # Load every tile once, remember per-column widths.
        tiles: list[dict[str, Image.Image]] = []
        widths = {col: 0 for col in columns}
        for step in steps:
            row: dict[str, Image.Image] = {}
            for col in columns:
                png = (
                    step["panel"]
                    if col == PANEL_COLUMN
                    else step["cameras"].get(col)
                )
                if png:
                    row[col] = load(png)
            for col in columns_cams:
                if col not in row:
                    # Camera added on a later step: blank placeholder.
                    row[col] = Image.new(
                        "RGB", (max(1, round(height * 4 / 3)), height), _PLACEHOLDER
                    )
            for col, image in row.items():
                widths[col] = max(widths[col], image.width)
            tiles.append(row)

        frame_width = sum(widths.values())
        frame_dir = out_dir / f"{name}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        # Per grid-column widths (tiles in a grid column share a width).
        grid_col_widths = [0] * grid_cols
        for i, col in enumerate(columns):
            c = i % grid_cols
            grid_col_widths[c] = max(grid_col_widths[c], widths[col])
        frame_width = sum(grid_col_widths)
        frame_height = grid_rows * height

        def cell_x(c: int) -> int:
            return sum(grid_col_widths[:c])

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
            # Hold each recorded frame for its duration (repeat at the output
            # frame rate) so the video is readable instead of a fast flip-book.
            repeats = max(1, round(durations[i] * fps))
            for _ in range(repeats):
                canvas.save(frame_dir / f"frame_{encoded:04d}.png")
                encoded += 1

        out_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        spans = chapter_spans()
        if ffmpeg:
            output = out_dir / f"{name}.mp4"
            command = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(frame_dir / "frame_%04d.png"),
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
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                str(output),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                shutil.rmtree(frame_dir, ignore_errors=True)
                meta_path.unlink(missing_ok=True)
                return {
                    "path": str(output),
                    "format": "mp4",
                    "steps": len(tiles),
                    "frames": encoded,
                    "seconds": round(sum(durations), 2),
                    "hold_seconds": [round(d, 2) for d in durations],
                    "chapters": [
                        {"start": round(s, 2), "end": round(e, 2), "title": t}
                        for s, e, t in spans
                    ],
                    "columns": columns,
                    "layout": "grid",
                    "rows": grid_rows,
                    "cols": grid_cols,
                    "fps": fps,
                }
            except subprocess.CalledProcessError:
                pass  # fall through to GIF

        # Fallback: animated GIF.
        output = out_dir / f"{name}.gif"
        images = [
            Image.open(frame_dir / f"frame_{i:04d}.png").convert("RGB")
            for i in range(encoded)
        ]
        images[0].save(
            output,
            save_all=True,
            append_images=images[1:],
            duration=max(1, round(1000 / fps)),
            loop=0,
        )
        shutil.rmtree(frame_dir, ignore_errors=True)
        return {
            "path": str(output),
            "format": "gif",
            "steps": len(tiles),
            "frames": len(images),
            "seconds": round(sum(durations), 2),
            "hold_seconds": [round(d, 2) for d in durations],
            "columns": columns,
            "layout": "grid",
            "rows": grid_rows,
            "cols": grid_cols,
            "fps": fps,
        }
