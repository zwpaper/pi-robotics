"""Operation reel: compose per-`look` frames (3D panel + cameras) into a video.

Each `look`/`nudge` records one *step*: a 3D overview frame (the scene the 3D
panel shows) plus one frame per camera. `build()` lays the columns out side by
side, in first-seen order (panel first, then cameras), and encodes an MP4 with
the system ``ffmpeg`` (falling back to an animated GIF via Pillow).
"""

from __future__ import annotations

import io
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

    def clear(self) -> None:
        with self._lock:
            self._steps = []
            self._cameras = []
            self._stamps = []

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
        name: str = "operation",
    ) -> dict[str, Any]:
        with self._lock:
            steps = list(self._steps)
            cameras = list(self._cameras)
            stamps = list(self._stamps)
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
        columns = [PANEL_COLUMN, *cameras]
        columns_cams = cameras

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
        encoded = 0
        for i, row in enumerate(tiles):
            canvas = Image.new("RGB", (frame_width, height), _PLACEHOLDER)
            x = 0
            for col in columns:
                image = row.get(col)
                if image is None:
                    x += widths[col]
                    continue
                canvas.paste(image, (x, 0))
                x += widths[col]
            # Hold each recorded frame for its duration (repeat at the output
            # frame rate) so the video is readable instead of a fast flip-book.
            repeats = max(1, round(durations[i] * fps))
            for _ in range(repeats):
                canvas.save(frame_dir / f"frame_{encoded:04d}.png")
                encoded += 1

        out_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
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
                return {
                    "path": str(output),
                    "format": "mp4",
                    "steps": len(tiles),
                    "frames": encoded,
                    "seconds": round(sum(durations), 2),
                    "hold_seconds": [round(d, 2) for d in durations],
                    "columns": columns,
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
            "fps": fps,
        }
