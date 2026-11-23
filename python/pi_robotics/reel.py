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

    def clear(self) -> None:
        with self._lock:
            self._steps = []
            self._cameras = []

    def add(self, panel_png: bytes, cameras: dict[str, bytes]) -> int:
        with self._lock:
            for name in cameras:
                if name not in self._cameras:
                    self._cameras.append(name)
            self._steps.append({"panel": panel_png, "cameras": dict(cameras)})
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
        fps: float = 4.0,
        height: int = 360,
        name: str = "operation",
    ) -> dict[str, Any]:
        with self._lock:
            steps = list(self._steps)
            cameras = list(self._cameras)
        if not steps:
            raise ValueError("no frames recorded; run 'look' or 'nudge' first")

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
            canvas.save(frame_dir / f"frame_{i:04d}.png")

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
                "-pix_fmt",
                "yuv420p",
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
                    "frames": len(tiles),
                    "columns": columns,
                    "fps": fps,
                }
            except subprocess.CalledProcessError:
                pass  # fall through to GIF

        # Fallback: animated GIF.
        output = out_dir / f"{name}.gif"
        images = []
        for i in range(len(tiles)):
            images.append(Image.open(frame_dir / f"frame_{i:04d}.png").convert("RGB"))
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
            "frames": len(images),
            "columns": columns,
            "fps": fps,
        }
