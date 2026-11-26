"""Offscreen MuJoCo rendering helpers for the pi-robotics daemon.

``mujoco.Renderer`` owns a GL context tied to the thread that created it, and
the control server handles each request on a new thread. A renderer must
therefore never outlive the request/thread that made it (or be shared between
threads). These helpers create and close one per call unless the caller passes
its own ``renderer`` — the operation reel creates one per camera inside a single
request and reuses it across every frame.

Everything here renders from a *scratch* ``MjData`` seeded from the published
state snapshot, never the live ``MjData`` the sim thread is stepping.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

import mujoco

# Free-camera defaults for the 3D-panel column of an operation reel.
PANEL_WIDTH = 800
PANEL_HEIGHT = 600
PANEL_CAMERA_OPTS = {
    "distance": 2.4,
    "elevation": -22,
    "azimuth": 130,
    "lookat": [-0.1, -0.4, 0.6],
}


def scratch_from_snapshot(
    model: mujoco.MjModel, snapshot: dict[str, Any]
) -> mujoco.MjData:
    """A scratch ``MjData`` seeded with the published state (for offscreen render)."""
    scratch = mujoco.MjData(model)
    qpos = np.asarray(snapshot.get("qpos", []), dtype=float)
    qvel = np.asarray(snapshot.get("qvel", []), dtype=float)
    if qpos.size == model.nq:
        scratch.qpos[:] = qpos
    if qvel.size == model.nv:
        scratch.qvel[:] = qvel
    return scratch


def render_array(
    model: mujoco.MjModel,
    scratch: mujoco.MjData,
    width: int,
    height: int,
    camera: str | None,
    camera_opts: dict[str, Any] | None = None,
    renderer: Any = None,
) -> np.ndarray:
    """Renders one camera (or the free panel view) to an RGB uint8 array."""
    own = renderer is None
    if own:
        renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    try:
        mujoco.mj_forward(model, scratch)
        if camera:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cam_id < 0:
                raise ValueError(f"unknown camera '{camera}'")
            renderer.update_scene(scratch, camera=cam_id)
        else:
            opts = camera_opts or {}
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = float(opts.get("distance", 2.0))
            cam.elevation = float(opts.get("elevation", -20))
            cam.azimuth = float(opts.get("azimuth", 120))
            if opts.get("lookat") is not None:
                cam.lookat[:] = np.asarray(opts["lookat"], dtype=float)[:3]
            renderer.update_scene(scratch, camera=cam)
        return np.asarray(renderer.render()).copy()
    finally:
        if own:
            try:
                renderer.close()
            except Exception:  # noqa: BLE001
                pass


def render_png(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    width: int,
    height: int,
    camera: str | None,
    camera_opts: dict[str, Any] | None = None,
) -> bytes:
    """Offscreen render to PNG bytes (best effort; needs a GL context)."""
    import imageio.v3 as iio  # optional dependency

    opts = camera_opts or {}
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        cam = mujoco.MjvCamera()
        if camera:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cam_id < 0:
                raise ValueError(f"unknown camera '{camera}'")
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = cam_id
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            if "distance" in opts:
                cam.distance = float(opts["distance"])
            if "elevation" in opts:
                cam.elevation = float(opts["elevation"])
            if "azimuth" in opts:
                cam.azimuth = float(opts["azimuth"])
            lookat = opts.get("lookat")
            if lookat is not None:
                cam.lookat[:] = np.asarray(lookat, dtype=float)[:3]
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
    finally:
        renderer.close()
    return iio.imwrite("<bytes>", frame, extension=".png")


def render_png_from_state(
    model: mujoco.MjModel,
    snapshot: dict[str, Any],
    width: int,
    height: int,
    camera: str | None,
    camera_opts: dict[str, Any] | None = None,
) -> bytes:
    """Renders a camera frame from the published state snapshot (no live data)."""
    scratch = scratch_from_snapshot(model, snapshot)
    mujoco.mj_forward(model, scratch)
    return render_png(model, scratch, width, height, camera, camera_opts)


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encodes an HxWx3 uint8 RGB frame as JPEG (needs Pillow)."""
    from PIL import Image  # optional dependency

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def encode_png(array: np.ndarray) -> str:
    """Encodes an HxWx3 uint8 RGB array as a base64 PNG string (needs Pillow)."""
    from PIL import Image  # optional dependency

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


CAMERA_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>pi-robotics camera</title>
<style>
 body{{margin:0;background:#111;color:#ddd;font:13px system-ui,sans-serif}}
 header{{padding:6px 10px;background:#1b1b1b;display:flex;gap:10px;align-items:center}}
 img{{display:block;width:100%;height:auto;max-width:100vw}}
 select{{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:4px;padding:3px 6px}}
</style></head>
<body>
<header>
  <strong>pi-robotics</strong>
  <label>camera <select id="c">{options}</select></label>
  <span id="s">connecting\u2026</span>
</header>
<img id="v" alt="robot camera" src="/stream.mjpg?camera={camera}&width={width}&height={height}&fps={fps}">
<script>
 const sel=document.getElementById('c'), img=document.getElementById('v'), s=document.getElementById('s');
 function go(){{img.src='/stream.mjpg?camera='+encodeURIComponent(sel.value)+'&width={width}&height={height}&fps={fps}';}}
 sel.addEventListener('change', go);
 img.addEventListener('load', ()=>s.textContent='live');
 img.addEventListener('error', ()=>s.textContent='stream ended (sim running?)');
</script>
</body></html>
"""
