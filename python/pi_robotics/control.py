"""Loopback HTTP control API and shared state for the pi-robotics daemon."""

from __future__ import annotations

import base64
import http.server
import io
import json
import pathlib
import threading
import time
import urllib.parse
from typing import Any

import numpy as np

import mujoco
from mujoco.studio import messages, sim, viewer_handle

from pi_robotics.reel import Recorder

# Free-camera defaults for the 3D-panel column of an operation reel.
PANEL_WIDTH = 480
PANEL_HEIGHT = 360
PANEL_CAMERA_OPTS = {
    "distance": 2.4,
    "elevation": -22,
    "azimuth": 130,
    "lookat": [-0.1, -0.4, 0.6],
}


class SharedState:
    """Thread-safe snapshot the control server reads and the sim thread writes."""

    def __init__(
        self,
        *,
        runtime_dir: pathlib.Path,
        scene_name: str,
        scene_params: dict[str, Any],
        viewer_port: int,
        control_port: int,
    ) -> None:
        self.runtime_dir = runtime_dir
        self.scene_name = scene_name
        self.scene_params = scene_params
        self.model_path = ""
        self.viewer_port = viewer_port
        self.control_port = control_port
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {"time": 0.0, "step": 0}
        self._model_meta: dict[str, Any] = {}
        self._scene_source = ""
        self._scene_xml = ""
        self._errors: dict[str, str] = {}
        self._pick: dict[str, Any] = {"active": False}
        self._servo: dict[str, Any] = {"active": False}

    # -- setters (sim thread) ------------------------------------------------
    def set_viewer_port(self, port: int) -> None:
        with self._lock:
            self.viewer_port = int(port)

    def set_scene_source(self, source: str, xml: str = "") -> None:
        with self._lock:
            self._scene_source = source
            if xml:
                self._scene_xml = xml

    def record_error(self, component: str | None, error: str | None) -> None:
        with self._lock:
            if component is None:
                self._errors = {}
            elif error:
                self._errors[component] = error
            else:
                self._errors.pop(component, None)

    def has_error(self, component: str) -> bool:
        with self._lock:
            return component in self._errors

    def set_pick(self, info: dict[str, Any]) -> None:
        with self._lock:
            self._pick = dict(info)

    def clear_pick(self) -> None:
        with self._lock:
            self._pick = {"active": False}

    def set_servo(self, info: dict[str, Any]) -> None:
        with self._lock:
            self._servo = dict(info)

    def publish_model(self, model: mujoco.MjModel) -> None:
        def name_of(objtype: mujoco.mjtObj, i: int, fallback: str) -> str:
            return mujoco.mj_id2name(model, objtype, i) or f"{fallback}{i}"

        with self._lock:
            self._model_meta = {
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nu": int(model.nu),
                "nbody": int(model.nbody),
                "nkey": int(model.nkey),
                "timestep": float(model.opt.timestep),
                "joints": [
                    name_of(mujoco.mjtObj.mjOBJ_JOINT, i, "joint")
                    for i in range(model.njnt)
                ],
                "actuators": [
                    name_of(mujoco.mjtObj.mjOBJ_ACTUATOR, i, "actuator")
                    for i in range(model.nu)
                ],
                "keyframes": [
                    name_of(mujoco.mjtObj.mjOBJ_KEY, i, "key")
                    for i in range(model.nkey)
                ],
            }
            self._state["step"] = 0

    def publish_data(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        ctrl = np.asarray(data.ctrl, dtype=float)
        snapshot = {
            "time": float(data.time),
            "step": None,  # filled below
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "ctrl": ctrl.tolist(),
            "ncon": int(data.ncon),
            "sites": self._site_positions(model, data),
        }
        with self._lock:
            step = int(self._state.get("step") or 0) + 1
            snapshot["step"] = step
            self._state = snapshot

    def _site_positions(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for sid in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
            if name:
                out[name] = data.site_xpos[sid].tolist()
        return out

    # -- getters (control thread) -------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._state)
            out["model"] = dict(self._model_meta)
            out["pick"] = dict(self._pick)
            out["servo"] = dict(self._servo)
            return out

    def model_meta(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._model_meta)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "viewer_port": self.viewer_port,
                "control_port": self.control_port,
                "viewer_url": f"http://127.0.0.1:{self.viewer_port}/"
                if self.viewer_port
                else "",
                "scene": self.scene_name,
                "scene_source": self._scene_source,
                "model_path": self.model_path,
                "runtime_dir": str(self.runtime_dir),
                "errors": dict(self._errors),
                "pick": dict(self._pick),
                "servo": dict(self._servo),
                "model": dict(self._model_meta),
            }

    def scene_xml(self) -> str:
        with self._lock:
            return self._scene_xml


def _render_png(
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
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera
            )
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
    scratch = mujoco.MjData(model)
    qpos = np.asarray(snapshot.get("qpos", []), dtype=float)
    qvel = np.asarray(snapshot.get("qvel", []), dtype=float)
    if qpos.size == model.nq:
        scratch.qpos[:] = qpos
    if qvel.size == model.nv:
        scratch.qvel[:] = qvel
    mujoco.mj_forward(model, scratch)
    return _render_png(model, scratch, width, height, camera, camera_opts)


def _encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encodes an HxWx3 uint8 RGB frame as JPEG (needs Pillow)."""
    from PIL import Image  # optional dependency

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


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


class ControlServer:
    """Threaded loopback HTTP server exposing the agent control surface."""

    def __init__(
        self,
        state: SharedState,
        handle: viewer_handle.ViewerHandle,
        sim_plugin: Any,
        vision: Any = None,
        render_lock: threading.Lock | None = None,
    ) -> None:
        self._state = state
        self._handle = handle
        self._vision = vision
        self._sim_plugin = sim_plugin
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._render_lock = render_lock or threading.Lock()
        self._stream_stop = threading.Event()
        self.recorder = Recorder()
        self.port = 0

    def start(self, port: int = 0) -> None:
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # noqa: D102
                return

            def _send(self, code: int, payload: Any) -> None:
                body = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:  # noqa: N802
                self._send(200, {"ok": True})

            def _body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length <= 0:
                    return {}
                try:
                    return json.loads(self.rfile.read(length).decode("utf-8"))
                except json.JSONDecodeError:
                    return {}

            def do_GET(self) -> None:  # noqa: N802
                path = urllib.parse.urlparse(self.path).path
                if path in ("/status", "/health"):
                    self._send(200, outer._state.status())
                elif path == "/state":
                    self._send(200, outer._state.snapshot())
                elif path in ("/scenes", "/list-scenes"):
                    from pi_robotics import menagerie, scenes

                    names = sorted(set(scenes.list_scenes()) | set(menagerie.MODELS))
                    self._send(200, {"scenes": names})
                elif path == "/scene":
                    self._send(
                        200,
                        {
                            "scene": outer._state.scene_name,
                            "source": outer._state.status().get("scene_source"),
                            "xml": outer._state.scene_xml(),
                        },
                    )
                elif path == "/cameras":
                    self._send(200, {"cameras": outer.list_cameras()})
                elif path == "/camera":
                    query = urllib.parse.parse_qs(
                        urllib.parse.urlparse(self.path).query
                    )
                    page = outer.camera_page(query)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                elif path == "/stream.mjpg":
                    outer.stream_mjpeg(self)
                else:
                    self._send(404, {"error": f"unknown endpoint {path}"})

            def do_POST(self) -> None:  # noqa: N802
                path = urllib.parse.urlparse(self.path).path
                body = self._body()
                try:
                    result = outer._handle_post(path, body)
                except Exception as exc:  # noqa: BLE001
                    self._send(400, {"ok": False, "error": str(exc)})
                    return
                self._send(200, result)

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="pi-robotics-control", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stream_stop.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    # -- camera streaming ----------------------------------------------------
    def list_cameras(self) -> list[str]:
        model = self._handle.model
        if model is None:
            return []
        return [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            or f"camera{i}"
            for i in range(model.ncam)
        ]

    def camera_page(self, query: dict[str, list[str]]) -> bytes:
        cameras = self.list_cameras()
        requested = (query.get("camera") or [""])[0]
        default = requested or ("camera_rgb" if "camera_rgb" in cameras else (cameras[0] if cameras else ""))
        width = int((query.get("width") or ["480"])[0])
        height = int((query.get("height") or ["360"])[0])
        fps = int((query.get("fps") or ["15"])[0])
        options = "".join(
            f'<option value="{c}"{" selected" if c == default else ""}>{c}</option>'
            for c in cameras
        )
        return CAMERA_PAGE.format(
            options=options or '<option value="">(no cameras in model)</option>',
            camera=default,
            width=width,
            height=height,
            fps=fps,
        ).encode("utf-8")

    def stream_mjpeg(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        """Streams the named camera as multipart/x-mixed-replace MJPEG."""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(handler.path).query)
        camera_name = (query.get("camera") or [""])[0]
        width = min(int((query.get("width") or ["480"])[0]), 640)
        height = min(int((query.get("height") or ["360"])[0]), 480)
        fps = max(1, min(int((query.get("fps") or ["15"])[0]), 60))
        quality = max(10, min(int((query.get("quality") or ["80"])[0]), 95))
        period = 1.0 / fps

        model = self._handle.model
        if model is None:
            handler.send_response(503)
            handler.end_headers()
            return
        cam_id = -1
        if camera_name:
            cam_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
        if cam_id < 0:
            handler.send_response(404)
            handler.send_header("Content-Type", "text/plain")
            handler.end_headers()
            handler.wfile.write(f"no camera '{camera_name}'".encode())
            return

        handler.send_response(200)
        handler.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()

        try:
            renderer = mujoco.Renderer(model, height=height, width=width)
        except Exception as exc:  # noqa: BLE001
            handler.wfile.write(f"renderer unavailable: {exc}\r\n".encode())
            return

        scratch = mujoco.MjData(model)
        try:
            while not self._stream_stop.is_set():
                # Render from the published state snapshot so we never read live
                # MjData while the sim thread is stepping.
                snap = self._state.snapshot()
                qpos = np.asarray(snap.get("qpos", []), dtype=float)
                qvel = np.asarray(snap.get("qvel", []), dtype=float)
                if qpos.size == model.nq:
                    scratch.qpos[:] = qpos
                if qvel.size == model.nv:
                    scratch.qvel[:] = qvel
                mujoco.mj_forward(model, scratch)
                with self._render_lock:
                    renderer.update_scene(scratch, camera=cam_id)
                    frame = renderer.render()
                jpeg = _encode_jpeg(frame, quality)
                handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                handler.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                handler.wfile.write(jpeg)
                handler.wfile.write(b"\r\n")
                handler.wfile.flush()
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            renderer.close()

    # -- request handling ----------------------------------------------------
    def _handle_post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if path == "/command":
            return self._command(body)
        if path == "/load":
            model_path = str(
                pathlib.Path(body.get("path", "")).expanduser().resolve()
            )
            self._sim_plugin.reload_scene(model_path=model_path)
            return {"ok": True, "model_path": model_path}
        if path == "/scene":
            name = body.get("name")
            params = body.get("params") or {}
            self._sim_plugin.reload_scene(scene_name=name, params=params)
            return {"ok": True, "scene": name}
        if path == "/fetch":
            name = body.get("name")
            directory = self._sim_plugin.ensure_scene_model(name)
            return {"ok": True, "name": name, "path": str(directory)}
        if path == "/capture":
            return self._capture(body)
        if path == "/shutdown":
            self._handle._is_running = False  # noqa: SLF001
            return {"ok": True}
        return {"ok": False, "error": f"unknown endpoint {path}"}

    def _command(self, body: dict[str, Any]) -> dict[str, Any]:
        cmd = body.get("cmd") or body.get("command")
        if cmd == "pause":
            paused = bool(body.get("paused", True))
            self._handle.dispatch(
                messages.RequestPauseEvent(
                    pause_state=(
                        sim.PauseState.NORMAL_PAUSED
                        if paused
                        else sim.PauseState.UNPAUSED
                    )
                )
            )
            return {"ok": True, "paused": paused}
        if cmd == "step":
            n = max(1, int(body.get("n", 1)))
            # Ensure paused, then request spaced single steps so none collapse.
            self._handle.dispatch(
                messages.RequestPauseEvent(pause_state=sim.PauseState.NORMAL_PAUSED)
            )
            time.sleep(0.02)
            for _ in range(n):
                self._handle.dispatch(messages.SingleStepEvent())
                time.sleep(0.004)
            return {"ok": True, "steps": n, "time": self._state.snapshot()["time"]}
        if cmd == "speed":
            speed = float(body.get("value", 100.0))
            snap = messages.StepControlSnapshot(speed=speed)
            self._handle.dispatch(snap)
            return {"ok": True, "speed": speed}
        if cmd == "pick":
            object_name = str(body.get("object") or "object")
            # The scripted pick needs the sim running (a prior `step`/`pause`
            # leaves it paused).
            self._handle.dispatch(
                messages.RequestPauseEvent(pause_state=sim.PauseState.UNPAUSED)
            )
            self._sim_plugin.start_pick(object_name=object_name)
            return {"ok": True, "pick": object_name}
        if cmd == "vision_pick":
            if self._vision is None:
                return {"ok": False, "error": "vision servo unavailable"}
            # The servo needs the sim running.
            self._handle.dispatch(
                messages.RequestPauseEvent(pause_state=sim.PauseState.UNPAUSED)
            )
            kwargs: dict[str, Any] = {}
            if body.get("plane_z") is not None:
                kwargs["plane_z"] = float(body["plane_z"])
            return self._vision.start(
                object_body=str(body.get("object") or "object"),
                camera=str(body.get("camera") or "camera_rgb"),
                **kwargs,
            )
        if cmd == "vision_stop":
            if self._vision is not None:
                self._vision.stop()
            return {"ok": True}
        if cmd == "clear":
            self._sim_plugin.set_servo(None)
            return {"ok": True}
        if cmd == "look":
            return self._frame(body)
        if cmd == "nudge":
            return self._nudge(body)
        if cmd == "reel":
            return self._reel(body)
        if cmd == "reel_clear":
            self.recorder.clear()
            return {"ok": True, "steps": 0}
        # Everything else is a data mutation applied on the sim thread.
        self._sim_plugin.enqueue({**body, "cmd": cmd})
        return {"ok": True, "queued": cmd}

    def _camera_and_size(
        self, body: dict[str, Any]
    ) -> tuple[str, int, int, dict[str, Any] | None]:
        camera = str(body.get("camera") or "camera_rgb")
        width = min(int(body.get("width", 480)), 640)
        height = min(int(body.get("height", 360)), 480)
        return camera, width, height, body.get("camera_opts")

    def _collect(self, body: dict[str, Any]) -> dict[str, Any]:
        """Renders a 3D-panel overview + one frame per camera and records them.

        The panel column is a free-camera render of the same scene the 3D panel
        shows; the camera columns are the requested robot cameras. All columns
        share one recorded step so a reel can play them side by side.
        """
        model = self._handle.model
        if model is None:
            return {"ok": False, "error": "no model loaded"}
        snapshot = self._state.snapshot()
        model_cameras = self.list_cameras()
        if body.get("cameras"):
            cameras = [str(c) for c in body["cameras"]]
        else:
            # Default: capture every camera in the model, so a reel always has
            # one column per camera with no blank frames.
            cameras = model_cameras or [str(body.get("camera") or "camera_rgb")]
        requested = str(body.get("camera") or cameras[0])
        width = min(int(body.get("width", 480)), 640)
        height = min(int(body.get("height", 360)), 480)
        cam_pngs: dict[str, bytes] = {}
        for name in cameras:
            try:
                with self._render_lock:
                    cam_pngs[name] = render_png_from_state(
                        model, snapshot, width, height, name, None
                    )
            except Exception:  # noqa: BLE001
                pass
        panel_opts = body.get("panel_opts") or PANEL_CAMERA_OPTS
        panel_w = min(int(body.get("panel_width", PANEL_WIDTH)), 640)
        panel_h = min(int(body.get("panel_height", PANEL_HEIGHT)), 480)
        panel: bytes | None = None
        try:
            with self._render_lock:
                panel = render_png_from_state(
                    model, snapshot, panel_w, panel_h, None, panel_opts
                )
        except Exception:  # noqa: BLE001
            panel = None
        if panel is not None:
            self.recorder.add(panel, cam_pngs)
        main = cam_pngs.get(requested) or (next(iter(cam_pngs.values())) if cam_pngs else None)
        return {
            "ok": main is not None,
            "error": None if main is not None else "render failed",
            "camera": requested,
            "cameras": cameras,
            "model_cameras": model_cameras,
            "png_base64": base64.b64encode(main).decode("ascii") if main else None,
            "panel_base64": base64.b64encode(panel).decode("ascii") if panel else None,
            "recorded": self.recorder.count(),
        }

    def _frame(self, body: dict[str, Any]) -> dict[str, Any]:
        """Renders the panel + camera frame(s) and records one reel step."""
        return self._collect(body)

    def _nudge(self, body: dict[str, Any]) -> dict[str, Any]:
        """Applies relative actuator deltas, settles, and returns a fresh frame.

        This is the primitive for a human/LLM-in-the-loop visual servo: the
        caller moves the arm a little, then looks at the returned image.
        """
        model = self._handle.model
        if model is None:
            return {"ok": False, "error": "no model loaded"}
        ranges: dict[str, tuple[float, float]] = {}
        current: dict[str, float] = {}
        ctrl_list = self._state.snapshot().get("ctrl") or []
        for i in range(model.nu):
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                or f"act{i}"
            )
            lo, hi = model.actuator_ctrlrange[i]
            ranges[name] = (float(lo), float(hi))
            current[name] = float(ctrl_list[i]) if i < len(ctrl_list) else 0.0

        def clamp(name: str, value: float) -> float:
            lo, hi = ranges.get(name, (0.0, 0.0))
            if hi > lo:
                return max(lo, min(hi, value))
            return value

        for key, actuator in (
            ("extend", "arm_extend"),
            ("lift", "lift"),
            ("wrist", "wrist_yaw"),
        ):
            if body.get(key) is not None:
                current[actuator] = clamp(
                    actuator, current.get(actuator, 0.0) + float(body[key])
                )
        if body.get("grip") is not None:
            current["grip"] = clamp("grip", float(body["grip"]))

        self._sim_plugin.set_servo(current)
        time.sleep(max(0.1, min(float(body.get("settle", 0.8)), 5.0)))
        frame = self._frame(body)
        view = {
            k: round(current[k], 4)
            for k in ("lift", "arm_extend", "wrist_yaw", "grip")
            if k in current
        }
        return {
            "ok": frame.get("ok", False),
            "error": frame.get("error"),
            "ctrl": view,
            "png_base64": frame.get("png_base64"),
            "panel_base64": frame.get("panel_base64"),
            "recorded": frame.get("recorded"),
        }

    def _reel(self, body: dict[str, Any]) -> dict[str, Any]:
        hold = body.get("hold")
        try:
            result = self.recorder.build(
                self._state.runtime_dir / "reels",
                fps=float(body.get("fps", 10)),
                hold=None if hold is None else float(hold),
                height=int(body.get("height", 360)),
                name=str(body.get("name", "operation")),
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        result["ok"] = True
        return result

    def _capture(self, body: dict[str, Any]) -> dict[str, Any]:
        model = self._handle.model
        data = self._handle.data
        if model is None or data is None:
            return {"ok": False, "error": "no model loaded"}
        width = int(body.get("width", 640))
        height = int(body.get("height", 480))
        camera = body.get("camera")
        camera_opts = body.get("camera_opts") or {}
        with self._render_lock:
            try:
                png = _render_png(model, data, width, height, camera, camera_opts)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"render failed: {exc}"}
        out_name = body.get("name") or f"frame_{int(time.time() * 1000)}.png"
        out_path = self._state.runtime_dir / "captures" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png)
        return {
            "ok": True,
            "path": str(out_path),
            "base64": base64.b64encode(png).decode("ascii")
            if body.get("inline")
            else None,
        }
