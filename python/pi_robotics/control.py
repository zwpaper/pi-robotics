"""Loopback HTTP control API and shared state for the pi-robotics daemon."""

from __future__ import annotations

import base64
import http.server
import json
import pathlib
import threading
import time
import urllib.parse
from typing import Any

import numpy as np

import mujoco
from mujoco.studio import messages, sim, viewer_handle


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
) -> bytes:
    """Offscreen render to PNG bytes (best effort; needs a GL context)."""
    import imageio.v3 as iio  # optional dependency

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        if camera:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, camera
            )
            renderer.update_scene(data, camera=cam)
        else:
            renderer.update_scene(data)
        frame = renderer.render()
    finally:
        renderer.close()
    return iio.imwrite("<bytes>", frame, extension=".png")


class ControlServer:
    """Threaded loopback HTTP server exposing the agent control surface."""

    def __init__(
        self,
        state: SharedState,
        handle: viewer_handle.ViewerHandle,
        sim_plugin: Any,
    ) -> None:
        self._state = state
        self._handle = handle
        self._sim_plugin = sim_plugin
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._render_lock = threading.Lock()
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
                    from pi_robotics import scenes

                    self._send(200, {"scenes": scenes.list_scenes()})
                elif path == "/scene":
                    self._send(
                        200,
                        {
                            "scene": outer._state.scene_name,
                            "source": outer._state.status().get("scene_source"),
                            "xml": outer._state.scene_xml(),
                        },
                    )
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
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

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
        # Everything else is a data mutation applied on the sim thread.
        self._sim_plugin.enqueue({**body, "cmd": cmd})
        return {"ok": True, "queued": cmd}

    def _capture(self, body: dict[str, Any]) -> dict[str, Any]:
        model = self._handle.model
        data = self._handle.data
        if model is None or data is None:
            return {"ok": False, "error": "no model loaded"}
        width = int(body.get("width", 640))
        height = int(body.get("height", 480))
        camera = body.get("camera")
        with self._render_lock:
            try:
                png = _render_png(model, data, width, height, camera)
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
