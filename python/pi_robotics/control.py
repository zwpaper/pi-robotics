"""Loopback HTTP control API and shared state for the pi-robotics daemon."""

from __future__ import annotations

import base64
import http.server
import pathlib
import threading
import time
import urllib.parse
from typing import Any

import numpy as np

import mujoco
from mujoco.studio import messages, sim, viewer_handle

from pi_robotics import audit, geometry, ik, motion, vision
from pi_robotics.reel import Recorder, compose_video, condense_static_timeline
from pi_robotics.rendering import (
    CAMERA_PAGE,
    PANEL_CAMERA_OPTS,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    encode_jpeg,
    encode_png,
    render_array,
    render_png,
    render_png_from_state,
    scratch_from_snapshot,
)
from pi_robotics.server import loopback_origin, make_handler

__all__ = [
    "ControlServer",
    "SharedState",
    "PANEL_CAMERA_OPTS",
    "PANEL_HEIGHT",
    "PANEL_WIDTH",
    "render_png_from_state",
]


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
                "sensors": [
                    name_of(mujoco.mjtObj.mjOBJ_SENSOR, i, "sensor")
                    for i in range(model.nsensor)
                ],
                "ctrlrange": {
                    name_of(mujoco.mjtObj.mjOBJ_ACTUATOR, i, "actuator"): [
                        round(float(model.actuator_ctrlrange[i][0]), 4),
                        round(float(model.actuator_ctrlrange[i][1]), 4),
                    ]
                    for i in range(model.nu)
                },
                "joint_limits": {
                    name_of(mujoco.mjtObj.mjOBJ_JOINT, i, "joint"): [
                        round(float(model.jnt_range[i][0]), 4),
                        round(float(model.jnt_range[i][1]), 4),
                    ]
                    for i in range(model.njnt)
                },
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
            "bodies": self._body_positions(model, data),
            "body_quat": self._body_quaternions(model, data),
            "sensordata": self._sensor_values(model, data),
            "contacts": self._contacts(model, data),
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

    def _body_positions(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> dict[str, list[float]]:
        # Body world positions let an agent localize objects/props by name.
        if model.nbody > 200:
            return {}
        out: dict[str, list[float]] = {}
        for bid in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name:
                out[name] = [round(float(v), 4) for v in data.xpos[bid]]
        return out

    def _body_quaternions(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> dict[str, list[float]]:
        # Orientation (w,x,y,z) alongside `bodies`, so a gripper's roll/pitch is
        # visible to the agent (e.g. to judge whether a grasp is level).
        if model.nbody > 200:
            return {}
        out: dict[str, list[float]] = {}
        for bid in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name:
                out[name] = [round(float(v), 4) for v in data.xquat[bid]]
        return out

    def _sensor_values(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> dict[str, Any]:
        # Named sensor readings (scalar or vector). Lets the agent read touch,
        # rangefinder, jointpos sensors directly instead of re-deriving them.
        out: dict[str, Any] = {}
        for sid in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            if not name:
                continue
            adr = int(model.sensor_adr[sid])
            dim = int(model.sensor_dim[sid])
            values = [
                round(float(v), 6) for v in data.sensordata[adr : adr + dim]
            ]
            out[name] = values[0] if dim == 1 else values
        return out

    def _contacts(
        self, model: mujoco.MjModel, data: mujoco.MjData, limit: int = 8
    ) -> list[dict[str, Any]]:
        # Strongest contact pairs first, so a grasp/slip is debuggable from the
        # physics alone (which geoms touch, and how hard).
        if data.ncon == 0:
            return []
        force = np.zeros(6)
        rows: list[dict[str, Any]] = []
        for i in range(data.ncon):
            con = data.contact[i]
            normal = 0.0
            try:
                mujoco.mj_contactForce(model, data, i, force)
                normal = float(force[0])
            except Exception:  # noqa: BLE001
                pass
            rows.append(
                {
                    "geom1": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1
                    )
                    or f"geom{con.geom1}",
                    "geom2": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2
                    )
                    or f"geom{con.geom2}",
                    "dist": round(float(con.dist), 5),
                    "normal": round(normal, 4),
                }
            )
        rows.sort(key=lambda row: abs(row["normal"]), reverse=True)
        return rows[:limit]

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


class ControlServer:
    """Threaded loopback HTTP server exposing the agent control surface."""

    def __init__(
        self,
        state: SharedState,
        handle: viewer_handle.ViewerHandle,
        sim_plugin: Any,
        vision: Any = None,
        render_lock: threading.Lock | None = None,
        workspace: Any = None,
    ) -> None:
        self._state = state
        self._handle = handle
        self._vision = vision
        self._sim_plugin = sim_plugin
        self._workspace = workspace
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._render_lock = render_lock or threading.Lock()
        self._stream_stop = threading.Event()
        self._recorder_lock = threading.Lock()
        self._rec_stop: threading.Event | None = None
        self._rec_thread: threading.Thread | None = None
        # Max seconds of static frames kept per idle run when composing (see
        # `condense_static_timeline`); the raw timeline is always kept too.
        self._idle_cap: float = 1.0
        self._rec_cfg: dict[str, Any] = {}
        self.recorder = Recorder()
        self.port = 0

    def start(self, port: int = 0) -> None:
        Handler = make_handler(self)

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="pi-robotics-control", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stream_stop.set()
        self._record_stop()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    # -- live workspace ------------------------------------------------------
    def workspace_status(self) -> dict[str, Any]:
        """Status of the live Python Studio workspace (empty when none)."""
        if self._workspace is None:
            return {
                "dir": "",
                "active": False,
                "files": {},
                "plugins": [],
                "hot_reload": "",
                "scene_xml": "",
                "errors": {},
            }
        return self._workspace.status()

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
        origin = loopback_origin(handler.headers.get("Origin"))
        if origin:
            handler.send_header("Access-Control-Allow-Origin", origin)
            handler.send_header("Vary", "Origin")
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
                jpeg = encode_jpeg(frame, quality)
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
        if path == "/audit":
            return self._audit(body)
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
            for key in ("site", "body", "gripper"):
                if body.get(key):
                    kwargs[key] = str(body[key])
            for key in ("approach", "lift", "open_value", "close_value"):
                if body.get(key) is not None:
                    kwargs[key] = float(body[key])
            if body.get("quat") is not None:
                kwargs["quat"] = list(body["quat"])
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
        if cmd == "move_ee":
            return self._move_ee(body)
        if cmd == "reel":
            return self._reel(body)
        if cmd == "reel_clear":
            self._record_stop()
            self.recorder.clear()
            return {"ok": True, "steps": 0}
        if cmd == "record_start":
            return self._record_start(body)
        if cmd == "record_stop":
            return self._record_stop()
        if cmd == "chapter":
            count = self.recorder.mark_chapter(
                str(body.get("title") or f"attempt {self.recorder.count() + 1}")
            )
            return {"ok": True, "chapter": count}
        if cmd == "contains":
            return self._contains(body)
        if cmd == "audit":
            return self._audit(body)
        if cmd == "bake":
            return self._bake(body)
        if cmd == "scaffold_workspace":
            return self._scaffold_workspace(body)
        # Everything else is a data mutation applied on the sim thread.
        self._sim_plugin.enqueue({**body, "cmd": cmd})
        return {"ok": True, "queued": cmd}

    # -- workspace commands --------------------------------------------------
    def _bake(self, body: dict[str, Any]) -> dict[str, Any]:
        """Freezes the live scene to ``<workspace>/scene.xml`` and shows it."""
        if self._workspace is None or self._workspace.dir is None:
            return {
                "ok": False,
                "error": "no live workspace; start the daemon with --workspace-dir",
            }
        name = str(body.get("name") or "scene.xml")
        target = self._workspace.dir / name
        try:
            result = self._workspace.bake(self._handle.model, target)
        except Exception as exc:  # noqa: BLE001 - report to the agent
            return {"ok": False, "error": str(exc)}
        # Show the baked file from now on (the workspace plugin declines paths).
        self._sim_plugin.reload_scene(model_path=str(target))
        return result

    def _scaffold_workspace(self, body: dict[str, Any]) -> dict[str, Any]:
        """Writes the starter workspace files (no-op when non-empty)."""
        from pi_robotics import workspace as workspace_mod

        raw = body.get("dir")
        if raw:
            target = pathlib.Path(raw).expanduser().resolve()
        elif self._workspace is not None and self._workspace.dir is not None:
            target = self._workspace.dir
        else:
            return {"ok": False, "error": "no workspace directory given"}
        written = workspace_mod.scaffold_workspace(target)
        return {
            "ok": True,
            "dir": str(target),
            "written": written,
            # Loading needs a restart: Studio registers plugins up front.
            "restart_required": bool(written),
        }

    def _camera_and_size(
        self, body: dict[str, Any]
    ) -> tuple[str, int, int, dict[str, Any] | None]:
        camera = str(body.get("camera") or "camera_rgb")
        width = min(int(body.get("width", 640)), 1280)
        height = min(int(body.get("height", 480)), 960)
        return camera, width, height, body.get("camera_opts")

    def _collect(self, body: dict[str, Any]) -> dict[str, Any]:
        """Renders the requested camera(s) (and, on request, the 3D panel).

        The agent sees only robot camera images; the 3D panel is a free-camera
        render used only for the recorded video (and returned only when
        ``panel: true``). Recording now stores the state timeline, so this no
        longer feeds the reel.
        """
        model = self._handle.model
        if model is None:
            return {"ok": False, "error": "no model loaded"}
        snapshot = self._state.snapshot()
        model_cameras = self.list_cameras()
        requested = str(
            body.get("camera")
            or (model_cameras[0] if model_cameras else "camera_rgb")
        )
        # Only render what is returned (the requested camera(s)).
        render_list = list(
            dict.fromkeys(
                [requested] + [str(c) for c in (body.get("cameras") or [])]
            )
        )
        width = min(int(body.get("width", 640)), 1280)
        height = min(int(body.get("height", 480)), 960)
        cam_arrays: dict[str, np.ndarray] = {}
        cam_error: str | None = None
        scratch = scratch_from_snapshot(model, snapshot)
        for name in render_list:
            try:
                with self._render_lock:
                    cam_arrays[name] = render_array(
                        model, scratch, width, height, name, None
                    )
            except Exception as exc:  # noqa: BLE001
                cam_error = f"{name}: {exc}"
        panel: np.ndarray | None = None
        if body.get("panel"):
            panel_opts = body.get("panel_opts") or PANEL_CAMERA_OPTS
            panel_w = min(int(body.get("panel_width", PANEL_WIDTH)), 1280)
            panel_h = min(int(body.get("panel_height", PANEL_HEIGHT)), 960)
            try:
                with self._render_lock:
                    panel = render_array(
                        model, scratch, panel_w, panel_h, None, panel_opts
                    )
            except Exception:  # noqa: BLE001
                panel = None
        main = cam_arrays.get(requested)
        if main is None and cam_arrays:
            main = next(iter(cam_arrays.values()))

        def png(array: np.ndarray | None) -> str | None:
            return None if array is None else encode_png(array)

        return {
            "ok": main is not None,
            "error": None if main is not None else (cam_error or "render failed"),
            "camera": requested,
            "cameras": render_list,
            "model_cameras": model_cameras,
            "png_base64": png(main),
            "panel_base64": png(panel) if body.get("panel") else None,
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
        current = motion.current_actuators(
            model, self._state.snapshot().get("ctrl")
        )
        motion.apply_manual_deltas(
            model,
            current,
            extend=body.get("extend"),
            lift=body.get("lift"),
            wrist=body.get("wrist"),
            grip=body.get("grip"),
            # Generic control for any robot: {actuator_name: delta}.
            actuators=body.get("actuators"),
        )

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
            "actuators": {k: round(v, 4) for k, v in current.items()},
            "png_base64": frame.get("png_base64"),
            "panel_base64": frame.get("panel_base64"),
            "recorded": frame.get("recorded"),
        }

    def _move_ee(self, body: dict[str, Any]) -> dict[str, Any]:
        """Plans a collision-checked IK path to a world pose and drives it.

        Uses :func:`ik.plan_path`: joint-limit-clamped IK waypoints with an
        optional null-space posture, each collision-checked on a scratch
        ``MjData``. Driving stops before the first waypoint that would create a
        new contact, so the arm does not sweep through a table or itself.
        """
        model = self._handle.model
        if model is None:
            return {"ok": False, "error": "no model loaded"}
        pos = body.get("pos")
        if not pos or len(pos) != 3:
            return {"ok": False, "error": "'pos' [x,y,z] required"}
        steps = max(1, min(int(body.get("steps", 6)), 40))
        total = max(0.2, min(float(body.get("settle", 1.5)), 6.0))
        target = np.asarray(pos, dtype=float)
        quat = body.get("quat")
        # End-effector: explicit site/body, else auto-resolve (site first, then
        # gripper body — Stretch ships no site).
        name = body.get("site") or body.get("body")
        kind = "site" if body.get("site") else ("body" if body.get("body") else None)
        if not name:
            try:
                kind, name = vision.resolve_ee(model)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
        posture = body.get("posture")
        allow_collision = bool(body.get("allow_collision", False))
        ignore_geoms = [int(g) for g in (body.get("ignore_geoms") or [])]

        snapshot = self._state.snapshot()
        qpos = np.asarray(snapshot.get("qpos", []), dtype=float)
        try:
            plan = ik.plan_path(
                model,
                qpos,
                str(name),
                target,
                quat,
                steps=steps,
                kind=kind,
                posture=posture,
                collision=not allow_collision,
                ignore_geoms=ignore_geoms,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

        per_step = total / max(1, len(plan["path"]))
        driven = 0
        last_error: dict[str, Any] = {}
        collision: list[dict[str, Any]] = []
        for entry in plan["path"]:
            if entry["collisions"]:
                collision = entry["collisions"]
                break
            current = motion.current_actuators(model, snapshot.get("ctrl"))
            current.update(
                motion.solution_to_actuators(model, {"joints": entry["joints"]})
            )
            self._sim_plugin.set_servo(current)
            last_error = entry["error"]
            driven += 1
            self._sleep(per_step)

        frame = self._frame(body)
        result: dict[str, Any] = {
            "ok": frame.get("ok", False),
            "error": frame.get("error"),
            "site": str(name),
            "target": [round(float(v), 4) for v in target],
            "ik_error": last_error,
            "steps": driven,
            "planned": plan["steps"],
            "blocked": plan["blocked"],
            "png_base64": frame.get("png_base64"),
            "panel_base64": frame.get("panel_base64"),
            "recorded": frame.get("recorded"),
        }
        if collision:
            result["collision"] = collision
            result["error"] = result["error"] or "motion blocked by collision"
        return result

    def _sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))

    def _record_start(self, body: dict[str, Any]) -> dict[str, Any]:
        """Starts continuous recording of the 3D panel + cameras at a frame rate.

        Unlike per-`look` steps, this samples the whole operation, so the reel is
        a real (smooth) video of the task instead of a key-frame slide show.
        """
        self._record_stop()
        if self._handle.model is None:
            return {"ok": False, "error": "no model loaded"}
        fps = max(1, min(int(body.get("fps", 8)), 30))
        self._idle_cap = max(0.0, float(body.get("idle_cap", self._idle_cap)))
        # The panel (a free-camera render) is only needed for the video backdrop;
        # capture it less often than the cameras so the cameras can run faster.
        panel_fps = max(0.5, min(float(body.get("panel_fps", max(1.0, fps / 4.0))), float(fps)))
        # The reel must show every camera: always capture all model cameras
        # (union with any explicitly requested extras).
        requested = [str(c) for c in (body.get("cameras") or [])]
        cameras = list(dict.fromkeys(requested + self.list_cameras()))
        if not cameras:
            cameras = [str(body.get("camera") or "camera_rgb")]
        width = min(int(body.get("width", 640)), 1280)
        height = min(int(body.get("height", 480)), 960)
        opts = body.get("panel_opts") or PANEL_CAMERA_OPTS
        panel_w = min(int(body.get("panel_width", PANEL_WIDTH)), 1280)
        panel_h = min(int(body.get("panel_height", PANEL_HEIGHT)), 960)
        stop = threading.Event()
        self._rec_stop = stop

        def loop() -> None:
            # Cheap: only samples qpos (no OpenGL), so it can run fast and never
            # starves the control requests. Rendering happens later in `reel`.
            period = 1.0 / fps
            while not stop.is_set():
                model = self._handle.model
                if model is not None:
                    qpos = np.asarray(
                        self._state.snapshot().get("qpos", []), dtype=float
                    )
                    if qpos.size == model.nq:
                        self.recorder.add_state(qpos)
                stop.wait(period)

        self._rec_thread = threading.Thread(
            target=loop, name="pi-robotics-rec", daemon=True
        )
        self._rec_thread.start()
        return {
            "ok": True,
            "recording": True,
            "fps": fps,
            "panel_fps": panel_fps,
            "cameras": cameras,
            "idle_cap": self._idle_cap,
        }

    def _record_stop(self) -> dict[str, Any]:
        stop = self._rec_stop
        thread = self._rec_thread
        self._rec_stop = None
        self._rec_thread = None
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=3.0)
        return {"ok": True, "recording": False}

    def _reel(self, body: dict[str, Any]) -> dict[str, Any]:
        """Renders the recorded state timeline into grid video(s).

        Recording only stored qpos, so the (expensive) multi-camera rendering
        happens here, once, after the run — keeping capture smooth and cheap.
        Runs of static frames longer than ``idle_cap`` seconds are condensed for
        the main video; the untouched timeline is rendered alongside as
        ``<name>_raw.mp4`` so the raw recording is never lost.
        """
        self._record_stop()
        raw_states, chapters = self.recorder.timeline()
        if not raw_states:
            return {"ok": False, "error": "nothing recorded; run record_start first"}
        model = self._handle.model
        if model is None:
            return {"ok": False, "error": "no model loaded"}
        cameras = self.list_cameras() or [str(body.get("camera") or "camera_rgb")]
        width = min(int(body.get("width", 480)), 1280)
        height = min(int(body.get("height", 480)), 960)
        panel_w = min(int(body.get("panel_width", PANEL_WIDTH)), 1280)
        panel_h = min(int(body.get("panel_height", PANEL_HEIGHT)), 960)
        opts = body.get("panel_opts") or PANEL_CAMERA_OPTS
        out_fps = float(body.get("fps", 10))
        name = str(body.get("name", "operation"))
        idle_cap = max(0.0, float(body.get("idle_cap", self._idle_cap)))
        states, condensed_chapters = condense_static_timeline(
            raw_states, chapters, idle_cap
        )
        keep_raw = bool(body.get("keep_raw", True)) and len(states) != len(raw_states)
        # One renderer per camera, created and used on THIS request's thread (a
        # Renderer's GL context is thread-bound) and reused across all frames.
        renderers: dict[str, Any] = {}
        panel_renderer = None

        def build(
            timeline: list[tuple[float, Any]],
            marks: list[tuple[int, str]],
            video_name: str,
        ) -> dict[str, Any]:
            # Decimate by default: render at most `fps` states per second of
            # time, so a long recording does not render every captured state.
            step = 1
            if body.get("decimate", True) and out_fps > 0 and len(timeline) > 1:
                span = max(1e-6, timeline[-1][0] - timeline[0][0])
                step = max(1, int(round((len(timeline) / span) / out_fps)))
            frames: list[dict[str, Any]] = []
            stamps: list[float] = []
            for t, qpos in timeline[::step]:
                scratch = mujoco.MjData(model)
                if qpos.size == model.nq:
                    scratch.qpos[:] = qpos
                cams: dict[str, Any] = {}
                for cam in cameras:
                    try:
                        cams[cam] = render_array(
                            model, scratch, width, height, cam, None,
                            renderer=renderers[cam],
                        )
                    except Exception:  # noqa: BLE001
                        pass
                panel = render_array(
                    model, scratch, panel_w, panel_h, None, opts,
                    renderer=panel_renderer,
                )
                frames.append({"panel": panel, "cameras": cams})
                stamps.append(t)
            composed = compose_video(
                self._state.runtime_dir / "reels",
                frames,
                stamps,
                marks,
                fps=out_fps,
                hold=None if body.get("hold") is None else float(body["hold"]),
                min_hold=float(body.get("min_hold", 0.0)),
                tail=float(body.get("tail", 0.8)),
                height=int(body.get("height", 480)),
                cols=(int(body["cols"]) if body.get("cols") else None),
                name=video_name,
            )
            composed["rendered"] = len(frames)
            return composed

        try:
            for cam in cameras:
                renderers[cam] = mujoco.Renderer(
                    model, height=height, width=width
                )
            panel_renderer = mujoco.Renderer(
                model, height=panel_h, width=panel_w
            )
            result = build(states, condensed_chapters, name)
            raw_result = (
                build(raw_states, chapters, f"{name}_raw") if keep_raw else None
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        finally:
            for renderer in [*renderers.values(), panel_renderer]:
                if renderer is not None:
                    try:
                        renderer.close()
                    except Exception:  # noqa: BLE001
                        pass
        result["ok"] = True
        result["raw_steps"] = len(raw_states)
        result["condensed_steps"] = len(states)
        result["idle_cap"] = idle_cap
        if raw_result is not None:
            result["raw_path"] = raw_result.get("path")
            result["raw_seconds"] = raw_result.get("seconds")
        return result

    def _contains(self, body: dict[str, Any]) -> dict[str, Any]:
        """VLABench-style `contain` condition: is `object` inside `container`?"""
        model = self._handle.model
        if model is None:
            return {"ok": False, "error": "no model loaded"}
        scratch = scratch_from_snapshot(model, self._state.snapshot())
        return geometry.contains(
            model,
            scratch,
            str(body.get("object") or ""),
            str(body.get("container") or body.get("region") or ""),
        )

    def _audit(self, body: dict[str, Any]) -> dict[str, Any]:
        """Runs the headless physics audit on a model (default: the loaded one).

        `scene` audits a built-in or Menagerie scene without loading it; `path`
        audits an MJCF/URDF/MJB file. Both compile a throwaway model, so the live
        simulation is untouched.
        """
        seconds = float(body.get("seconds", audit.DEFAULT_SECONDS))
        settle = bool(body.get("settle", True))
        path = body.get("path")
        scene = body.get("scene")
        source = "loaded"
        try:
            if path:
                resolved = str(pathlib.Path(path).expanduser().resolve())
                model = mujoco.MjModel.from_xml_path(resolved)
                source = resolved
            elif scene:
                from pi_robotics import menagerie, scenes

                if menagerie.is_menagerie_scene(scene):
                    directory = self._sim_plugin.ensure_scene_model(str(scene))
                    xml_file = menagerie.SCENE_FILES.get(
                        str(scene), menagerie.DEFAULT_SCENE_FILE
                    )
                    model = mujoco.MjModel.from_xml_path(str(directory / xml_file))
                else:
                    xml = scenes.build_scene(str(scene), body.get("params") or {})
                    model = mujoco.MjModel.from_xml_string(xml)
                source = f"scene:{scene}"
            else:
                model = self._handle.model
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        if model is None:
            return {"ok": False, "error": "no model loaded"}
        report = audit.audit_model(model, seconds=seconds, settle=settle)
        report["source"] = source
        return report

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
                png = render_png(model, data, width, height, camera, camera_opts)
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
