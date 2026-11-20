"""MuJoCo Studio simulation daemon for pi-robotics.

Launches the MuJoCo Studio *web* viewer (WASM client + Filament renderer in the
browser) in-process, and exposes a loopback HTTP control API so an agent can
load scenes, drive actuators, step, read state, and capture frames while a human
watches the same live 3D scene in the pi-dashboard canvas.

The daemon is meant to be spawned by the pi-robotics extension
(``.pi/extensions/robotics``) as a background process and controlled through
``python/pi_robotics/control.py``.

Never edits files in the workspace: runtime state lives under ``--runtime-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import signal
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np

import mujoco
from mujoco.studio import (
    app as studio_app,
    launch_web,
    messages,
    plugin,
    sim_history,
    step_control,
    viewer_protocol,
)

from pi_robotics import scenes
from pi_robotics.control import ControlServer, SharedState


class RoboticsSimPlugin(plugin.SimPlugin):
    """Sim-thread plugin: builds scenes, applies control commands, publishes state."""

    def __init__(self, state: SharedState) -> None:
        super().__init__()
        self._state = state
        self._scene_name = state.scene_name
        self._scene_params = dict(state.scene_params)
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()

    # -- command ingress (called from the control-server thread) -------------
    def enqueue(self, command: dict[str, Any]) -> None:
        self._commands.put(command)

    # -- model lifecycle -----------------------------------------------------
    @messages.handler(priority=messages.Priority.INTERNAL - 1)
    def on_sim_init(self, _event: plugin.SimInitEvent) -> None:
        # Build the initial scene once the handle exists and other plugins
        # (StepControl, SimHistory) have finished their own initialization.
        self.reload_scene()

    def reload_scene(
        self,
        scene_name: str | None = None,
        params: dict[str, Any] | None = None,
        model_path: str | None = None,
    ) -> None:
        if scene_name is not None:
            self._scene_name = scene_name
        if params is not None:
            self._scene_params = dict(params)
        self._state.scene_name = self._scene_name
        self._state.scene_params = dict(self._scene_params)
        self._state.model_path = model_path or ""
        if model_path:
            self.handle.dispatch(messages.LoadModelEvent(path=model_path))
        else:
            # A sentinel path signals "build the Python scene" to on_load_model.
            self.handle.dispatch(messages.LoadModelEvent(path=""))

    @messages.handler
    def on_load_model(self, event: messages.LoadModelEvent) -> bool:
        try:
            if event.path:
                model = mujoco.MjModel.from_xml_path(event.path)
                source = os.path.basename(event.path)
            else:
                xml = scenes.build_scene(self._scene_name, self._scene_params)
                model = mujoco.MjModel.from_xml_string(xml)
                source = f"scene:{self._scene_name}"
            self.handle.set_model(model)
            self._state.set_scene_source(source, xml if not event.path else "")
            return True
        except Exception:  # noqa: BLE001 - surface to the agent, keep daemon alive
            self._state.record_error("scene", traceback.format_exc())
            return False

    @messages.handler(priority=messages.Priority.CRITICAL)
    def on_post_model(self, event: messages.PostModelEvent) -> None:
        self._state.publish_model(event.model)
        self._state.record_error(None, None)

    # -- per-step command application ---------------------------------------
    @messages.handler
    def on_pre_step(self, event: messages.PreStepEvent) -> None:
        model, data = event.model, event.data
        applied: list[str] = []
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                break
            try:
                self._apply_command(model, data, cmd, applied)
            except Exception:  # noqa: BLE001
                self._state.record_error("command", traceback.format_exc())
        if applied:
            mujoco.mj_forward(model, data)

    def _apply_command(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cmd: dict[str, Any],
        applied: list[str],
    ) -> None:
        kind = cmd.get("cmd")
        if kind == "ctrl":
            values = np.asarray(cmd.get("values", []), dtype=float)
            if values.size:
                n = min(values.size, model.nu)
                data.ctrl[:n] = values[:n]
                applied.append(f"ctrl[{n}]")
        elif kind == "ctrl_named":
            name = str(cmd.get("name", ""))
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise KeyError(f"unknown actuator '{name}'")
            data.ctrl[aid] = float(cmd.get("value", 0.0))
            applied.append(f"ctrl:{name}")
        elif kind == "qpos":
            values = np.asarray(cmd.get("values", []), dtype=float)
            n = min(values.size, model.nq)
            data.qpos[:n] = values[:n]
            applied.append(f"qpos[{n}]")
        elif kind == "qvel":
            values = np.asarray(cmd.get("values", []), dtype=float)
            n = min(values.size, model.nv)
            data.qvel[:n] = values[:n]
            applied.append(f"qvel[{n}]")
        elif kind == "reset":
            key = cmd.get("key")
            if key is not None and int(key) >= 0:
                mujoco.mj_resetDataKeyframe(model, data, int(key))
            else:
                mujoco.mj_resetData(model, data)
            applied.append("reset")
        else:
            raise ValueError(f"unknown command '{kind}'")

    # -- state publication ---------------------------------------------------
    @messages.handler(priority=messages.Priority.LIBRARY)
    def on_post_step(self, event: messages.PostStepEvent) -> None:
        self._state.publish_data(event.model, event.data)


class PortCapturePlugin(plugin.ViewerPlugin):
    """Viewer-side plugin that records the bound web-viewer HTTP port."""

    def __init__(self, state: SharedState) -> None:
        super().__init__()
        self._state = state

    @messages.handler(priority=messages.Priority.CRITICAL)
    def on_viewer_init(self, event: plugin.ViewerInitEvent) -> None:
        port = int(
            getattr(event.viewer, "http_port", 0)
            or getattr(event.viewer, "port", 0)
            or getattr(event.viewer, "_http_port", 0)
        )
        self._state.set_viewer_port(port)


class AppInitPlugin(plugin.ViewerPlugin):
    """Viewer-side plugin that trims the Studio chrome for a compact panel."""

    def __init__(self, compact: bool) -> None:
        super().__init__()
        self._compact = compact

    @messages.handler
    def on_app_init(self, event: studio_app.InitEvent) -> None:
        if self._compact and hasattr(event.app, "config"):
            event.app.config.show_options = False
            event.app.config.show_inspector = False


def _write_session_file(
    path: pathlib.Path, payload: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run(args: argparse.Namespace) -> int:
    runtime_dir = pathlib.Path(args.runtime_dir).expanduser().resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    session_file = (
        pathlib.Path(args.session_file).expanduser().resolve()
        if args.session_file
        else runtime_dir / "session.json"
    )

    state = SharedState(
        runtime_dir=runtime_dir,
        scene_name=args.scene or scenes.DEFAULT_SCENE,
        scene_params={},
        viewer_port=0,
        control_port=0,
    )

    if args.model:
        model_path = str(pathlib.Path(args.model).expanduser().resolve())
        state.model_path = model_path
    else:
        model_path = None

    sim_plugin = RoboticsSimPlugin(state)

    plugins: list[Any] = [
        AppInitPlugin(compact=args.compact),
        PortCapturePlugin(state),
        sim_plugin,
        step_control.StepControl(),
        sim_history.SimHistory(),
    ]
    if not args.no_ui:
        plugins.insert(0, studio_app.App(studio_app.Config()))

    config = viewer_protocol.ViewerConfig(
        title="pi-robotics · MuJoCo Studio",
        gfx="web",
        http_port=args.http_port,
        open_browser=False,
    )

    exit_code = 0
    with launch_web.launch(config, plugins=plugins) as handle:
        # Initial model: file path if given, otherwise the Python scene.
        if model_path:
            sim_plugin.reload_scene(model_path=model_path)
        else:
            sim_plugin.reload_scene(scene_name=state.scene_name)

        # Wait briefly for the viewer port and a compiled model.
        deadline = time.time() + 15.0
        while time.time() < deadline:
            if state.viewer_port and handle.model is not None:
                break
            if state.has_error("scene"):
                print("[pi-robotics] scene load failed", file=sys.stderr)
                break
            handle.sync()
            time.sleep(0.02)
        if not state.viewer_port:
            print("[pi-robotics] viewer did not bind a port", file=sys.stderr)

        control = ControlServer(state, handle, sim_plugin)
        control.start()
        state.control_port = control.port

        payload = {
            "pid": os.getpid(),
            "viewer_port": state.viewer_port,
            "control_port": control.port,
            "viewer_url": f"http://127.0.0.1:{state.viewer_port}/",
            "runtime_dir": str(runtime_dir),
            "scene": state.scene_name,
            "model_path": state.model_path,
            "started_at": time.time(),
        }
        _write_session_file(session_file, payload)
        print(
            "PI_ROBOTICS_READY " + json.dumps(payload), flush=True
        )

        try:
            # Own loop (rather than viewer_handle.run_sim_loop) so SIGTERM and
            # SIGINT stop cleanly and remove the session file.
            stop_event = threading.Event()

            def _on_signal(_signum: int, _frame: Any) -> None:
                stop_event.set()

            signal.signal(signal.SIGTERM, _on_signal)
            signal.signal(signal.SIGINT, _on_signal)
            while handle.is_running() and not stop_event.is_set():
                handle.sync()
        except KeyboardInterrupt:
            pass
        finally:
            control.stop()
            try:
                session_file.unlink()
            except FileNotFoundError:
                pass

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MuJoCo Studio daemon for the pi-robotics extension."
    )
    parser.add_argument("--scene", default="", help="Built-in scene name.")
    parser.add_argument("--model", default="", help="Model file (XML/URDF/MJB).")
    parser.add_argument("--http-port", type=int, default=0, help="Viewer port (0=auto).")
    parser.add_argument("--control-port", type=int, default=0, help="Control port (0=auto).")
    parser.add_argument("--runtime-dir", default=".pi/robotics/runtime")
    parser.add_argument("--session-file", default="")
    parser.add_argument("--compact", action="store_true", default=True)
    parser.add_argument("--no-ui", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
