"""MuJoCo Studio simulation daemon for pi-robotics.

Launches the MuJoCo Studio *web* viewer (WASM client + Filament renderer in the
browser) in-process, and exposes a loopback HTTP control API so an agent can
load scenes, drive actuators, step, read state, and capture frames while a human
watches the same live 3D scene in the pi-dashboard canvas.

The daemon is meant to be spawned by the pi-robotics extension
(``.pi/extensions/robotics``) as a background process and controlled through
``python/pi_robotics/control.py``.

Never edits files in the workspace: runtime state lives under ``--runtime-dir``.

The one exception is the explicit ``--scaffold`` mode (and the ``scaffold_workspace``
control command), which writes the starter Live Python Studio workspace files at
the user's request; the daemon never writes them on its own.
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

from pi_robotics import menagerie, scenes
from pi_robotics import workspace as workspace_mod
from pi_robotics.control import ControlServer, SharedState
from pi_robotics.vision import VisionServo

# Default head pose so the RealSense camera looks at the tabletop workspace.
STRETCH_LOOK_AT: dict[str, float] = {"head_pan": -1.5, "head_tilt": -0.9}

# Scripted pick for the Hello Robot Stretch tabletop scene: approach the object,
# close the gripper, lift, retract, lift clear. Actuator values are the
# calibrated targets (see the repo README / calibration notes).
STRETCH_PICK_PROGRAM: list[tuple[float, dict[str, float]]] = [
    # Let the object settle on the table before approaching (arm retracted,
    # gripper open, at the calibration start pose).
    (2.5, {"lift": 0.0, "arm_extend": 0.0, "wrist_yaw": 0.0, "grip": 0.04, **STRETCH_LOOK_AT}),
    (1.6, {"lift": -0.07, "arm_extend": 0.28, "wrist_yaw": 0.15, "grip": 0.04, **STRETCH_LOOK_AT}),
    (1.3, {"lift": -0.07, "arm_extend": 0.28, "wrist_yaw": 0.15, "grip": -0.005, **STRETCH_LOOK_AT}),
    (2.0, {"lift": 0.18, "arm_extend": 0.28, "wrist_yaw": 0.15, "grip": -0.005, **STRETCH_LOOK_AT}),
    (2.0, {"lift": 0.40, "arm_extend": 0.28, "wrist_yaw": 0.15, "grip": -0.005, **STRETCH_LOOK_AT}),
]


class RoboticsSimPlugin(plugin.SimPlugin):
    """Sim-thread plugin: builds scenes, applies control commands, publishes state."""

    def __init__(
        self,
        state: SharedState,
        models_root: pathlib.Path,
        workspace_owned: bool = False,
    ) -> None:
        super().__init__()
        self._state = state
        self._models_root = models_root
        self._workspace_owned = workspace_owned
        self._scene_name = state.scene_name
        self._scene_params = dict(state.scene_params)
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._pick: dict[str, Any] | None = None
        self._servo: dict[str, float] | None = None
        self._servo_lock = threading.Lock()

    def set_servo(self, ctrl: dict[str, float] | None) -> None:
        """Sets (or clears) the actuator override written by the vision servo."""
        with self._servo_lock:
            self._servo = None if ctrl is None else dict(ctrl)

    # -- command ingress (called from the control-server thread) -------------
    def enqueue(self, command: dict[str, Any]) -> None:
        self._commands.put(command)

    def ensure_scene_model(self, name: str) -> pathlib.Path:
        """Fetches (if needed) and returns a Menagerie model directory."""
        return menagerie.ensure_model(self._models_root, name)

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
    ) -> bool:
        if scene_name is not None:
            self._scene_name = scene_name
        if params is not None:
            self._scene_params = dict(params)
        self._state.scene_name = self._scene_name
        self._state.scene_params = dict(self._scene_params)
        self._state.model_path = model_path or ""
        if model_path:
            self.handle.dispatch(messages.LoadModelEvent(path=model_path))
            return True
        if menagerie.is_menagerie_scene(self._scene_name):
            # Built-in Menagerie robot scene (e.g. Hello Robot Stretch): fetch on
            # first use, then load its scene.xml.
            try:
                directory = menagerie.ensure_model(self._models_root, self._scene_name)
            except Exception as exc:  # noqa: BLE001
                self._state.record_error("scene", str(exc))
                return False
            self.handle.dispatch(
                messages.LoadModelEvent(path=str(directory / "scene.xml"))
            )
            return True
        if (
            self._workspace_owned
            and scene_name is not None
            and scene_name != workspace_mod.WORKSPACE_SCENE
        ):
            # A live workspace owns ``LoadModelEvent(path="")``; refuse to build
            # a built-in Python scene over it (the workspace plugin built first).
            self._state.record_error(
                "scene",
                "a live workspace owns the scene; bake it and 'load' the file, "
                "or start the daemon without --workspace-dir for built-in scenes",
            )
            return False
        # A sentinel path signals "build the Python scene" to on_load_model. In a
        # workspace the workspace plugin consumes it first and builds its scene.
        self.handle.dispatch(messages.LoadModelEvent(path=""))
        return True

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
        # Enable larger offscreen renders (reels/captures) than the 640x480 default.
        try:
            global_ = event.model.vis.global_
            global_.offwidth = max(int(global_.offwidth), 1280)
            global_.offheight = max(int(global_.offheight), 960)
        except Exception:  # noqa: BLE001
            pass
        self._state.publish_model(event.model)
        self._state.record_error(None, None)
        if self._pick is None:
            self._state.clear_pick()
        self._apply_home_posture(event.model, event.data)

    def _apply_home_posture(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> None:
        """Points the head camera at the workspace on the Stretch scene."""
        if self._scene_name not in menagerie.MODELS:
            return
        joints = {"head_pan": "joint_head_pan", "head_tilt": "joint_head_tilt"}
        for name, value in STRETCH_LOOK_AT.items():
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                data.ctrl[aid] = value
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joints.get(name, name)
            )
            if jid >= 0:
                # Move the head immediately so the camera is aimed right away.
                data.qpos[model.jnt_qposadr[jid]] = value
        mujoco.mj_forward(model, data)

    # -- scripted manipulation ----------------------------------------------
    def start_pick(
        self,
        object_name: str = "object",
        program: list[tuple[float, dict[str, float]]] | None = None,
    ) -> None:
        self._pick = {
            "object": object_name,
            "program": program or STRETCH_PICK_PROGRAM,
            "index": 0,
            "phase_start": None,
            "start_z": None,
        }
        self._state.set_pick({"active": True, "object": object_name, "phase": 0})

    def _advance_pick(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        pick = self._pick
        assert pick is not None
        program = pick["program"]
        index = pick["index"]
        object_name = pick["object"]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if body_id < 0:
            self._state.set_pick(
                {"active": False, "error": f"unknown body '{object_name}'"}
            )
            self._pick = None
            return
        if pick["phase_start"] is None:
            pick["phase_start"] = float(data.time)
            pick["start_z"] = float(data.xpos[body_id][2])
        # Track the object height during the initial settle phase so the
        # reported lift is measured from its resting height, not its spawn.
        if index == 0:
            pick["start_z"] = float(data.xpos[body_id][2])

        duration, targets = program[index]
        data.ctrl[:] = 0.0
        for name, value in targets.items():
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                data.ctrl[aid] = value

        if float(data.time) - float(pick["phase_start"]) >= duration:
            pick["index"] = index + 1
            pick["phase_start"] = float(data.time)
            if pick["index"] >= len(program):
                end_z = float(data.xpos[body_id][2])
                start_z = float(pick["start_z"] or end_z)
                self._state.set_pick(
                    {
                        "active": False,
                        "done": True,
                        "object": object_name,
                        "start_z": round(start_z, 4),
                        "end_z": round(end_z, 4),
                        "lifted": round(end_z - start_z, 4),
                    }
                )
                self._pick = None
            else:
                self._state.set_pick(
                    {"active": True, "object": object_name, "phase": pick["index"]}
                )

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
        with self._servo_lock:
            servo = None if self._servo is None else dict(self._servo)
        if servo is not None:
            data.ctrl[:] = 0.0
            for name, value in servo.items():
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid >= 0:
                    data.ctrl[aid] = value
            applied.append("servo")
        elif self._pick is not None:
            self._advance_pick(model, data)
            applied.append("pick")
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
            self._pick = None
            self._state.clear_pick()
            self._apply_home_posture(model, data)
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

    models_root = runtime_dir.parent / "models"

    # A Menagerie scene may need a one-time download; do it before the viewer
    # starts so startup stays the same afterward.
    if not model_path and menagerie.is_menagerie_scene(state.scene_name):
        try:
            menagerie.ensure_model(models_root, state.scene_name)
        except Exception as exc:  # noqa: BLE001
            print(f"[pi-robotics] model fetch failed: {exc}", file=sys.stderr)

    # A live Python Studio workspace (optional): load its plugin modules before
    # the built-in sim plugin so they get first refusal on the scene, and let
    # Studio's own hot reload watch their files for saves.
    workspace = workspace_mod.Workspace()
    if args.workspace_dir:
        workspace = workspace_mod.load_workspace_plugins(
            pathlib.Path(args.workspace_dir).expanduser().resolve()
        )
        for component, error in workspace.errors.items():
            print(
                f"[pi-robotics] workspace {component} failed to load:\n{error}",
                file=sys.stderr,
            )

    workspace_owned = workspace.has_sim_plugin
    if workspace_owned and not model_path and not args.scene:
        # Let the workspace build the initial scene.
        state.scene_name = workspace_mod.WORKSPACE_SCENE

    sim_plugin = RoboticsSimPlugin(
        state, models_root, workspace_owned=workspace_owned
    )
    render_lock = threading.Lock()

    plugins: list[Any] = [
        AppInitPlugin(compact=args.compact),
        PortCapturePlugin(state),
        *workspace.plugins,
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
        # Reload a saved model file, plus the workspace plugin modules in place.
        hot_reload=workspace.hot_reload_spec(),
    )

    exit_code = 0
    with launch_web.launch(config, plugins=plugins) as handle:
        vision = VisionServo(state, handle, sim_plugin, render_lock)
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

        control = ControlServer(
            state,
            handle,
            sim_plugin,
            vision=vision,
            render_lock=render_lock,
            workspace=workspace,
        )
        control.start(args.control_port)
        state.control_port = control.port

        payload = {
            "pid": os.getpid(),
            "viewer_port": state.viewer_port,
            "control_port": control.port,
            "viewer_url": f"http://127.0.0.1:{state.viewer_port}/",
            "runtime_dir": str(runtime_dir),
            "scene": state.scene_name,
            "model_path": state.model_path,
            "workspace_dir": str(workspace.dir) if workspace.dir is not None else "",
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
    parser.add_argument(
        "--workspace-dir",
        default="",
        help="Live Python Studio workspace (scaffold + hot-reload plugins).",
    )
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Write the starter workspace files and exit.",
    )
    parser.add_argument("--compact", action="store_true", default=True)
    parser.add_argument("--no-ui", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scaffold:
        target = (
            pathlib.Path(args.workspace_dir).expanduser().resolve()
            if args.workspace_dir
            else workspace_mod.default_workspace_dir(pathlib.Path.cwd())
        )
        written = workspace_mod.scaffold_workspace(target)
        print(
            "PI_ROBOTICS_SCAFFOLD "
            + json.dumps({"dir": str(target), "written": written}),
            flush=True,
        )
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
