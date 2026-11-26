"""Vision-based closed-loop grasping for pi-robotics (robot-agnostic).

Runs on its own thread using the robot's own camera:

  1. **Camera** — renders a model camera offscreen with MuJoCo's segmentation
     renderer, from the published state snapshot (never the live ``MjData``).
  2. **Detect** — finds the target object's image centroid from the segmentation
     mask (per-geom, so it is robust to colour and lighting). The pixel is
     back-projected through the camera model onto the tabletop plane to get the
     object's world position: *look-then-move*, position-based visual servoing.
  3. **Control** — uses :mod:`pi_robotics.ik` to drive a configurable
     end-effector (a named site *or* body) onto the camera-localized object in
     ``steps`` waypoints, closes the gripper actuator, then lifts with a second
     IK target. No robot-specific actuator names are hard-coded: the EE and
     gripper are auto-detected (or passed in), and the joint targets are mapped
     to actuators by the model's own transmissions.

The sim plugin applies the servo's actuator targets every step; nothing here
touches the live ``MjData``.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

import mujoco

from pi_robotics import ik, motion

# End-effector candidates, in preference order, when the caller does not name
# one. Sites first (Panda/UR5e/FR3 ship `attachment_site`), then gripper bodies
# (Stretch ships no EE site, so it uses `link_gripper_slider`).
DEFAULT_EE_SITES = (
    "attachment_site",
    "pinch_site",
    "grasp_site",
    "ee_site",
    "end_effector",
    "tool_site",
    "tcp",
)
DEFAULT_EE_BODIES = (
    "link_gripper_slider",
    "link_gripper_finger_left",
    "gripper",
    "hand",
    "tool0",
    "ee_link",
    "tcp",
    "panda_hand",
    "fr3_link8",
    "wrist_3_link",
)

# Gripper actuator candidates when the caller does not name one.
DEFAULT_GRIPPERS = (
    "grip",
    "gripper",
    "panda_finger_joint1",
    "fr3_finger_joint1",
    "finger_joint",
)

# Head pose so the camera looks at the tabletop workspace (Stretch only).
HEAD_LOOK_AT = {"head_pan": -1.5, "head_tilt": -0.9}

# Tabletop object centre height (metres) for the back-projection plane.
DEFAULT_PLANE_Z = 0.52

# Alignment tolerance (metres) for the EE/site servo.
ALIGN_TOL = 0.03


def _name2id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    return int(mujoco.mj_name2id(model, objtype, name))


def resolve_ee(
    model: mujoco.MjModel, site: str | None = None, body: str | None = None
) -> tuple[str, str]:
    """Resolves the end-effector to ``("site"|"body", name)``.

    Prefers an explicitly named site/body, then the known site/body candidates,
    then the first named site in the model. Raises ``ValueError`` if none exist.
    """
    if site:
        if _name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
            raise ValueError(f"unknown end-effector site '{site}'")
        return ("site", site)
    if body:
        if _name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) < 0:
            raise ValueError(f"unknown end-effector body '{body}'")
        return ("body", body)
    for name in DEFAULT_EE_SITES:
        if _name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0:
            return ("site", name)
    for name in DEFAULT_EE_BODIES:
        if _name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
            return ("body", name)
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name:
            return ("site", name)
    raise ValueError("no end-effector site or body found; pass site= or body=")


def resolve_gripper(model: mujoco.MjModel, name: str | None = None) -> str | None:
    """Resolves the gripper actuator name, or ``None`` if the model has none."""
    if name:
        return name if _name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0 else None
    for candidate in DEFAULT_GRIPPERS:
        if _name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate) >= 0:
            return candidate
    return None


def head_ctrl(model: mujoco.MjModel) -> dict[str, float]:
    """Head-pose targets that exist in this model (Stretch only, else empty)."""
    return {
        name: value
        for name, value in HEAD_LOOK_AT.items()
        if _name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0
    }


class VisionServo:
    """Perceive -> detect -> IK-control loop that grasps the target object."""

    def __init__(
        self,
        state: Any,
        handle: Any,
        sim_plugin: Any,
        render_lock: threading.Lock,
    ) -> None:
        self._state = state
        self._handle = handle
        self._sim = sim_plugin
        self._render_lock = render_lock
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle -----------------------------------------------------------
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        object_body: str = "object",
        camera: str = "camera_rgb",
        *,
        plane_z: float = DEFAULT_PLANE_Z,
        width: int = 320,
        height: int = 240,
        site: str | None = None,
        body: str | None = None,
        gripper: str | None = None,
        open_value: float = 0.04,
        close_value: float = -0.005,
        approach: float = 0.1,
        lift: float = 0.2,
        steps: int = 6,
        quat: list[float] | None = None,
    ) -> dict[str, Any]:
        if self.busy():
            return {"ok": False, "error": "a vision servo is already running"}
        model = self._handle.model
        if model is None:
            return {"ok": False, "error": "no model loaded"}
        if _name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body) < 0:
            return {"ok": False, "error": f"unknown object body '{object_body}'"}
        cam_id = _name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            return {"ok": False, "error": f"unknown camera '{camera}'"}
        try:
            ee_kind, ee_name = resolve_ee(model, site, body)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        gripper_name = resolve_gripper(model, gripper)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(
                object_body, cam_id, plane_z, width, height,
                ee_kind, ee_name, gripper_name,
                open_value, close_value, approach, lift, steps, quat,
            ),
            name="pi-robotics-vision",
            daemon=True,
        )
        self._thread.start()
        return {
            "ok": True,
            "servo": object_body,
            "camera": camera,
            "end_effector": f"{ee_kind}:{ee_name}",
            "gripper": gripper_name,
        }

    def stop(self) -> None:
        self._stop.set()

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _geoms_of(model: mujoco.MjModel, body_id: int) -> set[int]:
        return {g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id}

    def _sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while not self._stop.is_set() and time.time() < end:
            time.sleep(0.02)

    def _set_ctrl(self, ctrl: dict[str, float]) -> None:
        self._sim.set_servo(dict(ctrl))

    def _report(self, **info: Any) -> None:
        self._state.set_servo(info)

    # -- main loop -----------------------------------------------------------
    def _run(self, *args: Any) -> None:
        try:
            self._run_inner(*args)
        except Exception as exc:  # noqa: BLE001
            self._sim.set_servo(None)
            self._report(active=False, done=True, error=str(exc))

    def _run_inner(
        self,
        object_body: str,
        cam_id: int,
        plane_z: float,
        width: int,
        height: int,
        ee_kind: str,
        ee_name: str,
        gripper_name: str | None,
        open_value: float,
        close_value: float,
        approach: float,
        lift: float,
        steps: int,
        quat: list[float] | None,
    ) -> None:
        model = self._handle.model
        assert model is not None
        object_id = _name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        object_geoms = self._geoms_of(model, object_id)
        if not object_geoms:
            self._finish(error="object body has no geoms")
            return
        ee_objtype = (
            mujoco.mjtObj.mjOBJ_SITE if ee_kind == "site" else mujoco.mjtObj.mjOBJ_BODY
        )
        ee_id = _name2id(model, ee_objtype, ee_name)
        head = head_ctrl(model)

        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.enable_segmentation_rendering()
        scratch = mujoco.MjData(model)
        try:
            fovy = float(model.cam_fovy[cam_id]) or 45.0
            focal = 0.5 * height / math.tan(math.radians(fovy) / 2.0)
            cx, cy = width / 2.0, height / 2.0

            def sync() -> None:
                snap = self._state.snapshot()
                qpos = np.asarray(snap.get("qpos", []), dtype=float)
                qvel = np.asarray(snap.get("qvel", []), dtype=float)
                if qpos.size == model.nq:
                    scratch.qpos[:] = qpos
                if qvel.size == model.nv:
                    scratch.qvel[:] = qvel
                mujoco.mj_forward(model, scratch)

            def ee_world() -> np.ndarray:
                sync()
                if ee_kind == "site":
                    return np.asarray(scratch.site_xpos[ee_id], dtype=float).copy()
                return np.asarray(scratch.xpos[ee_id], dtype=float).copy()

            def object_pixel() -> np.ndarray | None:
                sync()
                with self._render_lock:
                    renderer.update_scene(scratch, camera=cam_id)
                    ids = renderer.render()[..., 0]
                mask = np.isin(ids, list(object_geoms))
                ys, xs = np.nonzero(mask)
                if len(xs) < 15:
                    return None
                return np.array([float(xs.mean()), float(ys.mean())])

            def back_project(u: float, v: float) -> np.ndarray | None:
                sync()
                pos = scratch.cam_xpos[cam_id].copy()
                mat = scratch.cam_xmat[cam_id].reshape(3, 3)
                direction = mat @ np.array(
                    [(u - cx) / focal, -(v - cy) / focal, -1.0]
                )
                if abs(direction[2]) < 1e-6:
                    return None
                t = (plane_z - pos[2]) / direction[2]
                return pos + t * direction

            def ik_ctrl(point: np.ndarray, current_qpos: np.ndarray) -> dict[str, float]:
                solution = ik.solve_target(
                    model, current_qpos, ee_name, point, quat, kind=ee_kind
                )
                ctrl = dict(head)
                ctrl.update(motion.solution_to_actuators(model, solution))
                return ctrl

            def object_z_now() -> float:
                sync()
                return float(scratch.xpos[object_id][2])

            # 1) Camera: settle, then localize the object on the tabletop plane.
            opened = {**head}
            if gripper_name is not None:
                opened[gripper_name] = float(open_value)
            self._set_ctrl(opened)
            self._sleep(1.2)
            pixel = object_pixel()
            if pixel is None:
                self._finish(error="object not visible")
                return
            target = back_project(pixel[0], pixel[1])
            if target is None:
                self._finish(error="back-projection failed")
                return
            self._report(
                active=True,
                phase="localized",
                end_effector=f"{ee_kind}:{ee_name}",
                gripper=gripper_name,
                object_pixel=[round(float(pixel[0]), 1), round(float(pixel[1]), 1)],
                object_world=[round(float(v), 3) for v in target],
            )

            # 2) Control: descend from an approach height onto the object in IK
            #    waypoints; the EE/site is driven with the model's own actuators.
            aligned = False
            history: list[dict[str, Any]] = []
            grasp = np.asarray(target, dtype=float)
            for it in range(max(1, steps)):
                point = grasp + np.array(
                    [0.0, 0.0, approach * (1.0 - (it + 1) / max(1, steps))]
                )
                snap = self._state.snapshot()
                qpos = np.asarray(snap.get("qpos", []), dtype=float)
                try:
                    ctrl = ik_ctrl(point, qpos)
                except ValueError as exc:
                    self._finish(error=str(exc))
                    return
                self._set_ctrl(ctrl)
                self._sleep(0.6)
                grip = ee_world()
                ey = float(grasp[1] - grip[1])
                ez = float(grasp[2] - grip[2])
                history.append(
                    {"iteration": it, "ey": round(ey, 4), "ez": round(ez, 4)}
                )
                self._report(
                    active=True,
                    phase="servo",
                    iteration=it,
                    target=[round(float(v), 3) for v in grasp],
                    end_effector=[round(float(v), 3) for v in grip],
                    error_m=[round(ey, 4), round(ez, 4)],
                )
                if abs(ey) < ALIGN_TOL and abs(ez) < ALIGN_TOL:
                    aligned = True
                    break

            # 3) Close the gripper, then lift to a fresh IK target.
            if gripper_name is not None:
                self._report(active=True, phase="close", iterations=len(history))
                snap = self._state.snapshot()
                ctrl = ik_ctrl(grasp, np.asarray(snap.get("qpos", []), dtype=float))
                ctrl[gripper_name] = float(close_value)
                self._set_ctrl(ctrl)
                self._sleep(1.2)
            object_z0 = object_z_now()
            self._report(active=True, phase="lift", iterations=len(history))
            lift_target = grasp + np.array([0.0, 0.0, float(lift)])
            snap = self._state.snapshot()
            try:
                self._set_ctrl(
                    ik_ctrl(lift_target, np.asarray(snap.get("qpos", []), dtype=float))
                )
            except ValueError:
                pass
            self._sleep(2.0)
            object_z1 = object_z_now()

            result: dict[str, Any] = {
                "active": False,
                "done": True,
                "object": object_body,
                "end_effector": f"{ee_kind}:{ee_name}",
                "gripper": gripper_name,
                "aligned": aligned,
                "iterations": len(history),
                "target": [round(float(v), 3) for v in grasp],
                "object_z_start": round(object_z0, 4),
                "object_z_end": round(object_z1, 4),
                "lifted": round(object_z1 - object_z0, 4),
            }
            print(
                f"[vision] ee={ee_kind}:{ee_name} target={np.round(grasp, 3).tolist()} "
                f"aligned={aligned} iters={len(history)} lifted={result['lifted']}",
                flush=True,
            )
            self._sim.set_servo(None)
            self._report(**result)
        finally:
            renderer.close()

    def _finish(self, error: str) -> None:
        self._sim.set_servo(None)
        self._report(active=False, done=True, error=error)
