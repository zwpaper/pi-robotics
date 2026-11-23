"""Vision-based closed-loop grasping for pi-robotics.

Runs on its own thread using the robot's own camera:

  1. **Camera** — renders a model camera offscreen with MuJoCo's segmentation
     renderer, from the published state snapshot (never the live ``MjData``).
  2. **Detect** — finds the target object's image centroid from the segmentation
     mask (per-geom, so it is robust to colour and lighting). The pixel is
     back-projected through the camera model onto the tabletop plane to get the
     object's world position: *look-then-move*, position-based visual servoing.
  3. **Control** — measures the gripper fingertip position by forward kinematics
     (the robot has joint encoders) and drives ``arm_extend`` / ``lift`` to move
     the fingertip onto the camera-localized object. When aligned it closes the
     gripper and lifts.

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

# Fingertip geoms whose centroid is the visual grasp point.
_FINGERTIP_BODIES = (
    "link_gripper_finger_left",
    "link_gripper_finger_right",
    "rubber_tip_left",
    "rubber_tip_right",
)

# Head pose so the camera looks at the tabletop workspace.
HEAD_LOOK_AT = {"head_pan": -1.5, "head_tilt": -0.9}

# Tabletop object centre height (metres) for the back-projection plane.
DEFAULT_PLANE_Z = 0.52


class VisionServo:
    """Perceive -> detect -> control loop that grasps the target object."""

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
    ) -> dict[str, Any]:
        if self.busy():
            return {"ok": False, "error": "a vision servo is already running"}
        model = self._handle.model
        if model is None:
            return {"ok": False, "error": "no model loaded"}
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body) < 0:
            return {"ok": False, "error": f"unknown object body '{object_body}'"}
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            return {"ok": False, "error": f"unknown camera '{camera}'"}
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(object_body, cam_id, plane_z, width, height),
            name="pi-robotics-vision",
            daemon=True,
        )
        self._thread.start()
        return {"ok": True, "servo": object_body, "camera": camera}

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
        self._sim.set_servo(dict(ctrl, **HEAD_LOOK_AT))

    def _report(self, **info: Any) -> None:
        self._state.set_servo(info)

    # -- main loop -----------------------------------------------------------
    def _run(
        self, object_body: str, cam_id: int, plane_z: float, width: int, height: int
    ) -> None:
        try:
            self._run_inner(object_body, cam_id, plane_z, width, height)
        except Exception as exc:  # noqa: BLE001
            self._sim.set_servo(None)
            self._report(active=False, done=True, error=str(exc))

    def _run_inner(
        self, object_body: str, cam_id: int, plane_z: float, width: int, height: int
    ) -> None:
        model = self._handle.model
        assert model is not None
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)
        }
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        object_geoms = self._geoms_of(model, object_id)
        fingertip_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in _FINGERTIP_BODIES
        ]
        fingertip_ids = [b for b in fingertip_ids if b >= 0]
        if not object_geoms or not fingertip_ids:
            self._sim.set_servo(None)
            self._report(active=False, done=True, error="no object/fingertip bodies")
            return

        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.enable_segmentation_rendering()
        scratch = mujoco.MjData(model)

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
            direction = mat @ np.array([(u - cx) / focal, -(v - cy) / focal, -1.0])
            if abs(direction[2]) < 1e-6:
                return None
            t = (plane_z - pos[2]) / direction[2]
            return pos + t * direction

        def gripper_world() -> np.ndarray:
            sync()
            return np.mean([scratch.xpos[b] for b in fingertip_ids], axis=0)

        def object_z_now() -> float:
            sync()
            return float(scratch.xpos[object_id][2])

        def apply(ctrl: dict[str, float], settle: float) -> None:
            self._set_ctrl(ctrl)
            self._sleep(settle)

        base = {"lift": 0.0, "arm_extend": 0.0, "wrist_yaw": 0.15, "grip": 0.04}
        ctrl = dict(base)

        # 1) Camera: settle, then localize the object on the tabletop plane.
        self._report(active=True, phase="settle")
        apply(ctrl, 1.2)
        pixel = object_pixel()
        if pixel is None:
            self._sim.set_servo(None)
            self._report(active=False, done=True, error="object not visible")
            renderer.close()
            return
        target = back_project(pixel[0], pixel[1])
        if target is None:
            self._sim.set_servo(None)
            self._report(active=False, done=True, error="back-projection failed")
            renderer.close()
            return
        self._report(
            active=True,
            phase="localized",
            object_pixel=[round(float(pixel[0]), 1), round(float(pixel[1]), 1)],
            object_world=[round(float(v), 3) for v in target],
        )

        # 2) Control: drive the fingertip onto the object (arm_extend -> y,
        #    lift -> z; the Jacobian is ~identity for this rig).
        aligned = False
        history: list[dict[str, Any]] = []
        for it in range(8):
            apply(ctrl, 0.5)
            grip = gripper_world()
            ey = float(target[1] - grip[1])
            ez = float(target[2] - grip[2])
            history.append({"iteration": it, "ey": round(ey, 4), "ez": round(ez, 4)})
            self._report(
                active=True,
                phase="servo",
                iteration=it,
                target=[round(float(v), 3) for v in target],
                gripper=[round(float(v), 3) for v in grip],
                error_m=[round(ey, 4), round(ez, 4)],
                arm_extend=round(ctrl["arm_extend"], 3),
                lift=round(ctrl["lift"], 3),
            )
            if abs(ey) < 0.02 and abs(ez) < 0.02:
                aligned = True
                break
            ctrl["arm_extend"] = float(
                np.clip(ctrl["arm_extend"] + (grip[1] - float(target[1])), 0.0, 0.45)
            )
            ctrl["lift"] = float(
                np.clip(ctrl["lift"] + (float(target[2]) - grip[2]), -0.15, 0.40)
            )

        # 3) Close the gripper, then lift.
        self._report(active=True, phase="close", iterations=len(history))
        ctrl["grip"] = -0.005
        apply(ctrl, 1.2)
        object_z0 = object_z_now()
        self._report(active=True, phase="lift", iterations=len(history))
        ctrl["lift"] = min(0.6, ctrl["lift"] + 0.35)
        apply(ctrl, 2.0)
        object_z1 = object_z_now()

        renderer.close()
        self._sim.set_servo(None)
        result: dict[str, Any] = {
            "active": False,
            "done": True,
            "object": object_body,
            "aligned": aligned,
            "iterations": len(history),
            "target": [round(float(v), 3) for v in target],
            "object_z_start": round(object_z0, 4),
            "object_z_end": round(object_z1, 4),
            "lifted": round(object_z1 - object_z0, 4),
        }
        print(
            f"[vision] target={np.round(target, 3).tolist()} "
            f"aligned={aligned} iters={len(history)} lifted={result['lifted']}",
            flush=True,
        )
        self._report(**result)
