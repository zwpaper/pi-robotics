"""A hidden "director" body that the sim glides along the action front.

A scene may contain a body named ``director`` (a small, non-colliding,
transparent body) with a ``mode="fixed"`` camera parented to it. Whenever such
a body exists, :class:`Director` keeps it glued to the *leading edge* of the
motion -- the forward-most body or site that has displaced from its initial
pose -- so the camera becomes a native "follow the whole action" chase camera.

Because the director's pose is a joint (a freejoint) it lives in ``qpos``, so
the recorder captures it and the reel replays the same chase camera with no
per-frame scripting. It also works in the live viewer.

The rule is scene-agnostic: at each step, every body and site that has moved
more than ``threshold`` from its rest pose votes with its x-coordinate, and the
director's target is the maximum. The target is then rate-limited so the camera
glides instead of jumping.
"""

from __future__ import annotations

import numpy as np
import mujoco

#: Default name of the body the daemon animates as a camera director.
DIRECTOR_BODY = "director"


class Director:
    """Drives a ``director`` body to the leading edge of the current motion."""

    def __init__(
        self,
        model: mujoco.MjModel,
        body_name: str = DIRECTOR_BODY,
        lead: float = 0.20,
        threshold: float = 0.015,
        max_speed: float = 5.0,
        y: float = 0.0,
        z: float = 0.0,
    ) -> None:
        self.body_name = body_name
        self.lead = float(lead)
        self.threshold = float(threshold)
        self.max_speed = float(max_speed)
        self.y = float(y)
        self.z = float(z)
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.active = self.body >= 0
        self.front: float | None = None
        self.qadr = self.vadr = -1
        self._free = False
        if not self.active:
            return

        # Rest pose of every body and site (used to detect what has moved).
        scratch = mujoco.MjData(model)
        mujoco.mj_forward(model, scratch)
        self.p0 = np.array(scratch.xpos, dtype=float)
        self.s0 = np.array(scratch.site_xpos, dtype=float)

        jid = int(model.body_jntadr[self.body])
        self._free = jid >= 0 and int(model.jnt_type[jid]) == int(
            mujoco.mjtJoint.mjJNT_FREE
        )
        if jid >= 0:
            self.qadr = int(model.jnt_qposadr[jid])
            self.vadr = int(model.jnt_dofadr[jid])
        self.dt = float(model.opt.timestep)
        self.front = float(scratch.xpos[self.body][0]) - self.lead

    def _target(self, data: mujoco.MjData) -> float | None:
        votes: list[float] = []
        moved = np.linalg.norm(data.xpos - self.p0, axis=1) > self.threshold
        moved[0] = False  # the world never moves
        moved[self.body] = False  # the director is not part of the action
        if bool(moved.any()):
            votes.extend(float(x) for x in data.xpos[moved, 0])
        if self.s0.size:
            smoved = np.linalg.norm(data.site_xpos - self.s0, axis=1) > self.threshold
            if bool(smoved.any()):
                votes.extend(float(x) for x in data.site_xpos[smoved, 0])
        return max(votes) if votes else None

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """Glide the director toward the action front. Returns True if moved."""
        if not self.active or self.qadr < 0:
            return False
        target = self._target(data)
        if target is None:
            return False
        target += self.lead
        if self.front is None:
            self.front = target
        else:
            limit = self.max_speed * self.dt
            delta = float(np.clip(target - self.front, -limit, limit))
            self.front += delta

        if self._free:
            data.qpos[self.qadr : self.qadr + 3] = [self.front, self.y, self.z]
            data.qpos[self.qadr + 3 : self.qadr + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[self.vadr : self.vadr + 6] = 0.0
        else:
            data.qpos[self.qadr] = self.front
            data.qvel[self.vadr] = 0.0
        return True
