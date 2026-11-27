"""Simulation-side Studio plugin: a 榫卯 (Chinese joinery) physics sandbox.

Two classic joints, all real rigid bodies with gravity, contact and friction:

* **直榫 (straight tenon)** — a shoulder block with a square tenon drops into a
  socket cut into a fixed beam; the shoulder rests on the beam top.
* **十字搭接 (cross-lap)** — two beams each notched to half depth interlock
  flush when the upper one is lowered into the lower one.

Dragging is a real grasp, not a spring. A kinematic **mocap hand** follows the
mouse rigidly; when it comes close to a part the sim welds that part to the
hand (an `mjEQ_WELD` whose target is re-computed at grab time), so the part is
held rigidly but keeps its own mass — it collides, jams and drops for real.
Release with the panel's "Open hand" button, or by letting the agent drive the
parts' `qpos` through its control API.
"""

import math
import os
import pathlib

import messages as user_messages
import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin
import numpy as np

# This file lives in the workspace, so its folder is where scene.xml is baked.
WORKSPACE_DIR = pathlib.Path(os.path.abspath(__file__)).parent
SCENE_XML = WORKSPACE_DIR / 'scene.xml'

# -- layout (metres) ---------------------------------------------------------
BASE_X = -0.34  # centre of the straight-tenon socket block
LAP_X = 0.36  # centre of the cross-lap joint
TENON_DROP_Z = 0.22  # spawn height of the tenon block (its tenon tip)
LAP_DROP_Z = 0.12  # spawn height of the upper lap beam
HAND_START = [0.06, -0.30, 0.34]  # where the mocap hand starts
GRIP_RADIUS = 0.08  # the hand closes on a part within this distance
PARTS = ('tenon', 'lap_a', 'lap_b')

# (name, position, look-at target, world up) — cameras make `reel` multi-angle.
CAMERAS = (
    ('cam_iso', [-0.02, -0.95, 0.74], [0.0, 0.0, 0.10], [0.0, 0.0, 1.0]),
    ('cam_tenon', [-0.34, -0.52, 0.30], [-0.34, 0.0, 0.11], [0.0, 0.0, 1.0]),
    ('cam_lap', [0.36, -0.52, 0.28], [0.36, 0.0, 0.04], [0.0, 0.0, 1.0]),
    ('cam_top', [0.0, 0.0, 1.05], [0.0, 0.0, 0.05], [0.0, -1.0, 0.0]),
)

# -- palette -----------------------------------------------------------------
WOOD = [0.76, 0.55, 0.30, 1.0]
WOOD_DARK = [0.55, 0.37, 0.20, 1.0]
WOOD_LIGHT = [0.87, 0.70, 0.46, 1.0]
STEEL = [0.30, 0.50, 0.72, 1.0]


# -- quaternion helpers (w, x, y, z) -----------------------------------------
def _qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _qrot(q, v):
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def _camera_quat(pos, target, up):
    """Quaternion orienting a camera at ``pos`` to look at ``target``.

    MuJoCo cameras look along their local -Z with +X to the right.
    """
    pos, target, up = map(lambda v: np.asarray(v, dtype=float), (pos, target, up))
    forward = target - pos
    forward /= np.linalg.norm(forward)
    z_cam = -forward
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_cam = np.cross(z_cam, right)
    rot = np.column_stack([right, up_cam, z_cam])
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        quat = [0.25 * s, (rot[2, 1] - rot[1, 2]) / s,
                (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s]
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        quat = [(rot[2, 1] - rot[1, 2]) / s, 0.25 * s,
                (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s]
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        quat = [(rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s,
                0.25 * s, (rot[1, 2] + rot[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        quat = [(rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s,
                (rot[1, 2] + rot[2, 1]) / s, 0.25 * s]
    quat = np.asarray(quat)
    return (quat / np.linalg.norm(quat)).tolist()


def _box(body, name, center, half, rgba, **kwargs):
    """Adds an axis-aligned box geom (``half`` = half-extents)."""
    return body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(half),
        pos=list(center),
        rgba=list(rgba),
        **kwargs,
    )


class SimPlugin(plugin.SimPlugin):
    """Builds the joinery scene, the grasping hand, and the assembly telemetry."""

    # Live physics parameters, set from the panel (see messages.SceneConfig).
    gravity: float = 9.81
    friction: float = 0.6
    # Edit while running: anything slower than this counts as settled.
    settle_speed: float = 0.05

    spec: mujoco.MjSpec | None = None
    drop_requested: bool = False
    settled_at: float | None = None
    gripping: str = ''
    latched_part: str = ''  # just released: don't re-grab it until the hand leaves

    model: mujoco.MjModel | None = None
    data: mujoco.MjData | None = None
    ids: dict[str, int] = {}
    site_ids: dict[str, int] = {}
    eq_ids: dict[str, int] = {}

    # -- scene ---------------------------------------------------------------
    def build_spec(self) -> mujoco.MjSpec:
        """Returns the joinery scene for the current parameters."""
        spec = mujoco.MjSpec()
        spec.modelname = 'mortise_tenon'
        spec.option.timestep = 0.002
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.gravity = [0.0, 0.0, -self.gravity]
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.visual.headlight.specular = [0, 0, 0]
        spec.visual.global_.azimuth = 135
        spec.visual.global_.elevation = -20
        spec.add_texture(
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0, 0, 0],
            width=512,
            height=3072,
        )
        spec.add_texture(
            name='groundplane',
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            mark=mujoco.mjtMark.mjMARK_EDGE,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            markrgb=[0.8, 0.8, 0.8],
            width=300,
            height=300,
        )
        ground = spec.add_material(
            name='groundplane', texrepeat=[5, 5], texuniform=True, reflectance=0.2
        )
        ground.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'groundplane'

        # Wood: heavy, grippy, with torsional friction (condim 4) so the beams
        # settle instead of creeping. Short contact time constant = stiff wood.
        spec.default.geom.friction = [self.friction, 0.01, 0.001]
        spec.default.geom.condim = 4
        spec.default.geom.density = 550
        spec.default.geom.solref = [0.01, 1.0]

        world = spec.worldbody
        world.add_light(
            pos=[0, 0, 2.5],
            dir=[0, 0, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
        world.add_geom(
            name='floor',
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            material='groundplane',
        )
        self._add_base(world)
        self._add_tenon(world)
        self._add_lap(world)
        self._add_hand(spec, world)
        for name, pos, target, up in CAMERAS:
            world.add_camera(
                name=name, pos=list(pos), quat=_camera_quat(pos, target, up)
            )
        return spec

    def _add_base(self, world) -> None:
        """Fixed beam with a square socket cut into its top face."""
        base = world.add_body(name='base', pos=[BASE_X, 0, 0])
        _box(base, 'base_slab', (0, 0, 0.025), (0.17, 0.11, 0.025), WOOD_DARK)
        _box(base, 'base_left', (-0.1025, 0, 0.085), (0.0675, 0.11, 0.035), WOOD)
        _box(base, 'base_right', (0.1025, 0, 0.085), (0.0675, 0.11, 0.035), WOOD)
        _box(base, 'base_front', (0, -0.0725, 0.085), (0.035, 0.0375, 0.035), WOOD)
        _box(base, 'base_back', (0, 0.0725, 0.085), (0.035, 0.0375, 0.035), WOOD)

    def _add_tenon(self, world) -> None:
        """Free shoulder block whose square tenon fits the socket (直榫)."""
        body = world.add_body(name='tenon', pos=[BASE_X, 0, TENON_DROP_Z])
        body.add_freejoint(name='tenon_free')
        # Origin sits at the tenon tip; the shoulder rests on the beam top.
        _box(body, 'tenon_tip', (0, 0, 0.035), (0.031, 0.031, 0.035), WOOD_LIGHT)
        _box(body, 'tenon_shoulder', (0, 0, 0.12), (0.06, 0.06, 0.05), WOOD)
        body.add_site(name='tenon_grip', pos=[0, 0, 0.12])

    def _add_lap(self, world) -> None:
        """Two half-lapped beams that interlock flush (十字搭接).

        The notch is a hair wider than the crossing beam (0.076 vs 0.070 m) so
        the upper beam drops in instead of catching an edge and tipping.
        """
        lower = world.add_body(name='lap_a', pos=[LAP_X, 0, 0])
        lower.add_freejoint(name='lap_a_free')
        _box(lower, 'lap_a_left', (-0.084, 0, 0.035), (0.046, 0.035, 0.035), WOOD)
        _box(lower, 'lap_a_right', (0.084, 0, 0.035), (0.046, 0.035, 0.035), WOOD)
        _box(lower, 'lap_a_center', (0, 0, 0.0175), (0.038, 0.035, 0.0175), WOOD_DARK)
        lower.add_site(name='lap_a_grip', pos=[0, 0, 0.035])

        upper = world.add_body(name='lap_b', pos=[LAP_X, 0, LAP_DROP_Z])
        upper.add_freejoint(name='lap_b_free')
        _box(upper, 'lap_b_front', (0, -0.084, 0.035), (0.035, 0.046, 0.035), WOOD_LIGHT)
        _box(upper, 'lap_b_back', (0, 0.084, 0.035), (0.035, 0.046, 0.035), WOOD_LIGHT)
        _box(upper, 'lap_b_center', (0, 0, 0.0525), (0.035, 0.038, 0.0175), WOOD_DARK)
        upper.add_site(name='lap_b_grip', pos=[0, 0, 0.035])

    def _add_hand(self, spec, world) -> None:
        """Kinematic hand driven by the mouse, plus one inactive grip weld each.

        The hand's geometry is render-only (no contacts); holding is done by the
        weld, so the part stays a real dynamic body that collides and jams.
        """
        hand = world.add_body(name='hand', mocap=True, pos=list(HAND_START))
        hand.add_site(name='hand_grip', pos=[0, 0, 0], size=[0.012, 0, 0],
                      rgba=[1.0, 0.85, 0.2, 0.6])
        _box(hand, 'hand_palm', (0, 0, 0.11), (0.02, 0.075, 0.022), STEEL,
             contype=0, conaffinity=0)
        _box(hand, 'hand_finger_a', (0, -0.058, 0.035), (0.015, 0.018, 0.055), STEEL,
             contype=0, conaffinity=0)
        _box(hand, 'hand_finger_b', (0, 0.058, 0.035), (0.015, 0.018, 0.055), STEEL,
             contype=0, conaffinity=0)
        # A stiff weld gives a near-rigid grasp (~few mm sag under load).
        for part in PARTS:
            spec.add_equality(
                name=f'grip_{part}',
                type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1=part,
                name2='hand',
                active=0,
                solref=[0.005, 1.0],
            )

    # -- lifecycle -----------------------------------------------------------
    @messages.handler
    def on_load_model(self, event: messages.LoadModelEvent) -> bool:
        if event.path:
            # A model file (scene.xml after a bake, or a dropped file) is shown
            # instead; declining lets the daemon load that file.
            return False
        self.spec = self.build_spec()
        self.handle.set_model(self.spec)
        return True

    @messages.handler
    def on_post_model(self, event: messages.PostModelEvent) -> None:
        self.model, self.data = event.model, event.data
        self.ids = {
            name: mujoco.mj_name2id(event.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (*PARTS, 'hand')
        }
        self.site_ids = {
            name: mujoco.mj_name2id(event.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ('hand_grip', *(f'{p}_grip' for p in PARTS))
        }
        self.eq_ids = {
            name: mujoco.mj_name2id(
                event.model, mujoco.mjtObj.mjOBJ_EQUALITY, f'grip_{name}'
            )
            for name in PARTS
        }
        self.gripping, self.latched_part = '', ''
        event.data.eq_active[:] = 0
        if self.drop_requested:
            self.drop_requested = False
            mujoco.mj_resetData(event.model, event.data)
            mujoco.mj_forward(event.model, event.data)
        self.settled_at = None

    @messages.handler
    def on_hot_reload(self, event: messages.HotReloadEvent) -> None:
        if self in event.plugins:
            self.handle.reload()

    # -- grasping ------------------------------------------------------------
    def _retarget_weld(self, part: str) -> None:
        """Points the weld at the *current* relative pose of part and hand.

        MuJoCo freezes a body weld's target at compile time; re-writing
        ``eq_data[3:10]`` (and zeroing the recompute flag ``eq_data[11]``) makes
        activating it a no-op, so the grasp never snaps the part.
        """
        idx = self.eq_ids[part]
        pid, hid = self.ids[part], self.ids['hand']
        ppos, pquat = self.data.xpos[pid], self.data.xquat[pid]
        hpos, hquat = self.data.xpos[hid], self.data.xquat[hid]
        datum = self.model.eq_data[idx]
        datum[0:3] = 0.0  # weld point = hand origin
        datum[3:6] = _qrot(_qinv(pquat), hpos - ppos)  # in part's frame
        datum[6:10] = _qmul(_qinv(pquat), hquat)
        datum[11] = 0.0

    def _grip(self, part: str) -> None:
        for name in PARTS:
            self.data.eq_active[self.eq_ids[name]] = 0
        self._retarget_weld(part)
        self.data.eq_active[self.eq_ids[part]] = 1
        self.gripping, self.latched_part = part, ''

    def _release(self) -> None:
        for name in PARTS:
            self.data.eq_active[self.eq_ids[name]] = 0
        self.gripping = ''

    def _hand_gap(self) -> dict[str, float]:
        hand = self.data.site_xpos[self.site_ids['hand_grip']]
        return {
            p: float(np.linalg.norm(hand - self.data.site_xpos[self.site_ids[f'{p}_grip']]))
            for p in PARTS
        }

    # -- panel commands ------------------------------------------------------
    @messages.handler
    def on_scene_config(self, snapshot: user_messages.SceneConfig) -> None:
        # Applied in place so you can tune while the parts are sitting there.
        self.gravity = snapshot.gravity
        self.friction = snapshot.friction
        if self.model is not None and self.data is not None:
            self.model.opt.gravity[:] = [0.0, 0.0, -self.gravity]
            self.model.geom_friction[:, 0] = self.friction
            mujoco.mj_forward(self.model, self.data)

    @messages.handler
    def on_reset_request(self, event: user_messages.ResetRequest) -> None:
        del event
        self.drop_requested = True
        self.handle.reload()

    @messages.handler
    def on_drop_request(self, event: user_messages.DropRequest) -> None:
        """Lifts one part straight up and lets gravity re-seat it."""
        if self.model is None or self.data is None:
            return
        self._release()
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, event.part)
        if body < 0:
            return
        jid = self.model.body_jntadr[body]
        if jid < 0:
            return
        adr = self.model.jnt_qposadr[jid]
        dof = self.model.jnt_dofadr[jid]
        self.data.qpos[adr : adr + 7] = self.model.qpos0[adr : adr + 7]
        self.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @messages.handler
    def on_grab_request(self, event: user_messages.GrabRequest) -> None:
        """Snaps the hand onto the nearest part and closes on it.

        This is the reliable grab: it does not depend on picking the hand with
        the mouse, so a click on the panel always works.
        """
        del event
        if self.model is None or self.data is None or self.gripping:
            return
        gaps = self._hand_gap()
        nearest = min(gaps, key=gaps.get)
        mocap = self.model.body_mocapid[self.ids['hand']]
        self.data.mocap_pos[mocap] = self.data.site_xpos[
            self.site_ids[f'{nearest}_grip']
        ].copy()
        mujoco.mj_forward(self.model, self.data)
        self.latched_part = ''
        self._grip(nearest)

    @messages.handler
    def on_hand_move(self, event: user_messages.HandMove) -> None:
        """Panel-driven hand translation (left-click only, no right-drag)."""
        if self.model is None or self.data is None:
            return
        self.data.mocap_pos[self.model.body_mocapid[self.ids['hand']]] = [
            event.x, event.y, event.z,
        ]
        mujoco.mj_forward(self.model, self.data)

    @messages.handler
    def on_release_request(self, event: user_messages.ReleaseRequest) -> None:
        del event
        if self.model is not None:
            self.latched_part = self.gripping  # don't re-grab until the hand leaves
            self._release()

    @messages.handler
    def on_bake_request(self, event: user_messages.BakeRequest) -> None:
        """Writes the current scene to scene.xml and shows the file from now on."""
        del event
        if self.spec is None:
            return
        SCENE_XML.write_text(self.spec.to_xml(), encoding='utf-8')
        self.handle.dispatch(messages.LoadModelEvent(path=str(SCENE_XML)))

    # -- telemetry + auto-grab ----------------------------------------------
    @messages.handler
    def on_post_step(self, event: messages.PostStepEvent) -> None:
        data = event.data
        moving = bool(np.abs(data.qvel).max() >= self.settle_speed)
        if moving:
            self.settled_at = None
        elif self.settled_at is None:
            self.settled_at = float(data.time)

        # The hand closes on whatever wood it comes near, like a reflex grasp.
        if self.model is not None:
            gaps = self._hand_gap()
            nearest = min(gaps, key=gaps.get)
            if self.gripping:
                pass
            elif nearest == self.latched_part:
                # Let go of this one: only re-grab once the hand has moved away.
                if gaps[nearest] > 2.0 * GRIP_RADIUS:
                    self.latched_part = ''
            elif gaps[nearest] < GRIP_RADIUS:
                self._grip(nearest)

        gap, seated, lapped = 999.0, False, False
        if self.ids.get('tenon', -1) >= 0:
            tp = data.xpos[self.ids['tenon']]
            gap = float(np.hypot(tp[0] - BASE_X, tp[1]))
            seated = gap < 0.015 and 0.02 < tp[2] < 0.09
        if self.ids.get('lap_a', -1) >= 0 and self.ids.get('lap_b', -1) >= 0:
            bp = data.xpos[self.ids['lap_b']]
            lapped = float(np.hypot(bp[0] - LAP_X, bp[1])) < 0.02 and bp[2] < 0.05

        hand = tuple(
            float(v)
            for v in data.mocap_pos[self.model.body_mocapid[self.ids['hand']]]
        )
        self.handle.send_to_viewer(
            user_messages.Telemetry(
                sim_time=float(data.time),
                straight_gap=gap,
                straight_seated=seated,
                lap_seated=lapped,
                gripping=self.gripping,
                hand_pos=hand,
            )
        )
