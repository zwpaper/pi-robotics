"""Live Python Studio workspace: scaffold, load, hot-reload and bake.

A *workspace* is a visible directory holding three editable Python files that
MuJoCo Studio loads as plugins:

* ``messages.py``      — shared ``Snapshot`` / ``Event`` classes.
* ``sim_plugin.py``    — a ``plugin.SimPlugin`` that builds the scene in Python
  with ``mujoco.MjSpec`` and rebuilds it when the UI asks for new parameters.
* ``viewer_plugin.py`` — a ``plugin.ViewerPlugin`` whose Dear ImGui panel drives
  the scene parameters from the browser (sliders, buttons, a Bake button).

Because the files are regular importable modules (``studio_plugins.<name>``),
MuJoCo Studio's own hot reload watches them: saving any of them swaps the
plugin class in place without restarting the daemon. The panel lives **inside
the 3D viewer already open in the browser** — no extra viewer, no extra port.

The daemon loads the workspace once at startup (plugins must be registered up
front). The agent-facing side is the loopback control API: ``/workspace`` for
status, ``scaffold_workspace`` to write the starter files, and ``bake`` to
freeze the current scene into ``<workspace>/scene.xml``.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys
import tempfile
import traceback
from typing import Any

import mujoco

# The scene name the daemon uses while a live workspace owns the scene.
WORKSPACE_SCENE = "workspace"

# The files a workspace is made of, and the plugin modules among them.
TEMPLATE_FILES = ("messages.py", "sim_plugin.py", "viewer_plugin.py")
PLUGIN_FILES = ("sim_plugin.py", "viewer_plugin.py")

DEFAULT_WORKSPACE_DIRNAME = "studio_workspace"


# ---------------------------------------------------------------------------
# Scaffold templates
# ---------------------------------------------------------------------------
MESSAGES_TEMPLATE = '''"""Shared scene messages for the pi-robotics Studio workspace.

`sim_plugin.py` and `viewer_plugin.py` both import this module, so editing and
saving it hot-reloads the running simulation.
"""

import dataclasses

from mujoco.studio import messages


@dataclasses.dataclass(frozen=True)
class SceneConfig(messages.Snapshot):
    """UI -> sim: scene parameters. Any change rebuilds the scene."""

    seed: int = 0
    drop_height: float = 0.6  # Height of the lowest toy above the floor.
    toy_scale: float = 1.0  # Overall toy size.


@dataclasses.dataclass(frozen=True)
class BakeRequest(messages.Event):
    """UI -> sim: write the current scene to scene.xml and show that file."""


@dataclasses.dataclass(frozen=True)
class Telemetry(messages.Snapshot):
    """sim -> UI: latest-wins summary of the current drop."""

    sim_time: float = 0.0
    # Sim time at which every toy came to rest, or None while still moving.
    settled_at: float | None = None
'''

SIM_PLUGIN_TEMPLATE = '''"""Simulation-side Studio plugin, built in Python (hot-reloaded on save).

The whole scene is an `MjSpec`; the panel in `viewer_plugin.py` changes its
parameters and the scene is rebuilt in place. Ask the pi-robotics agent to
`bake` the scene into `scene.xml` when you are happy with it.
"""

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

# Each toy's geom type, base size, and color.
TOYS = {
    'box': (mujoco.mjtGeom.mjGEOM_BOX, (0.06, 0.06, 0.06), (0.26, 0.52, 0.96)),
    'sphere': (mujoco.mjtGeom.mjGEOM_SPHERE, (0.06, 0, 0), (0.92, 0.26, 0.21)),
    'cylinder': (
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        (0.05, 0.06, 0),
        (0.98, 0.74, 0.02),
    ),
    'ellipsoid': (
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        (0.07, 0.05, 0.045),
        (0.20, 0.66, 0.33),
    ),
}
# One drop cell per toy so they never spawn inside each other.
CELLS = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
SPREAD = 0.4  # Half-width of the square drop area, in meters.


class SimPlugin(plugin.SimPlugin):
    """Builds the toy-drop scene and reports when the toys settle."""

    # Scene parameters, set from the panel (see messages.SceneConfig).
    seed: int = 0
    drop_height: float = 0.6
    toy_scale: float = 1.0
    # Edit while running: each toy's size varies by up to this fraction.
    size_jitter: float = 0.2
    # Edit while running: anything slower than this counts as settled.
    settle_speed: float = 0.05

    spec: mujoco.MjSpec | None = None
    drop_requested: bool = False
    settled_at: float | None = None

    def build_spec(self) -> mujoco.MjSpec:
        """Returns the scene for the current parameters."""
        rng = np.random.default_rng(self.seed)
        spec = mujoco.MjSpec()
        spec.modelname = 'studio_workspace'
        spec.option.timestep = 0.002
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
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

        # condim 6 adds torsional and rolling friction so round toys stop.
        spec.default.geom.friction = [0.6, 0.005, 0.005]
        spec.default.geom.condim = 6
        spec.default.geom.density = 400

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
        for i, (toy, (geom_type, size, rgb)) in enumerate(TOYS.items()):
            scale = self.toy_scale * rng.uniform(
                1 - self.size_jitter, 1 + self.size_jitter
            )
            xy = np.array(CELLS[i]) * SPREAD / 2
            xy += rng.uniform(-0.25, 0.25, size=2) * SPREAD
            quat = rng.normal(size=4)
            body = world.add_body(
                name=toy,
                pos=[*xy, self.drop_height + 0.15 * i],
                quat=quat / np.linalg.norm(quat),
            )
            body.add_freejoint(name=f'{toy}_free')
            body.add_geom(
                name=f'{toy}_geom',
                type=geom_type,
                size=np.array(size) * scale,
                rgba=[*rgb, 1],
            )
        return spec

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
        # A new drop starts the toys where build_spec() placed them; a rebuild
        # after an edit keeps the current poses.
        if self.drop_requested:
            self.drop_requested = False
            mujoco.mj_resetData(event.model, event.data)
            mujoco.mj_forward(event.model, event.data)
        self.settled_at = None

    @messages.handler
    def on_hot_reload(self, event: messages.HotReloadEvent) -> None:
        if self in event.plugins:
            self.handle.reload()

    @messages.handler
    def on_scene_config(self, snapshot: user_messages.SceneConfig) -> None:
        self.seed = snapshot.seed
        self.drop_height = snapshot.drop_height
        self.toy_scale = snapshot.toy_scale
        self.drop_requested = True
        self.handle.reload()

    @messages.handler
    def on_bake_request(self, event: user_messages.BakeRequest) -> None:
        """Writes the current scene to scene.xml and shows the file from now on."""
        del event
        if self.spec is None:
            return
        SCENE_XML.write_text(self.spec.to_xml(), encoding='utf-8')
        self.handle.dispatch(messages.LoadModelEvent(path=str(SCENE_XML)))

    @messages.handler
    def on_post_step(self, event: messages.PostStepEvent) -> None:
        data = event.data
        moving = bool(np.abs(data.qvel).max() >= self.settle_speed)
        if moving:
            self.settled_at = None
        elif self.settled_at is None:
            self.settled_at = float(data.time)
        self.handle.send_to_viewer(
            user_messages.Telemetry(
                sim_time=float(data.time), settled_at=self.settled_at
            )
        )
'''

VIEWER_PLUGIN_TEMPLATE = '''"""Viewer-side Dear ImGui Studio plugin (hot-reloaded on save).

The panel below is rendered **inside the browser 3D viewer**. Drag a slider and
the scene in sim_plugin.py is rebuilt live; the pi-robotics agent can also drive
the same simulation through its control API.
"""

import os
import pathlib

import messages as user_messages
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio.dear_imgui import dear_imgui as imgui

# This file lives in the workspace; shown so you always know what hot-reloads.
WORKSPACE_DIR = pathlib.Path(os.path.abspath(__file__)).parent
EDITABLE_FILES = ('messages.py', 'sim_plugin.py', 'viewer_plugin.py')


def _short_path(path: pathlib.Path) -> str:
    home = pathlib.Path.home()
    if path.is_relative_to(home):
        return '~/' + str(path.relative_to(home))
    return str(path)


class ViewerPlugin(plugin.ViewerPlugin):
    """Live parameter panel for the workspace scene."""

    seed: int = 0
    drop_height: float = 0.6
    toy_scale: float = 1.0
    telemetry: user_messages.Telemetry = user_messages.Telemetry()
    window_open: bool = True

    @messages.handler
    def on_telemetry(self, snapshot: user_messages.Telemetry) -> None:
        self.telemetry = snapshot

    def _send_config(self) -> None:
        self.viewer.send_to_sim(
            user_messages.SceneConfig(
                seed=self.seed,
                drop_height=self.drop_height,
                toy_scale=self.toy_scale,
            )
        )

    @messages.handler
    def on_build_gui(self, event: messages.BuildGuiEvent) -> None:
        del event
        if not self.window_open:
            return
        expanded, self.window_open = imgui.Begin(
            'pi-robotics workspace###piRoboticsWorkspace', self.window_open
        )
        try:
            if not expanded:
                return
            imgui.TextUnformatted(f'Workspace: {_short_path(WORKSPACE_DIR)}')
            imgui.SetItemTooltip(
                'Edit and save any of these to hot-reload the running sim:\\n'
                + '\\n'.join(f'  {name}' for name in EDITABLE_FILES)
            )
            imgui.Separator()

            changed = False
            seed_changed, seed = imgui.InputInt('Seed', self.seed)
            if seed_changed:
                self.seed = max(0, seed)
                changed = True
            height_changed, self.drop_height = imgui.SliderFloat(
                'Drop height (m)', self.drop_height, 0.2, 2.0
            )
            scale_changed, self.toy_scale = imgui.SliderFloat(
                'Toy size', self.toy_scale, 0.5, 2.0
            )
            if changed or height_changed or scale_changed:
                self._send_config()
            if imgui.Button('Drop again'):
                self._send_config()

            t = self.telemetry
            if t.settled_at is None:
                imgui.TextUnformatted(f'Moving...  t = {t.sim_time:.2f} s')
            else:
                imgui.TextUnformatted(f'Settled at {t.settled_at:.2f} s')

            imgui.Separator()
            if imgui.Button('Bake to scene.xml'):
                self.viewer.send_to_sim(user_messages.BakeRequest())
            imgui.SetItemTooltip(
                'Write the current scene to scene.xml in the workspace and show'
                ' that file from now on.'
            )
        finally:
            imgui.End()
'''

SCAFFOLD_TEMPLATES = {
    "messages.py": MESSAGES_TEMPLATE,
    "sim_plugin.py": SIM_PLUGIN_TEMPLATE,
    "viewer_plugin.py": VIEWER_PLUGIN_TEMPLATE,
}


# ---------------------------------------------------------------------------
# Scaffold
# ---------------------------------------------------------------------------
def scaffold_workspace(
    workspace_dir: pathlib.Path, scene_file: str = ""
) -> list[str]:
    """Writes the starter plugin files into ``workspace_dir``.

    Does nothing once the workspace already has any scene XML or plugin file, so
    user files are never overwritten.

    Returns:
        The names of the files written (empty when the workspace was not empty).
    """
    workspace_dir = pathlib.Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    has_existing = (
        (bool(scene_file) and (workspace_dir / scene_file).exists())
        or any(workspace_dir.glob("*.xml"))
        or any((workspace_dir / name).exists() for name in TEMPLATE_FILES)
    )
    if has_existing:
        return []
    for name in TEMPLATE_FILES:
        (workspace_dir / name).write_text(SCAFFOLD_TEMPLATES[name], encoding="utf-8")
    return list(TEMPLATE_FILES)


# ---------------------------------------------------------------------------
# Load + hot reload
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Workspace:
    """A loaded workspace: its directory, plugins, errors and reload config."""

    dir: pathlib.Path | None = None
    plugins: list[Any] = dataclasses.field(default_factory=list)
    errors: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.dir is not None and bool(self.plugins)

    @property
    def has_sim_plugin(self) -> bool:
        from mujoco.studio import plugin as studio_plugin

        return any(isinstance(p, studio_plugin.SimPlugin) for p in self.plugins)

    def files(self) -> dict[str, bool]:
        if self.dir is None:
            return {}
        return {
            name: (self.dir / name).is_file()
            for name in (*TEMPLATE_FILES, "scene.xml")
        }

    def hot_reload_spec(self) -> str:
        """The ``--hot_reload`` spec covering the model and workspace classes."""
        names = sorted({type(p).__name__ for p in self.plugins})
        return ",".join(["model", *names]) if names else "model"

    def scene_xml(self) -> pathlib.Path | None:
        return None if self.dir is None else self.dir / "scene.xml"

    def status(self) -> dict[str, Any]:
        return {
            "dir": str(self.dir) if self.dir is not None else "",
            "active": self.active,
            "files": self.files(),
            "plugins": [type(p).__name__ for p in self.plugins],
            "hot_reload": self.hot_reload_spec(),
            "scene_xml": str(self.scene_xml()) if self.dir is not None else "",
            "errors": dict(self.errors),
        }

    # -- bake ----------------------------------------------------------------
    def bake(self, model: mujoco.MjModel | None, out_path: pathlib.Path) -> dict[str, Any]:
        """Freezes the current scene into ``out_path`` and returns a summary.

        Uses a workspace plugin's live ``MjSpec`` when one is exposed (the
        scaffold does), otherwise re-serializes the compiled model.
        """
        spec = next(
            (p.spec for p in self.plugins if isinstance(getattr(p, "spec", None), mujoco.MjSpec)),
            None,
        )
        if spec is not None:
            xml = spec.to_xml()
            source = "spec"
        else:
            if model is None:
                raise RuntimeError("no model is loaded to bake")
            xml = model_to_xml(model)
            source = "model"
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(xml, encoding="utf-8")
        return {
            "ok": True,
            "path": str(out_path),
            "source": source,
            "bytes": len(xml),
        }


def load_workspace_plugins(workspace_dir: pathlib.Path) -> Workspace:
    """Loads ``sim_plugin.py`` and ``viewer_plugin.py`` from ``workspace_dir``.

    Missing files are skipped; a file that fails to import is recorded in
    ``errors`` so the daemon (and the agent over the control API) can report it
    without crashing the simulation.
    """
    from mujoco.studio import plugin as studio_plugin

    workspace = Workspace(dir=pathlib.Path(workspace_dir))
    if str(workspace.dir) not in sys.path:
        sys.path.insert(0, str(workspace.dir))
    for name in PLUGIN_FILES:
        path = workspace.dir / name
        if not path.is_file():
            continue
        try:
            workspace.plugins.extend(studio_plugin.load_plugins([path]))
        except Exception:  # noqa: BLE001 - surface, keep the daemon alive
            workspace.errors[name] = traceback.format_exc()
    return workspace


def model_to_xml(model: mujoco.MjModel) -> str:
    """Serializes a compiled ``MjModel`` to XML via ``mj_saveLastXML``."""
    fd, tmp = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    try:
        mujoco.mj_saveLastXML(tmp, model)
        return pathlib.Path(tmp).read_text(encoding="utf-8")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def default_workspace_dir(cwd: pathlib.Path) -> pathlib.Path:
    """The default visible workspace directory for a project."""
    return pathlib.Path(cwd) / DEFAULT_WORKSPACE_DIRNAME
