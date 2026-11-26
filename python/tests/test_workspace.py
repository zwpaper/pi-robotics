"""Live Python Studio workspace: scaffold, load, hot-reload spec and bake."""

from __future__ import annotations

import sys

import mujoco
import pytest
from mujoco.studio import plugin as studio_plugin

from pi_robotics import workspace


@pytest.fixture(autouse=True)
def _fresh_plugin_package():
    """Drop Studio's synthetic plugin package between tests.

    ``plugin.load_plugins`` imports ``dir/sim_plugin.py`` as
    ``studio_plugins.sim_plugin`` and refuses to load a second file with the
    same name; clearing the package lets each test scaffold into its own dir.
    """
    yield
    for name in [n for n in sys.modules if n == "studio_plugins" or n.startswith("studio_plugins.")]:
        del sys.modules[name]
    studio_plugin.PLUGIN_PACKAGE  # noqa: B018 - ensure module imported


def test_scaffold_writes_the_three_files(tmp_path):
    written = workspace.scaffold_workspace(tmp_path)
    assert written == list(workspace.TEMPLATE_FILES)
    for name in workspace.TEMPLATE_FILES:
        assert (tmp_path / name).is_file()


def test_scaffold_is_idempotent_and_never_overwrites(tmp_path):
    workspace.scaffold_workspace(tmp_path)
    (tmp_path / "sim_plugin.py").write_text("# user edit\n", encoding="utf-8")
    assert workspace.scaffold_workspace(tmp_path) == []
    assert (tmp_path / "sim_plugin.py").read_text(encoding="utf-8") == "# user edit\n"


def test_scaffold_skips_a_dir_with_an_xml_scene(tmp_path):
    (tmp_path / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
    assert workspace.scaffold_workspace(tmp_path) == []
    assert not (tmp_path / "sim_plugin.py").exists()


def test_templates_are_valid_python():
    for name, text in workspace.SCAFFOLD_TEMPLATES.items():
        compile(text, name, "exec")


def test_load_workspace_plugins_builds_the_scene(tmp_path):
    workspace.scaffold_workspace(tmp_path)
    loaded = workspace.load_workspace_plugins(tmp_path)
    assert loaded.errors == {}
    assert loaded.active
    assert loaded.has_sim_plugin
    assert sorted(type(p).__name__ for p in loaded.plugins) == ["SimPlugin", "ViewerPlugin"]
    assert loaded.hot_reload_spec() == "model,SimPlugin,ViewerPlugin"

    sim = next(p for p in loaded.plugins if isinstance(p, studio_plugin.SimPlugin))
    model = sim.build_spec().compile()
    # floor + four toys.
    assert model.nbody == 5
    assert model.nq == 28  # four free joints


def test_missing_workspace_is_inactive(tmp_path):
    loaded = workspace.load_workspace_plugins(tmp_path)
    assert not loaded.active
    assert not loaded.has_sim_plugin
    assert not any(loaded.status()["files"].values())


def test_bake_uses_the_plugin_spec(tmp_path):
    workspace.scaffold_workspace(tmp_path)
    loaded = workspace.load_workspace_plugins(tmp_path)
    sim = next(p for p in loaded.plugins if isinstance(p, studio_plugin.SimPlugin))
    sim.spec = sim.build_spec()

    out = tmp_path / "scene.xml"
    result = loaded.bake(None, out)
    assert result["ok"] and result["source"] == "spec"
    assert mujoco.MjModel.from_xml_path(str(out)).nbody == 5


def test_bake_falls_back_to_the_compiled_model(tmp_path):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
    )
    loaded = workspace.Workspace(dir=tmp_path)
    out = tmp_path / "scene.xml"
    result = loaded.bake(model, out)
    assert result["source"] == "model"
    assert mujoco.MjModel.from_xml_path(str(out)).nbody == 1


def test_bake_without_a_model_or_spec_raises(tmp_path):
    loaded = workspace.Workspace(dir=tmp_path)
    with pytest.raises(RuntimeError):
        loaded.bake(None, tmp_path / "scene.xml")


def test_default_workspace_dir(tmp_path):
    assert workspace.default_workspace_dir(tmp_path).name == workspace.DEFAULT_WORKSPACE_DIRNAME
