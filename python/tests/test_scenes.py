"""Built-in scene builders: every scene must compile, and params must apply."""

import mujoco
import pytest

from pi_robotics import scenes


def test_list_scenes_is_sorted_and_nonempty():
    names = scenes.list_scenes()
    assert names == sorted(names)
    assert "arm3" in names
    assert "box_drop" in names


@pytest.mark.parametrize("name", scenes.list_scenes())
def test_every_scene_compiles(name):
    model = mujoco.MjModel.from_xml_string(scenes.build_scene(name))
    # Every MJCF model needs at least a worldbody.
    assert model.nbody >= 1


def test_unknown_scene_raises_keyerror():
    with pytest.raises(KeyError):
        scenes.build_scene("does_not_exist")


def test_scene_params_are_applied():
    model = mujoco.MjModel.from_xml_string(
        scenes.build_scene("box_drop", {"height": 2.0, "size": 0.25})
    )
    box = model.body("box")
    assert box.pos[2] == pytest.approx(2.0)
    assert model.geom("box_geom").size[0] == pytest.approx(0.25)
