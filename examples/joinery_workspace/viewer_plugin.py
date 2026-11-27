"""Viewer-side Dear ImGui Studio plugin (hot-reloaded on save).

The panel below is rendered **inside the browser 3D viewer**.  Drag a slider
and the physics updates live; press a button to re-drop a joint.  The
pi-robotics agent drives the same parts through its control API.
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
    """Live parameter panel for the joinery scene."""

    gravity: float = 9.81
    friction: float = 0.6
    hand_pos: tuple = (0.0, 0.0, 0.55)
    telemetry: user_messages.Telemetry = user_messages.Telemetry()
    window_open: bool = True

    @messages.handler
    def on_telemetry(self, snapshot: user_messages.Telemetry) -> None:
        self.telemetry = snapshot

    def _send_config(self) -> None:
        self.viewer.send_to_sim(
            user_messages.SceneConfig(gravity=self.gravity, friction=self.friction)
        )

    @messages.handler
    def on_build_gui(self, event: messages.BuildGuiEvent) -> None:
        del event
        if not self.window_open:
            return
        expanded, self.window_open = imgui.Begin(
            'pi-robotics joinery###piRoboticsJoinery', self.window_open
        )
        try:
            if not expanded:
                return
            imgui.TextUnformatted(f'Workspace: {_short_path(WORKSPACE_DIR)}')
            imgui.SetItemTooltip(
                'Drag parts with the mouse; let go and they fall for real.\n'
                'Edit and save any of these to hot-reload the sim:\n'
                + '\n'.join(f'  {name}' for name in EDITABLE_FILES)
            )
            imgui.Separator()

            gravity_changed, self.gravity = imgui.SliderFloat(
                'Gravity (m/s^2)', self.gravity, 0.0, 20.0
            )
            imgui.SetItemTooltip('Downward gravity. Applied live to the running sim.')
            friction_changed, self.friction = imgui.SliderFloat(
                'Wood friction', self.friction, 0.05, 1.5
            )
            imgui.SetItemTooltip(
                'Sliding friction of every wood geom (live).\n'
                'Higher = the joints grip and hold; lower = they slide apart.'
            )
            if gravity_changed or friction_changed:
                self._send_config()

            imgui.TextUnformatted('The BLUE hand is the only rigid grip.')
            imgui.TextUnformatted(
                '  in 3D: Ctrl/\u2318 + LEFT-drag ROTATES the hand;\n'
                '  Ctrl/\u2318 + RIGHT-drag TRANSLATES (add Shift: horizontal).'
            )
            imgui.SetItemTooltip(
                'MuJoCo drags dynamic bodies (the wood) with a spring force,\n'
                'which feels weak and off-axis. The hand is mocap, so it is\n'
                'dragged rigidly. Left-drag only rotates it, hence the sliders\n'
                'below. No move? Check the bottom-right Role badge says\n'
                'CONTROLLING (else click Request control).'
            )
            t = self.telemetry
            imgui.TextUnformatted('Move hand (left-click, in metres):')
            if not imgui.IsAnyItemActive():
                self.hand_pos = tuple(t.hand_pos)  # follow mouse drags / buttons
            hand = list(self.hand_pos)
            mx, hand[0] = imgui.SliderFloat('Hand X', hand[0], -0.9, 0.9)
            my, hand[1] = imgui.SliderFloat('Hand Y', hand[1], -0.9, 0.9)
            mz, hand[2] = imgui.SliderFloat('Hand Z', hand[2], 0.0, 0.95)
            if mx or my or mz:
                self.hand_pos = tuple(hand)
                self.viewer.send_to_sim(
                    user_messages.HandMove(x=hand[0], y=hand[1], z=hand[2])
                )
            if imgui.Button('Grab nearest'):
                self.viewer.send_to_sim(user_messages.GrabRequest())
            imgui.SameLine()
            if imgui.Button('Open hand'):
                self.viewer.send_to_sim(user_messages.ReleaseRequest())
            imgui.SetItemTooltip(
                'Drag the hand near a part and it closes on it (the part is\n'
                'welded to the hand: held rigidly, but still collides).\n'
                'Open hand lets go.'
            )

            imgui.Separator()
            if imgui.Button('Reset / drop everything'):
                self.viewer.send_to_sim(user_messages.ResetRequest())
            imgui.SameLine()
            if imgui.Button('Drop tenon'):
                self.viewer.send_to_sim(user_messages.DropRequest(part='tenon'))

            t = self.telemetry
            imgui.TextUnformatted(f't = {t.sim_time:.2f} s')
            imgui.TextUnformatted(
                f'Holding: {t.gripping or "(nothing)"}'
            )
            imgui.TextUnformatted(
                'Straight tenon: '
                + ('SEATED' if t.straight_seated else f'gap {t.straight_gap * 1000:.0f} mm')
            )
            imgui.TextUnformatted(
                'Cross-lap:      ' + ('LOCKED' if t.lap_seated else 'open')
            )

            imgui.Separator()
            if imgui.Button('Bake to scene.xml'):
                self.viewer.send_to_sim(user_messages.BakeRequest())
            imgui.SetItemTooltip('Write the current scene to scene.xml and show it.')
        finally:
            imgui.End()

# (panel reload marker)
