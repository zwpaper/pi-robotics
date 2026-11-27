"""Shared scene messages for the pi-robotics Studio workspace (榫卯 / joinery).

`sim_plugin.py` and `viewer_plugin.py` both import this module, so editing and
saving it hot-reloads the running simulation.
"""

import dataclasses

from mujoco.studio import messages


@dataclasses.dataclass(frozen=True)
class SceneConfig(messages.Snapshot):
    """UI -> sim: live physics parameters (applied in place, no rebuild)."""

    gravity: float = 9.81  # Downward gravity magnitude (m/s^2).
    friction: float = 0.6  # Sliding friction of every wood geom.


@dataclasses.dataclass(frozen=True)
class ResetRequest(messages.Event):
    """UI -> sim: lift every component back above its target and drop it."""


@dataclasses.dataclass(frozen=True)
class DropRequest(messages.Event):
    """UI -> sim: drop the straight-tenon block onto its socket."""

    part: str = "tenon"


@dataclasses.dataclass(frozen=True)
class ReleaseRequest(messages.Event):
    """UI -> sim: open the hand and let go of whatever it is holding."""


@dataclasses.dataclass(frozen=True)
class GrabRequest(messages.Event):
    """UI -> sim: close the hand on the nearest part."""


@dataclasses.dataclass(frozen=True)
class HandMove(messages.Event):
    """UI -> sim: move the kinematic hand to this world position (m).

    A left-click alternative to the viewer's Ctrl+right-drag translation.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.55


@dataclasses.dataclass(frozen=True)
class BakeRequest(messages.Event):
    """UI -> sim: write the current scene to scene.xml and show that file."""


@dataclasses.dataclass(frozen=True)
class Telemetry(messages.Snapshot):
    """sim -> UI: latest-wins summary of the assembly state."""

    sim_time: float = 0.0
    # Lateral distance (m) of the straight tenon from the centre of its socket.
    straight_gap: float = 999.0
    straight_seated: bool = False  # tenon centred over the socket and down in it
    lap_seated: bool = False  # the two half-lapped beams are interlocked flush
    gripping: str = ""  # name of the part the hand is currently holding, or ""
    hand_pos: tuple = (0.0, 0.0, 0.55)  # current world position of the hand
