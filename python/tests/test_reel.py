"""Reel composition: grid layout, pacing, and chapter spans (no sim, no GL)."""

import numpy as np
import pytest

from pi_robotics import reel


def _frame(value):
    panel = np.zeros((6, 8, 3), dtype=np.uint8)
    panel[..., 0] = value
    camera = np.zeros((6, 8, 3), dtype=np.uint8)
    camera[..., 1] = value
    return {"panel": panel, "cameras": {"camera_rgb": camera}}


def test_compose_video_lays_out_panel_and_cameras(tmp_path):
    frames = [_frame(10), _frame(20), _frame(30)]
    stamps = [0.0, 0.1, 0.2]
    result = reel.compose_video(
        tmp_path,
        frames,
        stamps,
        chapters=[(0, "attempt 1")],
        fps=10,
        hold=0.1,
        tail=0.0,
        name="unit",
    )
    assert result["steps"] == 3
    assert result["columns"] == ["panel", "camera_rgb"]
    assert result["rows"] == 1
    assert result["cols"] == 2
    assert result["chapters"] == [
        {"start": 0.0, "end": 0.3, "title": "attempt 1"}
    ]
    assert result["path"]
    suffix = ".mp4" if result["format"] == "mp4" else ".gif"
    assert (tmp_path / f"unit{suffix}").exists()


def _static(times, value):
    return [(t, np.array([float(value)])) for t in times]


def test_condense_static_timeline_caps_idle_runs():
    # 2.0 s of still frames, then two moving frames, then 1.5 s of still frames.
    states = _static([0.0, 0.5, 1.0, 1.5, 2.0], 0)
    states.append((2.5, np.array([1.0])))
    states += _static([3.0, 3.5, 4.0, 4.5], 2)
    condensed, chapters = reel.condense_static_timeline(
        states, [(6, "attempt")], max_idle=1.0
    )
    # Each long idle run keeps at most 1 s, and the dropped wall time is
    # removed so the remaining stamps flow without a stall.
    assert [round(t, 3) for t, _ in condensed] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert [float(q[0]) for _, q in condensed] == [0, 0, 0, 1, 2, 2, 2]
    # The chapter at raw index 6 is remapped to its position after condensing.
    assert chapters == [(4, "attempt")]


def test_condense_static_timeline_keeps_short_runs_and_disables_at_zero():
    states = [(0.0, np.array([0.0])), (0.5, np.array([0.0])), (0.6, np.array([9.0]))]
    kept, _ = reel.condense_static_timeline(states, [], max_idle=1.0)
    assert len(kept) == 3  # a 0.5 s pause is below the cap
    off, _ = reel.condense_static_timeline(states, [], max_idle=0.0)
    assert len(off) == 3


def test_compose_video_rejects_empty_frames(tmp_path):
    with pytest.raises(ValueError):
        reel.compose_video(tmp_path, [], [], [], name="empty")


def test_recorder_timeline_round_trips():
    rec = reel.Recorder()
    rec.clear()
    rec.add_state(np.array([1.0, 2.0]))
    rec.mark_chapter("first")
    rec.add_state(np.array([3.0, 4.0]))
    states, chapters = rec.timeline()
    assert rec.count() == 2
    assert chapters == [(1, "first")]
    assert len(states) == 2
    # timeline() copies, so mutating the returned array does not leak.
    states[0][1][0] = 99.0
    again, _ = rec.timeline()
    assert again[0][1][0] == pytest.approx(1.0)
