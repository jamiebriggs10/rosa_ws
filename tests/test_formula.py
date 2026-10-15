"""Tests for move_formula's safe evaluator: plain maths works, anything else is refused.

Run on the ZenBook:  ./venv/bin/python -m pytest tests
"""
import math

import pytest

from robot_tools import evaluate_formula


def test_circle_has_the_right_points_and_closes():
    points = evaluate_formula("[[8*cos(2*pi*i/16) - 8, 8*sin(2*pi*i/16), 0] for i in range(17)]")
    assert len(points) == 17
    assert points[0] == pytest.approx([0, 0, 0]) and points[-1] == pytest.approx([0, 0, 0], abs=1e-9)
    assert all(math.dist(p, [-8, 0, 0]) == pytest.approx(8) for p in points)


def test_spiral_with_a_named_value():
    points = evaluate_formula("[[10*t*cos(20*pi*t), 10*t*sin(20*pi*t), 10*t] for i in range(121) for t in [i/120]]")
    assert len(points) == 121
    assert points[-1] == pytest.approx([10, 0, 10], abs=1e-9)


@pytest.mark.parametrize("bad", [
    "__import__('os').system('ls')",
    "[().__class__ for i in range(3)]",
    "[[open('x'), 0, 0] for i in range(3)]",
    "[[0, 0, 0] for i in range(10**6)]",
    "[[0, 0, 0] for i in range(9) for j in range(9)]",
    "[['a', 0, 0] for i in range(3)]",
    "[[w, 0, 0] for i in range(3)]",
    "(lambda: 1)()",
])
def test_anything_but_plain_maths_is_refused(bad):
    with pytest.raises(ValueError):
        evaluate_formula(bad)


def test_x_y_z_mean_the_tools_own_position():
    points = evaluate_formula("[[x + 8*t, y + 3*sin(4*pi*t), z] for i in range(9) for t in [i/8]]")
    assert points[0] == pytest.approx([0, 0, 0]) and points[-1][0] == pytest.approx(8)


def test_absolute_coordinates_around_the_tool_are_read_as_absolute(monkeypatch):
    """A circle written around the tool's position (64.5, -37.4, 103.8) instead of around [0, 0, 0]."""
    import config
    from robot_tools import _as_offsets
    monkeypatch.setattr(config, "MAX_PATH_REACH_CM", 30.0)
    start = [64.5, -37.4, 103.8]
    circle = [[64.5 + 5 * math.cos(2 * math.pi * i / 8), -37.4 + 5 * math.sin(2 * math.pi * i / 8), 103.8] for i in range(9)]
    offsets, note = _as_offsets(circle, start)
    assert "absolute" in note
    assert offsets[0] == pytest.approx([5, 0, 0]) and max(math.hypot(*p) for p in offsets) == pytest.approx(5)
    # genuinely relative points are left alone, with no note
    assert _as_offsets([[0, 0, 5], [0, 5, 5]], start) == ([[0, 0, 5], [0, 5, 5]], "")
    # far-away points that are not around the tool stay as they are, so the reach check refuses them
    far = [[-40, -68, 6], [-58, -68, 6]]
    assert _as_offsets(far, start) == (far, "")
