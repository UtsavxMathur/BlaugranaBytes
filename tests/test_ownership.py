"""The controlled-possession convention (training/ownership.py) on small made-up scenes."""

import numpy as np

from possession.tracking import Tracking
from training import ownership

H, A, D = 0, 1, 2
PITCH = (105.0, 68.0)
SPOTS = {"H1": (20.0, 30.0), "H2": (40.0, 30.0), "A1": (60.0, 30.0), "A2": (80.0, 30.0)}
FAR = (50.0, 60.0)          # nobody within reach


def scene(ball_at, labels, height=None):
    """ball_at: [(first, last, player name or None)] -> (Tracking, labels array)."""
    n = len(labels)
    players = np.tile(np.array(list(SPOTS.values())) / PITCH, (n, 1, 1))
    ball = np.tile(np.array(FAR) / PITCH, (n, 1))
    for first, last, who in ball_at:
        if who:
            ball[first:last] = np.array(SPOTS[who]) / PITCH
    tracking = Tracking(ball=ball, players=players, team=np.array([0, 0, 1, 1], dtype=np.int8),
                        pitch=PITCH, periods=[(0, n)],
                        ball_z=None if height is None else np.full(n, height, dtype=np.float32))
    return tracking, np.array(labels, dtype=np.int8)


def run(ball_at, labels, height=None):
    tracking, labels = scene(ball_at, labels, height)
    return ownership.relabel(tracking, labels)


def spells(*parts):
    return [code for code, n in parts for _ in range(n)]


def test_a_touch_by_the_other_team_is_not_a_change():
    labels = spells((H, 100), (A, 3), (H, 97))
    out, given_back = run([(0, 60, "H1"), (100, 103, "A1"), (150, 200, "H2")], labels)
    assert given_back == 1 and (out[100:103] == H).all()


def test_keeping_the_ball_for_a_second_is_a_change():
    labels = spells((H, 100), (A, 60), (H, 40))
    out, given_back = run([(0, 60, "H1"), (100, 130, "A1"), (170, 200, "H2")], labels)
    assert given_back == 0 and (out[100:160] == A).all()


def test_a_completed_pass_is_a_change():
    labels = spells((H, 100), (A, 20), (H, 80))
    out, _ = run([(0, 60, "H1"), (100, 103, "A1"), (110, 113, "A2"), (150, 200, "H2")], labels)
    assert (out[100:120] == A).all()


def test_a_pass_the_opponent_touches_in_between_is_not_a_change():
    labels = spells((H, 100), (A, 20), (H, 80))
    out, _ = run([(0, 60, "H1"), (100, 103, "A1"), (105, 107, "H2"), (110, 113, "A2"),
                  (150, 200, "H2")], labels)
    assert (out[100:120] == H).all()


def test_the_first_possession_after_a_stoppage_is_kept():
    labels = spells((D, 50), (A, 5), (H, 145))
    out, given_back = run([(80, 110, "H1")], labels)
    assert (out[50:55] == A).all() and (out[:50] == D).all()


def test_a_ball_in_the_air_is_not_controlled():
    labels = spells((H, 100), (A, 60), (H, 40))
    out, given_back = run([(0, 60, "H1"), (100, 130, "A1"), (170, 200, "H2")], labels, height=3.0)
    assert given_back == 1 and (out[100:160] == H).all()
