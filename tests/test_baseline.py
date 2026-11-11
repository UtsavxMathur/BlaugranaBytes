"""Our EPTS reader + baseline must reproduce the organisers' starter/baseline.py exactly."""

import importlib.util
import os

import numpy as np
import pytest

from possession import baseline, epts
from synthetic_epts import random_match, write_epts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PITCH = (105.3, 68.0)


@pytest.fixture(scope="module")
def official():
    """The organisers' baseline module, imported from its file."""
    spec = importlib.util.spec_from_file_location(
        "official_baseline", os.path.join(ROOT, "starter", "baseline.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("with_spec", [True, False])
def test_nearest_team_matches_official(tmp_path, official, with_spec):
    n_frames = 3000
    players, frames = random_match(n_frames, n_slots=10, seed=1)
    meta, raw = write_epts(tmp_path, players, frames, with_spec=with_spec)
    expected = official.nearest_player_labels(raw, official.team_of_each_track(meta),
                                              n_frames, PITCH)
    tracking = epts.load(meta, raw, n_frames=n_frames, pitch=PITCH, verbose=False)
    got = baseline.nearest_team(tracking)
    assert (expected == baseline.NOBODY).sum() > 100     # the test covers missing balls
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("half_width", [0, 1, 25, 50, 125])
def test_smoothing_matches_official(official, half_width):
    rng = np.random.default_rng(half_width)
    codes = rng.choice(np.array([0, 1, -1], dtype=np.int8), size=5000, p=[0.4, 0.3, 0.3])
    np.testing.assert_array_equal(baseline.majority_smooth(codes, half_width),
                                  official.smooth(codes, half_width))


def test_reader_values_and_teams(tmp_path):
    players = [("P1", "T1"), ("P2", "T2"), ("P3", "T1")]
    frames = [(1, {"P1": (0.1, 0.2, 3.0), "P3": (0.5, 0.6, 1.5)}, (0.11, 0.21, 0.0)),
              (2, {"P2": (0.9, 0.8, 7.0)}, None)]
    meta, raw = write_epts(tmp_path, players, frames)
    tracking = epts.load(meta, raw, verbose=False)
    assert tracking.n_frames == 2 and tracking.n_slots == 3
    assert tracking.periods == [(0, 2)]
    np.testing.assert_array_equal(tracking.team, [0, 1, 0])
    assert tracking.team_names == ("FC Barcelona", "Athletic Club")
    np.testing.assert_allclose(tracking.players[0, 0], [0.1, 0.2])
    np.testing.assert_allclose(tracking.players[0, 2], [0.5, 0.6])
    assert np.isnan(tracking.players[0, 1]).all()
    np.testing.assert_allclose(tracking.speed[1, 1], 7.0)
    np.testing.assert_allclose(tracking.ball[0], [0.11, 0.21])
    assert np.isnan(tracking.ball[1]).all()


def test_reader_follows_layout_per_frame_range(tmp_path):
    """Sample Game 3 style: the layout (players and their order) changes mid-file."""
    players = [("P1", "T1"), ("P2", "T2"), ("P3", "T1"), ("P4", "T2")]
    layouts = [(1, 2, ["P1", "P2"]), (3, 4, ["P4", "P3", "P1"])]
    frames = [(1, {"P1": (0.1, 0.1), "P2": (0.2, 0.2)}, (0.5, 0.5)),
              (2, {"P1": (0.1, 0.1), "P2": (0.2, 0.2)}, (0.5, 0.5)),
              (3, {"P4": (0.4, 0.4), "P3": (0.3, 0.3), "P1": (0.1, 0.1)}, (0.5, 0.5)),
              (4, {"P4": (0.4, 0.4), "P3": (0.3, 0.3), "P1": (0.1, 0.1)}, (0.5, 0.5))]
    meta, raw = write_epts(tmp_path, players, frames, layouts=layouts, channels=("x", "y"),
                           ball_channels=("x", "y"))
    tracking = epts.load(meta, raw, verbose=False)
    np.testing.assert_allclose(tracking.players[2, 3], [0.4, 0.4])   # P4 -> slot 3
    np.testing.assert_allclose(tracking.players[2, 2], [0.3, 0.3])   # P3 -> slot 2
    assert np.isnan(tracking.players[2, 1]).all()                     # P2 subbed off
    assert np.isnan(tracking.speed).all()                             # no speed channel


def test_reader_tolerates_frames_beyond_n_frames(tmp_path):
    players, frames = random_match(50, n_slots=4, seed=3)
    meta, raw = write_epts(tmp_path, players, frames)
    assert epts.load(meta, raw, n_frames=40, verbose=False).n_frames == 40


def test_tracking_cache_round_trip(tmp_path):
    players, frames = random_match(60, n_slots=6, seed=5)
    meta, raw = write_epts(tmp_path, players, frames)
    tracking = epts.load(meta, raw, periods=[(5, 20), (30, 55)], verbose=False)
    path = str(tmp_path / "cache.npz")
    tracking.save(path)
    again = type(tracking).load(path)
    np.testing.assert_array_equal(again.ball, tracking.ball)
    np.testing.assert_array_equal(again.players, tracking.players)
    assert again.periods == [(5, 20), (30, 55)] and again.pitch == tracking.pitch
    assert again.in_periods().sum() == 15 + 25
