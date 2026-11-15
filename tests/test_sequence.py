"""The sequence model (training/sequence.py): runs end to end and is exactly fair between teams."""

import numpy as np
import pytest

from possession import features
from possession.tracking import Tracking
from test_broadcast import PITCH, made_up_play

pytest.importorskip("torch")


def test_swapping_the_teams_swaps_home_and_away_exactly():
    from training import sequence
    ball, players = made_up_play(3000, seed=3)
    tracking = Tracking(ball=ball, players=players, team=np.repeat([0, 1], 11).astype(np.int8),
                        pitch=PITCH, periods=[(0, 1500), (1500, 3000)])
    X, _ = features.compute(tracking)
    labels = np.where(np.arange(3000) % 600 < 300, 0, 1).astype(np.int8)
    labels[1000:1100] = 2
    ones = np.ones(3000, dtype=np.float32)
    parts = [(X[a:b:sequence.STRIDE], labels[a:b:sequence.STRIDE], ones[a:b:sequence.STRIDE],
              ones[a:b:sequence.STRIDE]) for a, b in tracking.periods]
    net, scaler = sequence.fit(parts, steps=5)

    p = sequence.predict_probabilities(net, scaler, tracking)
    q = sequence.predict_probabilities(net, scaler, features.swap_teams(tracking))
    assert p.shape == (3000, 3) and np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[:, 0], q[:, 1], atol=1e-9) and np.allclose(p[:, 1], q[:, 0], atol=1e-9)
    assert np.allclose(p[:, 2], q[:, 2], atol=1e-9)


def test_event_heads_mark_changes_stoppages_restarts_and_ignore_the_team_swap(monkeypatch):
    from training import sequence
    labels = np.array([3] * 50 + [0] * 100 + [1] * 100 + [2] * 50 + [0] * 100, dtype=np.int8)
    targets = sequence.event_targets(labels, 0, len(labels))
    assert targets.shape == (80, 3)
    assert np.isnan(targets[:10]).all()                      # unknown labels: no target
    assert set(np.flatnonzero(targets[:, 0] == 1)) == {29, 30, 31}     # change of team, frame 150
    assert set(np.flatnonzero(targets[:, 1] == 1)) == {49, 50, 51}     # stoppage, frame 250
    assert set(np.flatnonzero(targets[:, 2] == 1)) == {59, 60, 61}     # restart, frame 300
    assert np.nansum(targets) == 9

    ball, players = made_up_play(3000, seed=5)
    tracking = Tracking(ball=ball, players=players, team=np.repeat([0, 1], 11).astype(np.int8),
                        pitch=PITCH, periods=[(0, 1500), (1500, 3000)])
    X, _ = features.compute(tracking)
    labels = np.where(np.arange(3000) % 600 < 300, 0, 1).astype(np.int8)
    labels[1000:1100] = 2
    ones = np.ones(3000, dtype=np.float32)
    step = sequence.STRIDE
    parts = [(X[a:b:step], labels[a:b:step], ones[a:b:step], ones[a:b:step],
              np.full(((b - a) // step, 3), np.nan, np.float32), sequence.event_targets(labels, a, b))
             for a, b in tracking.periods]
    monkeypatch.setattr(sequence, "CHANGE", True)
    net, scaler = sequence.fit(parts, steps=3)
    assert net.out.out_channels == 5
    swapped = features.swap_teams(tracking)
    plain_X, _ = features.compute(tracking)
    swapped_X, _ = features.compute(swapped)
    p = sequence.predict_events(net, scaler, tracking, plain_X, swapped_X)
    q = sequence.predict_events(net, scaler, swapped, swapped_X, plain_X)
    assert p.shape == (3000, 3) and ((p > 0) & (p < 1)).all()
    assert np.allclose(p, q, atol=1e-12)


def test_recipe_options_train_and_export_exactly(monkeypatch):
    from possession import seqnet
    from training import sequence
    ball, players = made_up_play(3000, seed=6)
    tracking = Tracking(ball=ball, players=players, team=np.repeat([0, 1], 11).astype(np.int8),
                        pitch=PITCH, periods=[(0, 1500), (1500, 3000)])
    X, _ = features.compute(tracking)
    labels = np.where(np.arange(3000) % 600 < 300, 0, 1).astype(np.int8)
    ones = np.ones(3000, dtype=np.float32)
    step = sequence.STRIDE
    parts = [(X[a:b:step], labels[a:b:step], ones[a:b:step], ones[a:b:step])
             for a, b in tracking.periods]
    for name, value in (("WIDTH", 32), ("DROPOUT", 0.1), ("COSINE", True)):
        monkeypatch.setattr(sequence, name, value)
    net, scaler = sequence.fit(parts, steps=4)
    assert net.out.in_channels == 32 and hasattr(net, "drop") and not net.training
    floating = sequence.scores(net, scaler, X[0:1500:step])
    integer = seqnet.Network(seqnet.from_torch(net.state_dict(), scaler.mean, scaler.std, step,
                                               sequence.DILATIONS))
    exact = integer.scores(X[0:1500:step])
    assert np.abs(exact - floating).max() < 1e-3 * max(1.0, np.abs(floating).max())


def test_ball_side_task_learns_and_is_left_out_of_the_whole_number_network(monkeypatch):
    from possession import seqnet
    from training import sequence
    ball, players = made_up_play(3000, seed=4)
    ball[500:700] = np.nan
    height = np.zeros(3000, dtype=np.float32)
    height[100:200], height[300:310] = 4.0, np.nan
    tracking = Tracking(ball=ball, players=players, team=np.repeat([0, 1], 11).astype(np.int8),
                        pitch=PITCH, periods=[(0, 1500), (1500, 3000)], ball_z=height)

    targets = sequence.ball_targets(tracking)
    assert targets.shape == (3000, 3)
    assert np.isnan(targets[500:700, :2]).all() and np.isfinite(targets[:500, :2]).all()
    assert ((targets[:500, :2] >= 0) & (targets[:500, :2] <= 1)).all()
    assert (targets[100:200, 2] == 1).all() and (targets[:100, 2] == 0).all()
    assert np.isnan(targets[300:310, 2]).all()

    X, _ = features.compute(tracking)
    labels = np.where(np.arange(3000) % 600 < 300, 0, 1).astype(np.int8)
    ones = np.ones(3000, dtype=np.float32)
    step = sequence.STRIDE
    parts = [(X[a:b:step], labels[a:b:step], ones[a:b:step], ones[a:b:step], targets[a:b:step])
             for a, b in tracking.periods]
    monkeypatch.setattr(sequence, "AUX", True)
    net, scaler = sequence.fit(parts, steps=3)
    assert net.out.out_channels == 5

    floating = sequence.scores(net, scaler, X[0:1500:step])
    integer = seqnet.Network(seqnet.from_torch(net.state_dict(), scaler.mean, scaler.std, step,
                                               sequence.DILATIONS))
    exact = integer.scores(X[0:1500:step])
    assert floating.shape == exact.shape == (300, 2)
    assert np.abs(exact - floating).max() < 1e-3 * max(1.0, np.abs(floating).max())
