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
