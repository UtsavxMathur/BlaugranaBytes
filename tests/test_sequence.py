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
