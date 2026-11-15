"""The whole-number sequence network (possession/seqnet.py) against PyTorch."""

import numpy as np
import pytest

from possession import seqnet

pytest.importorskip("torch")


def make(seed=0, n_features=12, steps=900):
    import torch
    from training import sequence
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = sequence.network(2 * n_features).eval()
    with torch.no_grad():                      # weights of a trained network's size, not tiny ones
        for p in net.parameters():
            p.mul_(3.0)
    X = rng.normal(2.0, 3.0, size=(steps, n_features))
    X[rng.random(X.shape) < 0.1] = np.nan
    scaler = sequence.Scaler(X)
    arrays = seqnet.from_torch(net.state_dict(), scaler.mean, scaler.std, sequence.STRIDE,
                               sequence.DILATIONS)
    return net, scaler, seqnet.Network(arrays), X


def test_whole_numbers_match_pytorch():
    from training import sequence
    net, scaler, integer, X = make()
    floating = sequence.scores(net, scaler, X)
    exact = integer.scores(X)
    assert exact.shape == floating.shape == (len(X), 2)
    assert np.abs(exact - floating).max() < 1e-3 * max(1.0, np.abs(floating).max())


def test_whole_numbers_are_repeatable_and_saved_exactly(tmp_path):
    net, scaler, integer, X = make(seed=1)
    path = tmp_path / "net.npz"
    np.savez_compressed(path, **integer.arrays)
    again = seqnet.Network.load(path)
    assert np.array_equal(integer.scores(X), again.scores(X))
