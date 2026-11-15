"""The sequence model in whole numbers: no PyTorch, and the same output on every machine.

training/sequence.py trains a temporal convolutional network in PyTorch. Floating-point matrix
products can add up in a different order on another machine, and then a last digit, and so a
decision, can change. So for the submission the network is stored with integer weights
(models/sequence.npz) and run here in integer arithmetic only:

    inputs and activations   real value x 2**SCALE_BITS, rounded down after every layer
    weights                  real value x 2**WEIGHT_BITS, rounded to the nearest integer
    products and sums        exact in int64 (numpy's integer matrix product does not use BLAS)

The rounding happens in the same way everywhere. tests/test_seqnet.py checks the result against
PyTorch.
"""

import os

import numpy as np

from .tracking import ROOT

DEFAULT_PATH = os.path.join(ROOT, "models", "sequence.npz")
SCALE_BITS = 16
WEIGHT_BITS = 16
LIMIT = 1 << 42          # an activation this large would mean the weights are not what we trained


def _quantise(values, bits):
    return np.floor(np.asarray(values, dtype=np.float64) * float(1 << bits) + 0.5).astype(np.int64)


def from_torch(state, mean, std, stride, dilations):
    """Integer arrays (dict, for np.savez) from a PyTorch state dict of training/sequence.py."""
    def tensor(name):
        return state[name].detach().cpu().double().numpy()

    out = {"mean": np.asarray(mean, dtype=np.float64), "std": np.asarray(std, dtype=np.float64),
           "stride": np.array(stride), "dilations": np.array(dilations, dtype=np.int64)}
    layers = [("inp", "inp")] + [("tap{}".format(i), "taps.{}".format(i)) for i in range(len(dilations))] \
        + [("mix{}".format(i), "mix.{}".format(i)) for i in range(len(dilations))] + [("out", "out")]
    for short, name in layers:
        weight, bias = tensor(name + ".weight"), tensor(name + ".bias")
        if short == "out":                       # only dead? and home?; a side task is dropped
            weight, bias = weight[:2], bias[:2]
        out[short + "_w"] = _quantise(weight, WEIGHT_BITS)
        out[short + "_b"] = _quantise(bias, WEIGHT_BITS + SCALE_BITS)
    return out


class Network:
    def __init__(self, arrays):
        import json
        self.arrays = {k: np.asarray(v) for k, v in arrays.items()}
        self.stride = int(self.arrays["stride"])
        self.dilations = [int(d) for d in self.arrays["dilations"]]
        self.names = json.loads(str(self.arrays["names"])) if "names" in self.arrays else None
        self.prices = json.loads(str(self.arrays["prices"])) if "prices" in self.arrays else None

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] for k in z.files})

    def inputs(self, X):
        """(channels, steps) int64: standardised features and missing flags, like sequence.Scaler."""
        mean, std = self.arrays["mean"], self.arrays["std"]
        missing = np.isnan(X)
        z = np.clip((np.where(missing, mean, X) - mean) / std, -10.0, 10.0)
        both = np.concatenate([z, missing.astype(np.float64)], axis=1)
        return _quantise(both, SCALE_BITS).T

    def _layer(self, name, x, dilation=None):
        w, b = self.arrays[name + "_w"], self.arrays[name + "_b"]
        if dilation is None:
            acc = np.matmul(w[:, :, 0], x)
        else:
            steps = x.shape[1]
            padded = np.zeros((x.shape[0], steps + 2 * dilation), dtype=np.int64)
            padded[:, dilation:dilation + steps] = x
            acc = sum(np.matmul(w[:, :, k], padded[:, k * dilation:k * dilation + steps])
                      for k in range(w.shape[2]))
        out = (acc + b[:, None]) >> WEIGHT_BITS
        if np.abs(out).max(initial=0) > LIMIT:
            raise OverflowError("sequence network activation out of range in layer " + name)
        return out

    def scores(self, X):
        """(steps, 2) log-odds of dead and home for one half, at the network's stride."""
        h = np.maximum(self._layer("inp", self.inputs(X)), 0)
        for i, d in enumerate(self.dilations):
            h = h + self._layer("mix{}".format(i),
                                np.maximum(self._layer("tap{}".format(i), h, dilation=d), 0))
        out = self._layer("out", np.maximum(h, 0))
        return (out.astype(np.float64) / float(1 << SCALE_BITS)).T

    def raw(self, plain, swapped, periods, n_frames):
        """Four (n,) score arrays at 25 fps, like the trees': dead and home, plain and swapped.
        NaN outside the periods."""
        out = [np.full(n_frames, np.nan) for _ in range(4)]
        for a, b in periods:
            s1, s2 = self.scores(plain[a:b:self.stride]), self.scores(swapped[a:b:self.stride])
            steps = np.arange(0, b - a, self.stride, dtype=np.float64)
            frames = np.arange(b - a, dtype=np.float64)
            for k, column in enumerate((s1[:, 0], s2[:, 0], s1[:, 1], s2[:, 1])):
                out[k][a:b] = np.interp(frames, steps, column)
        return out
