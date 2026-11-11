"""The model: features -> two tree ensembles -> decoder -> one label per frame.

Two questions are asked of every frame:
    dead?   is the ball out of play                      (ensemble "dead")
    home?   if it is in play, does the home team have it (ensemble "team")

Both are asked twice, once as the data is and once with the teams exchanged, and the answers are
combined so the model cannot prefer "home" over "away": swapping the teams exactly swaps H and A.

The decoder (decode.py) then turns the frame probabilities into whole possessions.

A model file (models/*.npz) holds the trees, the feature names they expect and the decoder
prices. It is produced by training/train.py from public games only.
"""

import dataclasses
import json
import os

import numpy as np

from . import decode, features, trees
from .tracking import ROOT

DEFAULT_PATH = os.path.join(ROOT, "models", "possession.npz")
DRIFT_PERCENTILES = (5, 25, 50, 75, 95)


@dataclasses.dataclass
class Model:
    names: list        # feature names, in the order the trees index them
    dead: dict         # trees: log-odds that the ball is dead
    team: dict         # trees: log-odds that home has the ball, given it is alive
    prices: dict       # decoder prices: {"flip": .., "stop": .., "start": .., "cap": ..}
    info: dict         # how it was trained (for the technical document)

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arrays = {"dead_" + key: value for key, value in self.dead.items()}
        arrays.update({"team_" + key: value for key, value in self.team.items()})
        meta = json.dumps({"names": self.names, "prices": self.prices, "info": self.info},
                          sort_keys=True)
        np.savez_compressed(path, meta=np.array(meta), **arrays)

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            dead = {name: z["dead_" + name] for name in trees.FIELDS}
            team = {name: z["team_" + name] for name in trees.FIELDS}
        return cls(names=meta["names"], dead=dead, team=team, prices=meta["prices"],
                   info=meta["info"])


def combine(dead_raw, dead_raw_swapped, team_raw, team_raw_swapped):
    """(n, 3) probabilities of H, A, D from the four raw scores (plain and teams-swapped)."""
    dead = 0.5 * (dead_raw + dead_raw_swapped)          # "dead" must not depend on the swap
    home = 0.5 * (team_raw - team_raw_swapped)          # "home" must flip sign under the swap
    with np.errstate(over="ignore"):
        p_dead = 1.0 / (1.0 + np.exp(-dead))
        p_home = 1.0 / (1.0 + np.exp(-home))
    return np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], axis=1)


def probabilities(model, tracking):
    """(n_frames, 3) probabilities of H, A, D. Rows outside the halves are NaN."""
    plain, names = features.compute(tracking)
    swapped, _ = features.compute(features.swap_teams(tracking))
    if names != model.names:
        raise ValueError("the model was trained on different features; retrain it")
    inside = tracking.in_periods()
    out = np.full((tracking.n_frames, 3), np.nan)
    out[inside] = combine(trees.raw_score(model.dead, plain[inside]),
                          trees.raw_score(model.dead, swapped[inside]),
                          trees.raw_score(model.team, plain[inside]),
                          trees.raw_score(model.team, swapped[inside]))
    return out


def decode_periods(prob, periods, prices, forced_dead=()):
    """int8 codes (0 H, 1 A, 2 D, 3 X outside the halves) from frame probabilities.

    forced_dead  [(start, stop)] frame ranges known to be dead ball (e.g. before a kick-off).
    """
    labels = np.full(len(prob), 3, dtype=np.int8)
    for start, stop in periods:
        cost = decode.frame_costs(prob[start:stop], prices.get("cap", decode.CAP))
        for a, b in forced_dead:
            a, b = max(a, start), min(b, stop)
            if b > a:
                cost[a - start:b - start, :2] = decode.CAP * 50
                cost[a - start:b - start, 2] = 0
        labels[start:stop] = decode.viterbi(cost, prices["flip"], prices["stop"], prices["start"])
    return labels


def predict(model, tracking, forced_dead=()):
    """int8 label codes for every frame of a Tracking."""
    return decode_periods(probabilities(model, tracking), tracking.periods, model.prices,
                          forced_dead)


def drift_report(model, tracking):
    """Text table: every feature on this recording next to the same feature in training.

    Aggregates only (percentiles over all graded frames). A feature whose values here sit far
    from what the model saw in training is a place where the simulator does not match reality.
    """
    stats = model.info.get("feature_stats")
    if not stats:
        return "  (this model file has no training statistics; retrain to get a drift report)"
    X, names = features.compute(tracking)
    X = X[tracking.in_periods()]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        here = np.nanpercentile(X, stats["percentiles"], axis=0).T
    missing = np.isnan(X).mean(axis=0)
    lines = ["  {:<22} {:>13}   {:>31}   {:>31}  {}".format(
        "feature", "missing here/train", "here p5/25/50/75/95", "training p5/25/50/75/95", "shift")]
    rows = []
    for i, name in enumerate(names):
        train = np.array(stats["values"][i], dtype=float)
        spread = max(train[3] - train[1], 0.05 * max(abs(train[4] - train[0]), 1e-6), 1e-6)
        shift = abs(here[i][2] - train[2]) / spread if np.isfinite(here[i][2]) else float("nan")
        shift = max(shift, abs(missing[i] - stats["missing"][i]) * 5)
        rows.append((shift if np.isfinite(shift) else 99.0, "  {:<22} {:>6.1%} /{:>6.1%}   {}   {}  {:.2f}".format(
            name, missing[i], stats["missing"][i],
            " ".join("{:>5.1f}".format(v) for v in here[i]),
            " ".join("{:>5.1f}".format(v) for v in train), shift)))
    rows.sort(key=lambda row: -row[0])
    return "\n".join(lines + [text for _, text in rows])
