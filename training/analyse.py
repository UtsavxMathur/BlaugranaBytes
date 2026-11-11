"""Where do the held-out errors come from? Possessions found / missed by length, boundary timing.

    python training/analyse.py                 # uses the predictions saved by training/train.py

Reads the held-out predictions in .cache/train/ (public games only).
"""

import bisect
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import model as model_module  # noqa: E402
from possession.submission import pm  # noqa: E402
from training import train  # noqa: E402

BINS = [0, 1, 2, 4, 8, 16, 1e9]


def best_match(spell, others):
    """(IoU, partner) of the best same-team overlap among spells sorted by start."""
    i = bisect.bisect_left(others, (spell[0],))
    best, partner = 0.0, None
    for other in others[max(0, i - 8):i + 8]:
        if other[2] != spell[2] or other[1] <= spell[0] or other[0] >= spell[1]:
            continue
        inter = min(spell[1], other[1]) - max(spell[0], other[0])
        union = max(spell[1], other[1]) - min(spell[0], other[0])
        if inter / union > best:
            best, partner = inter / union, other
    return best, partner


def analyse(oof, prices, source, regime="typical"):
    found, total = np.zeros(len(BINS) - 1), np.zeros(len(BINS) - 1)
    right, made = np.zeros(len(BINS) - 1), np.zeros(len(BINS) - 1)
    starts, stops = [], []
    for key in [k for k in oof if k[0].startswith(source) and k[1] == regime]:
        prob, _, truth, periods = oof[key]
        pred = model_module.decode_periods(prob, periods, prices)
        true_spells = sorted(pm.possession_spells(truth, list(periods)))
        pred_spells = sorted(pm.possession_spells(pred, list(periods)))
        for spell in true_spells:
            iou, partner = best_match(spell, pred_spells)
            k = int(np.searchsorted(BINS, (spell[1] - spell[0]) / 25.0, side="right")) - 1
            total[k] += 1
            found[k] += iou >= 0.5
            if iou >= 0.5:
                starts.append((partner[0] - spell[0]) / 25.0)
                stops.append((partner[1] - spell[1]) / 25.0)
        for spell in pred_spells:
            iou, _ = best_match(spell, true_spells)
            k = int(np.searchsorted(BINS, (spell[1] - spell[0]) / 25.0, side="right")) - 1
            made[k] += 1
            right[k] += iou >= 0.5
    names = ["{}-{}s".format(a, b if b < 1e8 else "") for a, b in zip(BINS[:-1], BINS[1:])]
    print("{} / {}".format(source, regime))
    print("  true possessions : " + "  ".join("{} {:.0f} ({:.0%} found)".format(n, t, f / max(t, 1))
                                             for n, t, f in zip(names, total, found)))
    print("  predicted        : " + "  ".join("{} {:.0f} ({:.0%} right)".format(n, m, r / max(m, 1))
                                             for n, m, r in zip(names, made, right)))
    print("  matched: start error s p10/50/90 {:.2f} {:.2f} {:.2f}; stop error {:.2f} {:.2f} {:.2f}".format(
        *np.percentile(starts, [10, 50, 90]), *np.percentile(stops, [10, 50, 90])))


if __name__ == "__main__":
    games = [name for fold in train.FOLDS for name in fold]
    oof = train.load_oof(games)
    with open(os.path.join(train.WORK_DIR, "prices.json")) as handle:
        prices = json.load(handle)
    print("decoder prices", prices)
    for source in ("metrica", "dfl"):
        analyse(oof, prices, source)
