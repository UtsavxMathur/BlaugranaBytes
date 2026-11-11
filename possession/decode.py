"""Frame probabilities -> whole possessions.

The classifier scores every frame on its own, so its raw output flickers, and the ranking metric
(spellF1) counts possessions, not frames: every flicker is an invented possession. The decoder
finds the label sequence that best explains the probabilities while paying a price for every
change of label (Viterbi algorithm on three states H, A, D):

    flip    price of H <-> A      (possession changes hands without a stoppage)
    stop    price of H/A -> D     (the ball goes dead)
    start   price of D -> H/A     (play restarts)
    cap     the most a single frame can count against a label

Higher prices = fewer, longer possessions. The prices are chosen on held-out public games.

All arithmetic is on integers, so the result is identical on every machine.
"""

import numpy as np

H, A, D = 0, 1, 2
SCALE = 100          # cost units per unit of log-probability
CAP = 1200           # default cap on one frame's cost (about exp(-12))
TIE = 1 << 20        # costs are multiplied by this; the low bits count label changes, so that
                     # of two equally good paths the one with fewer changes wins


def frame_costs(prob, cap=CAP):
    """(n, 3) probabilities for H, A, D -> (n, 3) integer costs (lower = more likely).

    cap limits how much a single frame can cost. Neighbouring frames are almost copies of each
    other, so without a cap one confident second would outweigh any price.
    """
    cost = np.round(-SCALE * np.log(np.maximum(prob, 1e-12)))
    return np.minimum(cost, cap).astype(np.int64)


def viterbi(cost, flip, stop, start, first=None, last=None):
    """Cheapest label sequence for one stretch of frames.

    cost               (n, 3) integer frame costs from frame_costs()
    flip, stop, start  prices, scalars or (c,) arrays to decode with c settings at once
    first, last        force the first / last frame's label (e.g. D before a kick-off)

    Returns int8 labels: (n,) for scalar prices, (c, n) for arrays.
    """
    scalar = np.ndim(flip) == 0
    flip, stop, start = (np.atleast_1d(np.asarray(v, dtype=np.int64)) for v in (flip, stop, start))
    c, n = len(flip), len(cost)
    if n == 0:
        return np.zeros((0,) if scalar else (c, 0), dtype=np.int8)

    move = np.zeros((c, 3, 3), dtype=np.int64)          # move[k, from, to]
    move[:, H, A] = move[:, A, H] = flip * TIE + 1
    move[:, H, D] = move[:, A, D] = stop * TIE + 1
    move[:, D, H] = move[:, D, A] = start * TIE + 1
    frame = cost.astype(np.int64) * TIE
    big = np.int64(1) << 60

    value = np.broadcast_to(frame[0], (c, 3)).copy()
    if first is not None:
        value[:, [s for s in (H, A, D) if s != first]] = big
    back = np.zeros((n, c, 3), dtype=np.int8)
    for t in range(1, n):
        through = value[:, :, None] + move               # (c, from, to)
        back[t] = through.argmin(axis=1)
        value = through.min(axis=1) + frame[t]
    if last is not None:
        value[:, [s for s in (H, A, D) if s != last]] = big

    labels = np.zeros((c, n), dtype=np.int8)
    state = value.argmin(axis=1)
    rows = np.arange(c)
    for t in range(n - 1, -1, -1):
        labels[:, t] = state
        state = back[t, rows, state]
    return labels[0] if scalar else labels
