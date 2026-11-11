"""The organisers' reference baseline, on Tracking arrays.

    1. On every frame where the ball is seen, the team of the nearest tracked player has it.
    2. A centred majority vote over +/- half_width frames removes the flicker. D where nobody was
       seen in the whole window.

Same algorithm and tie-breaks as starter/baseline.py; tests/test_baseline.py checks it gives the
same labels as the original on synthetic EPTS files. It is here for two reasons:

  * it is the only model whose full-match score is published (spellF1 0.2433 at half_width=50),
    so it is our known floor;
  * reproducing evaluation/results/baseline_smooth50.csv exactly on the real data proves that our
    reader, frame indexing and home/away mapping are right.
"""

import numpy as np

from .tracking import AWAY, HOME

NOBODY = -1
HALF_WIDTH = 50  # frames each side: +/- 2 s at 25 fps


def nearest_team(tracking, chunk=50000):
    """int8 (n,): HOME / AWAY of the player nearest the ball, NOBODY where there is no answer."""
    length, width = tracking.pitch
    out = np.full(tracking.n_frames, NOBODY, dtype=np.int8)
    for start in range(0, tracking.n_frames, chunk):
        stop = min(start + chunk, tracking.n_frames)
        ball = tracking.ball[start:stop]
        players = tracking.players[start:stop]
        # Same arithmetic in the same order as the original: dx = px*L - bx*L, d = dx^2 + dy^2.
        dx = players[:, :, 0] * length - (ball[:, 0] * length)[:, None]
        dy = players[:, :, 1] * width - (ball[:, 1] * width)[:, None]
        distance = dx * dx + dy * dy
        distance[np.isnan(distance)] = np.inf      # a missing ball or player never wins
        best = distance.argmin(axis=1)             # first minimum on ties, like the original
        found = np.isfinite(distance[np.arange(stop - start), best])
        out[start:stop][found] = tracking.team[best[found]]
    return out


def majority_smooth(codes, half_width):
    """Centred majority vote -> array of 'H' / 'A' / 'D'. Ties go to H, as in the original."""
    if half_width <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    home_cum = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cum = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(codes), index + half_width + 1)
    home = home_cum[high] - home_cum[low]
    away = away_cum[high] - away_cum[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def predict(tracking, half_width=HALF_WIDTH):
    """Baseline labels for every frame: array of 'H' / 'A' / 'D'."""
    return majority_smooth(nearest_team(tracking), half_width)
