"""Small numeric helpers shared by the features, the simulator and the profile script.

Everything works in metres and uses only +, -, *, / and sqrt, which give bit-identical results on
every machine (that matters: RULES.md wants the same file out of a fresh clone).
"""

import numpy as np

from .tracking import AWAY, HOME


def runs_of(mask):
    """(starts, lengths) of the maximal runs of True in a boolean array."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2] - edges[0::2]


def to_metres(tracking):
    """(ball (n, 2), players (n, s, 2)) in metres, float32. NaN stays NaN."""
    scale = np.array(tracking.pitch, dtype=np.float64)
    return ((tracking.ball * scale).astype(np.float32),
            (tracking.players * scale).astype(np.float32))


def team_distances(ball, players, team, k=3, chunk=20000):
    """(n, 2, k) float32: distance from the ball to the k nearest HOME and AWAY players.

    inf where the ball is missing or the team has fewer than that many tracked players.
    """
    n = len(ball)
    out = np.full((n, 2, k), np.inf, dtype=np.float32)
    columns = [np.flatnonzero(team == HOME), np.flatnonzero(team == AWAY)]
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dx = players[start:stop, :, 0] - ball[start:stop, None, 0]
        dy = players[start:stop, :, 1] - ball[start:stop, None, 1]
        distance = np.sqrt(dx * dx + dy * dy)
        distance[np.isnan(distance)] = np.inf
        for side, cols in enumerate(columns):
            part = np.sort(distance[:, cols], axis=1)[:, :k]
            out[start:stop, side, :part.shape[1]] = part
    return out


def speed_of(xy, fps, half_span=2):
    """Speed in m/s from positions (n, ..., 2): |p[t+h] - p[t-h]| / (2h / fps).

    NaN where either end is missing, and on the first and last h frames.
    """
    h = half_span
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    if len(xy) > 2 * h:
        step = xy[2 * h:] - xy[:-2 * h]
        out[h:-h] = np.sqrt((step * step).sum(axis=-1)) * (fps / (2.0 * h))
    return out


def edge_distance(xy, pitch):
    """Distance in metres from a position to the nearest boundary line. Negative = off the pitch."""
    length, width = pitch
    x, y = xy[..., 0], xy[..., 1]
    return np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))
