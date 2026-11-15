"""One possession convention for all training games: the challenge's "controlled possession".

CHALLENGE.md: possession changes only when a player of the other team gains *controlled*
possession; a touch without control, a deflection that does not intercept, or an aerial duel
without control is not a change; a loose or contested ball stays with the last team that had it;
after a restart there is a clear possessor.

Our two label sources follow other conventions. DFL's possession flag changes on single touches;
the Metrica labels change at the other team's events. Both give far more short possessions than
that definition allows. So every possession spell that follows the other team's possession
directly (no stoppage in between) is checked against the clean tracking of the public game:

    controlled   a player of the new team keeps the ball for CONTROL_S (it is the nearest
                 player, within CONTROL_M, and the ball is no higher than HEIGHT_M), or two
                 players of the new team have it one after the other with nobody else in
                 between: a completed pass
    otherwise    the spell was a touch or a deflection, and it goes back to the team before

The first possession after a stoppage (or at the start of a half) is always kept, and dead-ball
frames never change. The constants come from the definition and from what a touch looks like at
25 frames per second, not from the provided match: nothing here reads it.

    python training/ownership.py        # how much this changes each public game
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from possession import geometry  # noqa: E402

H, A, D, X = 0, 1, 2, 3
CONTROL_M = 1.5        # the ball is at a player's feet: well inside this, with tracking noise
CONTROL_S = 1.0        # clearly longer than a touch or a deflection (a few frames at 25 fps)
HEIGHT_M = 2.0         # above this the ball is not under anyone's control
LENGTH_BINS_S = (0, 1, 2, 4, 8, np.inf)


def nearest(tracking):
    """(slot, team, close) per frame: the nearest player to the ball, his team, and whether the
    ball is within CONTROL_M of him and low enough to be controlled. slot = -1 when unknown."""
    scale = np.asarray(tracking.pitch, dtype=np.float64)
    ball = tracking.ball * scale
    n = len(ball)
    slot = np.full(n, -1, dtype=np.int64)
    distance = np.full(n, np.inf)
    for start in range(0, n, 20000):
        part = slice(start, start + 20000)
        dx = tracking.players[part, :, 0] * scale[0] - ball[part, None, 0]
        dy = tracking.players[part, :, 1] * scale[1] - ball[part, None, 1]
        d2 = dx * dx + dy * dy
        d2[np.isnan(d2)] = np.inf
        best = d2.argmin(axis=1)
        found = np.isfinite(d2[np.arange(len(best)), best])
        slot[part] = np.where(found, best, -1)
        distance[part] = np.where(found, np.sqrt(d2[np.arange(len(best)), best]), np.inf)
    team = np.where(slot >= 0, tracking.team[np.maximum(slot, 0)], -1)
    close = distance <= CONTROL_M
    if tracking.ball_z is not None:
        close &= ~(np.asarray(tracking.ball_z, dtype=np.float64) > HEIGHT_M)
    return slot, team, close


def controlled(slot, team, close, start, stop, side, fps):
    """True if `side` controls the ball somewhere in frames [start, stop)."""
    owner = np.where(close[start:stop], slot[start:stop], -1)
    mine = (owner >= 0) & (team[start:stop] == side)
    # one player keeps it
    hold = int(round(CONTROL_S * fps))
    for s, n in zip(*geometry.runs_of(mine)):
        values = owner[s:s + n]
        changes = np.flatnonzero(values[1:] != values[:-1]) + 1
        bounds = np.concatenate(([0], changes, [n]))
        if (np.diff(bounds) >= hold).any():
            return True
    # a completed pass: two of its players have it in turn, with no opponent in between
    touched = owner[owner >= 0]
    sides = team[start:stop][owner >= 0]
    if len(touched) < 2:
        return False
    keep = np.concatenate(([True], touched[1:] != touched[:-1]))
    touched, sides = touched[keep], sides[keep]
    return bool(((sides[1:] == side) & (sides[:-1] == side)).any())


def relabel(tracking, labels, periods=None):
    """(labels under the controlled-possession convention, number of spells given back)."""
    periods = tracking.periods if periods is None else periods
    slot, team, close = nearest(tracking)
    out = labels.copy()
    given_back = 0
    for a, b in periods:
        codes = labels[a:b]
        breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(codes)]))
        before = None                   # team in possession just before, None after a stoppage
        for s, e in zip(bounds[:-1], bounds[1:]):
            code = int(codes[s])
            if code not in (H, A):
                before = None
            elif before is None or code == before:
                before = code
            elif controlled(slot, team, close, a + s, a + e, code, tracking.fps):
                before = code
            else:
                out[a + s:a + e] = before
                given_back += 1
    return out, given_back


def summary(labels, periods, fps=25):
    """Possessions per live minute, share under 2 s, median length (s), live minutes."""
    lengths, live = [], 0
    for a, b in periods:
        codes = labels[a:b]
        live += int(((codes == H) | (codes == A)).sum())
        breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(codes)]))
        lengths += [(e - s) / float(fps) for s, e in zip(bounds[:-1], bounds[1:])
                    if codes[s] in (H, A)]
    lengths = np.array(lengths)
    minutes = live / float(fps) / 60.0
    return {"per_minute": len(lengths) / minutes if minutes else 0.0,
            "under_2s": float((lengths < 2).mean()) if len(lengths) else 0.0,
            "median_s": float(np.median(lengths)) if len(lengths) else 0.0,
            "minutes": minutes, "possessions": len(lengths)}


def main():
    from training import dataset
    print("{:<12} {:>7} {:>13} {:>13} {:>13}  {:>11} {:>10}".format(
        "game", "live min", "poss / min", "under 2 s", "median s", "given back", "frames moved"))
    totals = {}
    for name in dataset.ALL_GAMES:
        game = dataset.load(name, verbose=False)
        new, given_back = relabel(game.tracking, game.labels)
        old_s = summary(game.labels, game.tracking.periods)
        new_s = summary(new, game.tracking.periods)
        live = (game.labels == H) | (game.labels == A)
        moved = float((new[live] != game.labels[live]).mean())
        print("{:<12} {:>7.1f}   {:>4.1f} -> {:>4.1f}   {:>4.0%} -> {:>4.0%}   {:>4.1f} -> {:>4.1f}  {:>11} {:>10.1%}"
              .format(name, old_s["minutes"], old_s["per_minute"], new_s["per_minute"],
                      old_s["under_2s"], new_s["under_2s"], old_s["median_s"], new_s["median_s"],
                      given_back, moved))
        source = game.source
        t = totals.setdefault(source, {"minutes": 0.0, "old": 0, "new": 0})
        t["minutes"] += old_s["minutes"]
        t["old"] += old_s["possessions"]
        t["new"] += new_s["possessions"]
    for source, t in totals.items():
        print("{:<12} {:>7.1f}   {:>4.1f} -> {:>4.1f}  possessions per live minute".format(
            source, t["minutes"], t["old"] / t["minutes"], t["new"] / t["minutes"]))


if __name__ == "__main__":
    main()
