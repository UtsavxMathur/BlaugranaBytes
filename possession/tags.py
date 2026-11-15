"""Evidence from the Smart Tagging file, used when the frame probabilities are decoded.

The tags are automatic and can be wrong (DATA.md), so a tag never overrides the model on its own.
It is used for one thing only: goal to kick-off. A GOALS tag runs until just before the next
"Kick Off (After Goal)" tag, and in between the ball is dead. If the model already calls most of
that stretch dead, all of it is set to dead: this removes stray "possessions" while players
celebrate or wait for a review, when a ball is sometimes tracked. The stretch ends where the
kick-off tag starts; the seconds after that are left to the model.

Nothing here is measured on the match. The two constants below are design choices, and the only
inputs are the tags and the model's own probabilities. (An earlier version also timed set-piece
restarts from the tags and the ball's speed, with a typical delay per kind of set piece measured
on the match itself. This version leaves restarts to the model.)
"""

import numpy as np

from . import smart_tagging

D = 2
GOAL_MARGIN_S = 20.0              # the first seconds of a GOALS tag are left to the model
GOAL_AGREEMENT = 0.5              # ... and the model must call this share of the rest dead


def goal_stretches(tags, prob, fps):
    """([(start, stop)], n_candidates): goal-to-kick-off stretches the model agrees are dead."""
    goals = [t for t in tags if t.code == "goals"]
    kick_offs = [t for t in tags if t.code == smart_tagging.SET_PIECES
                 and t.kind == smart_tagging.KICK_OFF_GOAL]
    windows, candidates = [], 0
    for goal in goals:
        later = [k for k in kick_offs if goal.start < k.start <= goal.stop + 5 * fps]
        if not later:
            continue
        kick_off = min(later, key=lambda k: k.start)
        start = goal.start + int(GOAL_MARGIN_S * fps)
        stop = kick_off.start
        if stop - start < 5 * fps:
            continue
        candidates += 1
        part = prob[start:stop, D]
        if np.isfinite(part).all() and part.mean() >= GOAL_AGREEMENT:
            windows.append((start, stop))
    return windows, candidates
