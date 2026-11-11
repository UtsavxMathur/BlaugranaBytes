"""Evidence from the Smart Tagging file, used when the frame probabilities are decoded.

The tags are automatic and can be wrong (DATA.md), so a tag never overrides the model on its
own: each use below needs the tracking or the model to agree.

  1. Goal to kick-off. A GOALS tag runs until just before the next "Kick Off (After Goal)" tag,
     and in between the ball is dead. If the model already calls most of that stretch dead, all
     of it is set to dead: this removes stray "possessions" while players celebrate or wait for a
     review, when a ball is sometimes tracked.

  2. The seconds before a restart. Every set piece is a restart, and the ball is dead until it is
     struck. The moment it is struck shows in the tracking: the ball goes from slow or unseen to
     fast. For each set-piece tag we look for such moments near the tag's start; the tagging
     places its windows a fixed time before the restart, so the tags of one kind should agree on
     when the strike comes. If they do and this tag fits, the stretch up to the strike is dead. It is joined to
     the stoppage the model found just before, because the tracker often follows the ball again
     while a player carries it to the spot, which looks like play.

  3. A set-piece tag with no strike found and no stoppage anywhere near gets a short dead stretch
     where the restart usually is.

All timing comes from the tracking and the tags of the match itself. No label is involved.
"""

import numpy as np

from . import smart_tagging

D, OUTSIDE = 2, 3
KICK_OFF_DELAY_S = 3.4            # a kick-off is taken about this long after its tag starts
GOAL_MARGIN_S = 20.0              # the first seconds of a GOALS tag are left to the model
GOAL_AGREEMENT = 0.5              # ... and the model must call this share of the rest dead
SET_PIECE_KINDS = ("throw-in", "corner kick", "goal kick", "free kick")

FAST_MPS, KEEP_MPS, KEEP_S = 6.0, 4.0, 0.4   # "struck": faster than 6 m/s, staying over 4 ...
QUIET_MPS, QUIET_S = 3.5, (2.0, 0.4)         # ... after being under 3.5 m/s (or unseen) from 2.0 s
                                             # to 0.4 s before
SEARCH_S = (-2.0, 14.0)           # where to look for the strike, relative to the tag's start
MIN_TAGS, MAX_SPREAD_S = 4, 4.0   # a kind is used if 4+ tags agree within this spread (IQR)
FITS_S = 2.5                      # ... and a tag is used if it is this close to its kind's median
STRIKE_LAG_S = 0.5                # the dead stretch ends this long before the ball is fast
JOIN_S = 10.0                     # join to a stoppage the model found within this time before
ALONE_S = 3.5                     # otherwise set this long a stretch to dead
FALLBACK_S = (1.0, 4.5)           # no strike found, no stoppage near: dead stretch from tag start
NEARBY_S = (-6.0, 8.0)            # ... where "near" means a model stoppage in this window


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
        stop = kick_off.start + int((KICK_OFF_DELAY_S - 1.0) * fps)
        if stop - start < 5 * fps:
            continue
        candidates += 1
        part = prob[start:stop, D]
        if np.isfinite(part).all() and part.mean() >= GOAL_AGREEMENT:
            windows.append((start, stop))
    return windows, candidates


def strike_frames(speed, start, fps):
    """Frames near `start` where a quiet ball is struck: slow or unseen, then fast and staying fast.

    Requiring the quiet spell keeps out ordinary fast play just before the stoppage.
    """
    low = max(0, start + int(SEARCH_S[0] * fps))
    high = min(len(speed), start + int(SEARCH_S[1] * fps))
    keep, far, close = int(KEEP_S * fps), int(QUIET_S[0] * fps), int(QUIET_S[1] * fps)
    found, previous = [], -10 ** 9
    with np.errstate(invalid="ignore"):
        fast = np.flatnonzero(speed[low:high] > FAST_MPS) + low
    for frame in fast:
        if frame - previous < fps:                    # one strike per second at most
            continue
        after = speed[frame:frame + keep]
        before = speed[max(0, frame - far):max(0, frame - close)]
        if len(after) < keep or not np.nanmin(after) > KEEP_MPS:
            continue
        if np.isfinite(before).any() and np.nanmax(before) >= QUIET_MPS:
            continue
        found.append(int(frame))
        previous = frame
    return found


def strike_frame(speed, start, fps, expected=None):
    """The strike nearest `expected` seconds after `start` (the first one if None), or None."""
    found = strike_frames(speed, start, fps)
    if not found:
        return None
    if expected is None:
        return found[0]
    return min(found, key=lambda frame: abs((frame - start) / float(fps) - expected))


def restart_stretches(tags, codes, speed, fps):
    """Dead stretches before set-piece restarts. Returns (windows, report).

    codes  the model's labels from a first decoding pass (0 H, 1 A, 2 D, 3 outside the halves)
    speed  ball speed in m/s per frame, NaN where the ball is not tracked
    report {kind: (tags, strikes found, median s after tag start, spread s, used)} plus totals
    """
    pieces = [t for t in tags if t.code == smart_tagging.SET_PIECES and t.kind in SET_PIECE_KINDS
              and 0 <= t.start < len(codes) and codes[t.start] != OUTSIDE]
    strikes = {}
    windows, report = [], {}
    counts = {"joined": 0, "alone": 0, "fallback": 0}
    for kind in SET_PIECE_KINDS:
        group = [t for t in pieces if t.kind == kind]
        # The tagging puts its window a fixed time before the restart, so the tags of one kind
        # should agree on when the strike comes. Start from each tag's first strike, then let
        # every tag take the strike nearest the group's median, twice.
        expected = None
        for _ in range(3):
            for t in group:
                strikes[id(t)] = strike_frame(speed, t.start, fps, expected)
            known = [(strikes[id(t)] - t.start) / float(fps) for t in group
                     if strikes[id(t)] is not None]
            if not known:
                break
            expected = float(np.median(known))
        offsets = np.array([(strikes[id(t)] - t.start) / float(fps) for t in group
                            if strikes[id(t)] is not None])
        median = float(np.median(offsets)) if len(offsets) else float("nan")
        spread = float(np.subtract(*np.percentile(offsets, [75, 25]))) if len(offsets) else float("nan")
        usable = len(offsets) >= MIN_TAGS and spread <= MAX_SPREAD_S
        report[kind] = (len(group), len(offsets), median, spread, usable)
        for tag in group:
            strike = strikes[id(tag)]
            fits = (usable and strike is not None
                    and abs((strike - tag.start) / float(fps) - median) <= FITS_S)
            if fits:
                stop = strike - int(STRIKE_LAG_S * fps)
                look = max(0, stop - int(JOIN_S * fps))
                dead = np.flatnonzero(codes[look:stop] == D)
                if len(dead):
                    windows.append((look + int(dead[-1]), stop))        # join the model's stoppage
                    counts["joined"] += 1
                else:
                    windows.append((stop - int(ALONE_S * fps), stop))
                    counts["alone"] += 1
                continue
            low = max(0, tag.start + int(NEARBY_S[0] * fps))
            high = min(len(codes), tag.start + int(NEARBY_S[1] * fps))
            if not (codes[low:high] == D).any():
                windows.append((tag.start + int(FALLBACK_S[0] * fps),
                                tag.start + int(FALLBACK_S[1] * fps)))
                counts["fallback"] += 1
    report["counts"] = counts
    return [(a, b) for a, b in windows if b > a], report


def describe(report):
    """The restart report as text (aggregates per kind of set piece)."""
    lines = []
    for kind in SET_PIECE_KINDS:
        n, found, median, spread, usable = report[kind]
        lines.append("    {:<12} {:>2} tags, strike found for {:>2}: median {:>4.1f} s after the tag "
                     "starts, spread {:>4.1f} s -> {}".format(kind, n, found, median, spread,
                                                              "used" if usable else "not used"))
    c = report["counts"]
    lines.append("    dead stretch before the strike: joined to a stoppage the model found {}, "
                 "on its own {}; no strike and no stoppage near {}".format(
                     c["joined"], c["alone"], c["fallback"]))
    return "\n".join(lines)
