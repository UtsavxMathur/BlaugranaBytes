"""The broadcast recording, lined up frame by frame with the tactical one.

DATA.md: the broadcast feed has its own tracking (ATD) and homography, but it is "not
synchronised to tactical" and no mapping table is provided. Both trackings are positions on the
same pitch, though, so whenever the broadcast shows live play its players must stand where the
tactical players stand at that moment. That is how the two are lined up:

    1. coarse  for every 5 minutes of broadcast, the shift (1 s steps) at which its ball track
               best matches the tactical ball track
    2. fine    for every 10 seconds, the shift (1-frame steps, near a coarse one) at which its
               players best match the tactical players
    3. map     the shifts are joined into one table: tactical frame -> broadcast frame

Every tactical frame then gets a broadcast state:

    live     the broadcast shows the pitch and its players are where the tactical players are
    replay   it shows the pitch but its players are elsewhere: an earlier moment, replayed
    other    no pitch in the picture (close-up, crowd, graphics): its homography is not valid
    unclear  the pitch is shown, but too few players (or too far from both cases) to tell
    none     no broadcast frame for this moment

Only comparisons, +, -, *, sqrt, medians and integer sums are used, so the same bundle gives the
same map on every machine.
"""

import dataclasses
import math
import os
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import epts, geometry
from .competition import GAME, GAMES, require, source_frames
from .tracking import Tracking, cache_path, show_path

LIVE, REPLAY, OTHER, UNCLEAR, NONE = 0, 1, 2, 3, 4
STATES = ("live", "replay", "other", "unclear", "none")

# How the broadcast axes may relate to the tactical ones: (mirror x, mirror y).
FLIPS = ((False, False), (True, False), (False, True), (True, True))
FLIP_NAMES = ("as is", "x mirrored", "y mirrored", "x and y mirrored")

SECOND = 25            # frames per second of both recordings (DATA.md)

CHUNK = 7500           # broadcast frames per coarse chunk (5 minutes)
STEP = 25              # coarse search samples one frame per second
NEAR_M = 3.0           # two ball positions "match" within this distance
MIN_HITS = 15          # matching samples a chunk needs ...
CLEAR = 3.0            # ... and that many times more than the best shift APART seconds away
APART = 10

WINDOW = 250           # broadcast frames per fine window (10 s)
EVERY = 5              # one frame in EVERY is compared inside a window
SEARCH = 40            # fine search covers +- this many frames around a coarse shift
MIN_PLAYERS = 3        # broadcast players a frame needs to be compared at all
MIN_FRAMES = 10        # compared frames a window needs
GOOD_M = 1.5           # a window counts if its median player distance is below this ...
SHARP_M = 0.25         # ... and at least this much lower one second either side
NEIGHBOURS = 4         # shifts are compared with the median of +- this many good windows ...
OUTLIER = 4            # ... and dropped if further from it than this (frames)
JUMP = 25              # a change of more than this between neighbouring windows is a cut
CAP_M = 10.0           # one frame's distance counts at most this much when placing a cut

LIVE_M, REPLAY_M = 2.0, 4.0   # per-frame player distance: live below, replay above
SMOOTH = 6                    # the state uses the median distance over +- this many frames
HOLE = 2                      # tactical frames a growing shift skips are filled from the left


@dataclasses.dataclass
class Alignment:
    flip: int              # index into FLIPS: how the broadcast axes were turned
    flip_totals: list      # per FLIPS entry: (clear chunks, matching samples in them)
    chunks: list           # coarse result per chunk (dicts), for the chosen flip
    windows: list          # fine result per window (dict or None)
    stretches: list        # runs of windows with one shift (dicts)
    shift: np.ndarray      # (n_broadcast,) tactical frame = broadcast frame + shift
    t_to_b: np.ndarray     # (n_tactical,) broadcast frame shown at each tactical frame, -1 = none
    state: np.ndarray      # (n_tactical,) int8, one of LIVE ... NONE
    distance: np.ndarray   # (n_tactical,) median player distance behind the state, metres
    ball: np.ndarray       # (n_tactical, 2) broadcast ball on live frames (0..1, tactical axes)


# -- reading ----------------------------------------------------------------------------------------

def frame_ids(meta_path, raw_path):
    """(smallest frame id, largest frame id, number of lines) of an EPTS raw file."""
    layout = epts.read_metadata(meta_path, raw_path).layouts[0]
    first = last = None
    lines = 0
    with open(raw_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            frame_id = int(line.split(layout.sep, layout.i_frame + 1)[layout.i_frame])
            first = frame_id if first is None else min(first, frame_id)
            last = frame_id if last is None else max(last, frame_id)
            lines += 1
    return first, last, lines


def read_homography(path, n_frames):
    """(valid (n,) bool, reprojection error (n,) float) from a homography CSV (DATA.md)."""
    import pandas as pd
    table = pd.read_csv(path, usecols=["frame", "h00", "reprojection_error", "valid"])
    rows = table["frame"].to_numpy(dtype=np.int64)
    keep = (rows >= 0) & (rows < n_frames)
    valid = np.zeros(n_frames, dtype=bool)
    error = np.full(n_frames, np.nan)
    flag = table["valid"].to_numpy() == 1
    has_matrix = ~np.isnan(table["h00"].to_numpy(dtype=np.float64))
    valid[rows[keep]] = (flag & has_matrix)[keep]
    error[rows[keep]] = table["reprojection_error"].to_numpy(dtype=np.float64)[keep]
    return valid, error


def load(data_dir=None, use_cache=True, verbose=True):
    """(Tracking, homography valid, reprojection error) of the broadcast recording.

    Rows are broadcast video frames, not tactical frames: align() relates the two.
    The parsed arrays are cached like the tactical ones (gitignored, never to be shared).
    """
    meta_path, raw_path, homography_path = require(
        ["atd_meta", "atd_raw", "homography"], GAME, data_dir, source="broadcast")
    expected = source_frames("broadcast", GAME)
    cache = cache_path([meta_path, raw_path], tag="competition-broadcast")
    if use_cache and os.path.isfile(cache):
        if verbose:
            print("  using cached broadcast arrays {}".format(show_path(cache)))
        tracking = Tracking.load(cache)
    else:
        if verbose:
            print("reading the broadcast ATD (takes a little while) ...")
        first, last, lines = frame_ids(meta_path, raw_path)
        # The tactical file numbers its frames 1..n (DATA.md); expect the same here, but cope
        # with 0..n-1, and say so if it is neither.
        if last == expected:
            base, n = 1, expected
        elif last == expected - 1:
            base, n = 0, expected
        else:
            base, n = first, last - first + 1
            print("  NOTE: broadcast frame ids run {}..{} ({} lines), expected 1..{}; row 0 is "
                  "taken to be id {}".format(first, last, lines, expected, first))
        if verbose:
            print("  broadcast frame ids {}..{} on {} lines".format(first, last, lines))
        tracking = epts.load(meta_path, raw_path, n_frames=n, pitch=GAMES[GAME]["pitch"],
                             verbose=verbose, first_id=base)
        if use_cache:
            tracking.save(cache)
    valid, error = read_homography(homography_path, tracking.n_frames)
    return tracking, valid, error


# -- comparing positions ----------------------------------------------------------------------------

def oriented(xy, flip):
    """Broadcast positions (0..1) turned to the tactical axes by FLIPS[flip]."""
    mirror_x, mirror_y = FLIPS[flip]
    out = np.array(xy, dtype=np.float64, copy=True)
    if mirror_x:
        out[..., 0] = 1.0 - out[..., 0]
    if mirror_y:
        out[..., 1] = 1.0 - out[..., 1]
    return out


def _match(P, Q):
    """Per frame: median over P's players of the distance to the nearest player of Q (metres).

    P (..., s, 2) and Q (..., r, 2), NaN for an absent player, leading dimensions broadcast.
    NaN where P has fewer than MIN_PLAYERS players or Q has none.
    """
    shape = np.broadcast_shapes(P.shape[:-2], Q.shape[:-2])
    if P.shape[-2] == 0 or Q.shape[-2] == 0:
        return np.full(shape, np.nan)
    dx = P[..., :, None, 0] - Q[..., None, :, 0]
    dy = P[..., :, None, 1] - Q[..., None, :, 1]
    d2 = dx * dx + dy * dy
    d2[np.isnan(d2)] = np.inf
    nearest = np.sqrt(d2.min(axis=-1))
    nearest[~(np.isfinite(nearest) & ~np.isnan(P[..., 0]))] = np.nan
    enough = np.isfinite(nearest).sum(axis=-1) >= MIN_PLAYERS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(nearest, axis=-1)
    return np.where(enough, median, np.nan)


def _match_frames(b_pl, b_rows, t_pl, t_rows, chunk=2000):
    """_match for pairs of frames (broadcast row, tactical row), in chunks to bound memory."""
    out = np.full(len(b_rows), np.nan)
    for start in range(0, len(b_rows), chunk):
        part = slice(start, start + chunk)
        out[part] = _match(b_pl[b_rows[part]], t_pl[t_rows[part]])
    return out


def _rolling_nanmedian(values, half):
    padded = np.concatenate([np.full(half, np.nan), values, np.full(half, np.nan)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(sliding_window_view(padded, 2 * half + 1), axis=1)


# -- step 1: coarse, from the ball ------------------------------------------------------------------

def coarse(t_ball, b_ball, b_seen):
    """Best ball-track match for each CHUNK of broadcast frames (positions in metres).

    For every shift d on a 1-second grid (tactical frame = broadcast frame + d), count the
    sampled frames on which both balls are seen within NEAR_M of each other.
    """
    ts = t_ball[::STEP]
    rows = []
    for c0 in range(0, len(b_ball), CHUNK):
        bs = np.where(b_seen[c0:c0 + CHUNK:STEP, None], b_ball[c0:c0 + CHUNK:STEP], np.nan)
        k = len(bs)
        if k < 60:
            continue
        pad = np.full((k, 2), np.nan)
        tp = np.concatenate([pad, ts, pad])
        dx = sliding_window_view(tp[:, 0], k) - bs[:, 0]
        dy = sliding_window_view(tp[:, 1], k) - bs[:, 1]
        hits = (dx * dx + dy * dy < NEAR_M * NEAR_M).sum(axis=1)   # NaN compares False
        j = int(np.argmax(hits))
        others = hits.copy()
        others[max(0, j - APART):j + APART + 1] = 0
        runner_up = int(others.max())
        rows.append(dict(start=c0, offset=STEP * (j - k) - c0, hits=int(hits[j]),
                         both=int((~np.isnan(dx[j] + dy[j])).sum()), runner_up=runner_up,
                         clear=bool(hits[j] >= MIN_HITS and hits[j] >= CLEAR * runner_up)))
    return rows


def flip_totals(tables):
    """Per axis turn: (clear chunks, matching samples in them)."""
    return [(sum(row["clear"] for row in table), sum(row["hits"] for row in table if row["clear"]))
            for table in tables]


def choose_flip(tables):
    """Index of the axis turn whose clear chunks hold the most matching samples."""
    return int(np.argmax([hits for _, hits in flip_totals(tables)]))


# -- step 2: fine, from the players -----------------------------------------------------------------

def _scores(t_pl, P, frames, shifts):
    """Median over `frames` of the player distance at each shift; NaN if too few compare."""
    n_t = len(t_pl)
    t = frames[None, :] + shifts[:, None]
    inside = (t >= 0) & (t < n_t)
    Q = t_pl[np.clip(t, 0, n_t - 1)]
    Q[~inside] = np.nan
    Q = Q[:, :, ~np.isnan(Q[..., 0]).all(axis=(0, 1))]
    per_frame = _match(P[None], Q)
    ok = np.isfinite(per_frame).sum(axis=1) >= MIN_FRAMES
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        score = np.nanmedian(per_frame, axis=1)
    return np.where(ok, score, np.nan)


def _best_near(t_pl, P, frames, base):
    """(shift, score) minimising the player distance within base +- SEARCH, or None."""
    grid = np.arange(base - SEARCH, base + SEARCH + 1, 4)
    scores = _scores(t_pl, P, frames, grid)
    if np.isnan(scores).all():
        return None
    centre = int(grid[int(np.nanargmin(scores))])
    grid = np.arange(centre - 3, centre + 4)
    scores = _scores(t_pl, P, frames, grid)
    if np.isnan(scores).all():
        return None
    i = int(np.nanargmin(scores))
    return int(grid[i]), float(scores[i])


def fine(t_pl, b_pl, valid, chunks):
    """Frame-accurate shift of every WINDOW of broadcast frames (players in metres)."""
    n_b = len(b_pl)
    usable = valid & ((~np.isnan(b_pl[..., 0])).sum(axis=1) >= MIN_PLAYERS)
    clear = {row["start"] // CHUNK: row["offset"] for row in chunks if row["clear"]}
    everywhere = sorted(set(clear.values()))
    rows = []
    for w0 in range(0, n_b, WINDOW):
        frames = np.arange(w0, min(w0 + WINDOW, n_b), EVERY)
        frames = frames[usable[frames]]
        c = w0 // CHUNK
        bases = sorted({clear[i] for i in (c - 1, c, c + 1) if i in clear}) or everywhere
        if len(frames) < MIN_FRAMES or not bases:
            rows.append(None)
            continue
        P = b_pl[frames]
        P = P[:, ~np.isnan(P[..., 0]).all(axis=0)]
        found = [hit for hit in (_best_near(t_pl, P, frames, base) for base in bases) if hit]
        if not found:
            rows.append(None)
            continue
        shift, score = min(found, key=lambda hit: (hit[1], hit[0]))
        side = _scores(t_pl, P, frames, np.array([shift - SECOND, shift + SECOND]))
        sharp = float(np.nanmin(side) - score) if np.isfinite(side).any() else float("nan")
        rows.append(dict(start=w0, offset=shift, score=score, sharp=sharp, frames=len(frames)))
    return rows


# -- step 3: one shift per broadcast frame, then the tactical -> broadcast table --------------------

def good_windows(windows):
    return [w for w in windows if w is not None and w["score"] <= GOOD_M
            and w["sharp"] >= SHARP_M]


def _cut(a, b, before, after, t_pl, b_pl, valid):
    """First broadcast frame in [a, b) to take the `after` shift: the cut the players fit best."""
    frames = np.arange(a, b)
    costs = []
    for shift in (before, after):
        t = frames + shift
        ok = np.flatnonzero(valid[frames] & (t >= 0) & (t < len(t_pl)))
        error = np.zeros(len(frames))
        error[ok] = _match_frames(b_pl, frames[ok], t_pl, t[ok])
        error = np.nan_to_num(np.minimum(error, CAP_M), nan=0.0)   # no players: no opinion
        costs.append(np.round(error * 100).astype(np.int64))     # cm, so the sums are exact
    if not (costs[0].any() or costs[1].any()):
        return (a + b + 1) // 2
    total = (np.concatenate([[0], np.cumsum(costs[0])])
             + np.concatenate([np.cumsum(costs[1][::-1])[::-1], [0]]))
    return a + int(np.argmin(total))


def _running_median(values, half):
    """Lower median of values[i - half : i + half + 1] at every i (integers in, integers out)."""
    out = np.empty_like(values)
    for i in range(len(values)):
        part = np.sort(values[max(0, i - half):i + half + 1])
        out[i] = part[(len(part) - 1) // 2]
    return out


def _stretch_ids(shift):
    return np.concatenate([[0], np.cumsum(np.abs(np.diff(shift)) > JUMP)])


def shifts(windows, n_b, t_pl, b_pl, valid):
    """(shift of every broadcast frame (n_b,) int64, [stretch dicts]) from the good windows.

    One window's shift is a little noisy (a frame or three), and now and then a window in a slow
    spell fits a wrong one. So: a change of more than JUMP frames starts a new stretch (a cut in
    the recording), and stretches of fewer than 3 windows are dropped as flukes; inside a
    stretch, windows far from the running median of their neighbours are dropped; and the shift
    used is the running median of the windows that remain.
    """
    good = good_windows(windows)
    if not good:
        raise ValueError("no stretch of the broadcast matches the tactical tracking")
    centre = np.array([w["start"] + WINDOW // 2 for w in good], dtype=np.int64)
    shift = np.array([w["offset"] for w in good], dtype=np.int64)
    score = np.array([w["score"] for w in good])

    keep = np.bincount(_stretch_ids(shift))[_stretch_ids(shift)] >= 3
    centre, shift, score = centre[keep], shift[keep], score[keep]
    stretch = _stretch_ids(shift)
    keep = np.zeros(len(shift), dtype=bool)
    for k in range(int(stretch[-1]) + 1):
        members = stretch == k
        keep[members] = np.abs(shift[members]
                               - _running_median(shift[members], NEIGHBOURS)) <= OUTLIER
    centre, shift, score = centre[keep], shift[keep], score[keep]
    if not len(shift):
        raise ValueError("no stretch of the broadcast matches the tactical tracking")
    stretch = _stretch_ids(shift)
    smooth = shift.copy()
    for k in range(int(stretch[-1]) + 1):
        members = stretch == k
        smooth[members] = _running_median(shift[members], NEIGHBOURS)

    out = np.empty(n_b, dtype=np.int64)
    out[:centre[0]] = smooth[0]
    out[centre[-1]:] = smooth[-1]
    for i in range(len(centre) - 1):
        a, b = centre[i], centre[i + 1]
        if stretch[i] == stretch[i + 1]:
            middle = (a + b + 1) // 2
            out[a:middle], out[middle:b] = smooth[i], smooth[i + 1]
        else:
            cut = _cut(a, b, smooth[i], smooth[i + 1], t_pl, b_pl, valid)
            out[a:cut], out[cut:b] = smooth[i], smooth[i + 1]

    runs = []
    for k in range(int(stretch[-1]) + 1):
        members = np.flatnonzero(stretch == k)
        runs.append(dict(first=int(centre[members[0]] - WINDOW // 2),
                         last=int(centre[members[-1]] + WINDOW // 2 - 1),
                         shift=int(_running_median(smooth[members], len(members))[0]),
                         low=int(smooth[members].min()), high=int(smooth[members].max()),
                         windows=int(len(members)), dropped=len(good) - len(shift),
                         score=float(np.median(score[members]))))
    return out, runs


def tactical_map(shift, n_t):
    """(n_t,) broadcast frame shown at each tactical frame, -1 where there is none."""
    b = np.arange(len(shift), dtype=np.int64)
    t = b + shift
    inside = (t >= 0) & (t < n_t)
    t, b = t[inside], b[inside]
    out = np.full(n_t, -1, dtype=np.int64)
    unique, first = np.unique(t, return_index=True)
    out[unique] = b[first]
    starts, lengths = geometry.runs_of(out < 0)
    for s, n in zip(starts, lengths):
        if n <= HOLE and s > 0 and s + n < n_t:
            out[s:s + n] = out[s - 1]
    return out


def frame_states(t_pl, b_pl, valid, t_to_b):
    """(state (n_t,) int8, smoothed player distance (n_t,)) for every tactical frame."""
    n_t = len(t_to_b)
    mapped = t_to_b >= 0
    b = np.where(mapped, t_to_b, 0)
    shown = mapped & valid[b]
    distance = np.full(n_t, np.nan)
    rows = np.flatnonzero(shown)
    distance[rows] = _match_frames(b_pl, b[rows], t_pl, rows)
    smooth = _rolling_nanmedian(distance, SMOOTH)
    state = np.full(n_t, NONE, dtype=np.int8)
    state[mapped] = OTHER
    state[shown] = UNCLEAR
    with np.errstate(invalid="ignore"):
        state[shown & (smooth <= LIVE_M)] = LIVE
        state[shown & (smooth >= REPLAY_M)] = REPLAY
    return state, smooth


def align(tactical, broadcast, valid):
    """Line the broadcast Tracking up with the tactical one. Returns an Alignment."""
    pitch = np.asarray(tactical.pitch, dtype=np.float64)
    t_ball = tactical.ball * pitch
    t_pl = tactical.players * pitch
    b_seen = valid & broadcast.ball_visible()
    tables = [coarse(t_ball, oriented(broadcast.ball, flip) * pitch, b_seen)
              for flip in range(len(FLIPS))]
    flip = choose_flip(tables)
    b_pl = oriented(broadcast.players, flip) * pitch
    windows = fine(t_pl, b_pl, valid, tables[flip])
    shift, runs = shifts(windows, broadcast.n_frames, t_pl, b_pl, valid)
    t_to_b = tactical_map(shift, tactical.n_frames)
    state, distance = frame_states(t_pl, b_pl, valid, t_to_b)
    ball = np.full((tactical.n_frames, 2), np.nan)
    live = state == LIVE
    ball[live] = oriented(broadcast.ball[t_to_b[live]], flip)
    return Alignment(flip=flip, flip_totals=flip_totals(tables), chunks=tables[flip],
                     windows=windows, stretches=runs,
                     shift=shift, t_to_b=t_to_b, state=state, distance=distance, ball=ball)


# -- evidence: while the broadcast is not showing live play, the ball is usually dead -------------

NONLIVE_JOIN = SECOND          # non-live pictures less than 1 s apart count as one stretch
NONLIVE_MIN = 2 * SECOND       # shorter stretches are ignored
UNTIL_LIVE_S = (2, 5, 10)      # stages of a stretch: seconds until live pictures come back
MIN_SUPPORT = 10 * SECOND      # a stage needs this many graded frames to be calibrated
MAX_RATIO = 20.0               # no stage counts for more than this likelihood ratio
COST_SCALE = 100               # decode.SCALE: cost units per unit of log-probability
STAGE_NAMES = ("last 2 s", "2-5 s before", "5-10 s before", "earlier")


def nonlive_stages(state, periods):
    """(n,) int8 stage of every frame in a non-live stretch (replay or no pitch), counted back
    from the moment live pictures return: 0 = the last 2 s ... 3 = more than 10 s; -1 elsewhere.

    Directors cut away a second or more after the ball goes dead and often come back only after
    the restart (a replay running on), so the end of a stretch is weaker evidence than the rest.
    """
    nonlive = geometry.close_gaps((state == OTHER) | (state == REPLAY), NONLIVE_JOIN)
    edges = np.array(UNTIL_LIVE_S, dtype=np.int64) * SECOND
    stage = np.full(len(state), -1, dtype=np.int8)
    for a, b in periods:
        for s, n in zip(*geometry.runs_of(nonlive[a:b])):
            if n >= NONLIVE_MIN:
                stage[a + s:a + s + n] = np.searchsorted(edges, np.arange(n, 0, -1), side="left")
    return stage


def dead_evidence(state, p_dead, periods, exclude=()):
    """(extra cost of H and A per frame (n,) int64, [one dict per stage]).

    Calibrated on the match itself, without labels: for each stage, the likelihood ratio
    P(stage | dead) / P(stage | live) is estimated with the model's own frame probabilities as
    soft counts over the graded frames (minus `exclude`, e.g. the wait before a kick-off). In
    the decoder, adding COST_SCALE * ln(ratio) to H and A is the same as multiplying the odds of
    D by the ratio, so a frame changes only where the model was unsure. Stages with a ratio of
    1 or less, and stages too rare to calibrate, add nothing.
    """
    stage = nonlive_stages(state, periods)
    graded = np.zeros(len(state), dtype=bool)
    for a, b in periods:
        graded[a:b] = True
    for a, b in exclude:
        graded[a:b] = False
    p = np.round(np.nan_to_num(p_dead, nan=0.0) * 1000).astype(np.int64)   # integer soft counts
    dead_all, live_all = int(p[graded].sum()), int((1000 - p[graded]).sum())
    extra = np.zeros(len(state), dtype=np.int64)
    rows = []
    for k, name in enumerate(STAGE_NAMES):
        here = graded & (stage == k)
        dead, live = int(p[here].sum()), int((1000 - p[here]).sum())
        calibrated = int(here.sum()) >= MIN_SUPPORT and live > 0 and dead_all > 0
        ratio = (dead * live_all) / (live * dead_all) if calibrated else 1.0
        cost = int(round(COST_SCALE * math.log(min(ratio, MAX_RATIO)))) if ratio > 1.0 else 0
        extra[stage == k] = cost
        rows.append(dict(stage=name, frames=int(here.sum()), ratio=ratio, cost=cost,
                         calibrated=calibrated))
    return extra, rows
