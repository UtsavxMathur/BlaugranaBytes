"""Tracking -> one row of numbers per frame, for the classifier.

Design rules:
  * Nothing depends on which slot a player is in (slots are tracklets, not people). Everything
    is "nearest", "how many", "average of the team".
  * Every feature copes with a missing ball. Short gaps are bridged with a straight line; the
    model is also told how long the ball has been gone and what the players are doing meanwhile.
  * The pitch is turned so that the home team always attacks towards +x, in both halves.
  * Windows look both back and forward in time (this is offline labelling, not live).
  * Each half is processed on its own, so nothing leaks across halftime.

Missing values stay NaN; the trees handle NaN natively. Values are rounded to 3 decimals so that
last-bit differences between machines cannot change a tree's decision.
"""

import warnings

import numpy as np

from . import geometry
from .tracking import AWAY, HOME

BRIDGE_S = 3.0             # ball gaps up to this long are bridged with a straight line
CONTROL_M = 2.5            # a player this close to the ball can play it
TIGHT_M = 1.0              # ... and this close has it at their feet
TOUCH_SWING = 4.0          # m/s change of ball velocity across a frame that counts as a touch
WINDOWS_S = (0.4, 1.0, 2.0, 4.0, 8.0, 16.0)
MAX_PLAYER_SPEED = 12.0    # m/s; anything faster is a tracking jump
WALK_SPEED = 0.8           # m/s
CLIP_S = 60.0


def _bridge(ball, vis, max_len):
    """Copy of the ball track with gaps of up to max_len frames filled by a straight line."""
    out = ball.copy()
    n = len(ball)
    for start, length in zip(*geometry.runs_of(~vis)):
        if length <= max_len and start > 0 and start + length < n:
            weight = (np.arange(1, length + 1, dtype=np.float32) / (length + 1))[:, None]
            out[start:start + length] = ball[start - 1] * (1 - weight) + ball[start + length] * weight
    return out


def _past(values, w):
    """Mean of values over the w frames ending at t (frames before the start count as 0)."""
    total = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    index = np.arange(len(values))
    return (total[index + 1] - total[np.maximum(0, index + 1 - w)]) / w


def _future(values, w):
    """Mean of values over the w frames after t (frames past the end count as 0)."""
    total = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    index = np.arange(len(values))
    n = len(values)
    return (total[np.minimum(n, index + 1 + w)] - total[np.minimum(n, index + 1)]) / w


def _nan_mean(values, low, high):
    """Mean of values[t+low : t+high] ignoring NaN (NaN if the window has none)."""
    good = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(good, values, 0.0), dtype=np.float64)))
    count = np.concatenate(([0], np.cumsum(good)))
    index = np.arange(len(values))
    a = np.clip(index + low, 0, len(values))
    b = np.clip(index + high, 0, len(values))
    n = count[b] - count[a]
    with np.errstate(all="ignore"):
        return np.where(n > 0, (total[b] - total[a]) / n, np.nan)


def _since(mask):
    """Frames since mask was last True (0 where it is True now; a huge number if never)."""
    index = np.arange(len(mask))
    last = np.maximum.accumulate(np.where(mask, index, -10 ** 9))
    return index - last


def _until(mask):
    """Frames until mask is next True (0 where it is True now; a huge number if never)."""
    index = np.arange(len(mask))
    nxt = np.minimum.accumulate(np.where(mask, index, 10 ** 9)[::-1])[::-1]
    return nxt - index


def _velocity(xy, fps, h=2):
    """Velocity vectors (same shape as xy), m/s, from positions h frames either side."""
    out = np.full(xy.shape, np.nan, dtype=np.float32)
    if len(xy) > 2 * h:
        out[h:-h] = (xy[2 * h:] - xy[:-2 * h]) * (fps / (2.0 * h))
    return out


def _period(ball, players, team, pitch, fps):
    """Features for one half: {name: (n,) array}."""
    length, width = pitch
    n = len(ball)
    index = np.arange(n)
    home, away = team == HOME, team == AWAY
    f = {}

    # Turn the pitch so that home attacks towards +x (teams sit, on average, nearer their own goal).
    with np.errstate(all="ignore"):
        flip = np.nanmean(players[:, home, 0]) > np.nanmean(players[:, away, 0])
    if flip:
        ball, players = ball.copy(), players.copy()
        ball[:, 0] = length - ball[:, 0]
        players[:, :, 0] = length - players[:, :, 0]

    vis = ~np.isnan(ball[:, 0])
    filled = _bridge(ball, vis, int(BRIDGE_S * fps))
    known = ~np.isnan(filled[:, 0])
    seconds = 1.0 / fps

    # -- ball and nearest players ---------------------------------------------------------------
    dx = players[:, :, 0] - filled[:, None, 0]
    dy = players[:, :, 1] - filled[:, None, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    dist[np.isnan(dist)] = np.inf
    nearest_slot = dist.argmin(axis=1)

    def three(columns):
        part = np.sort(dist[:, columns], axis=1)[:, :3]
        if part.shape[1] < 3:
            part = np.concatenate([part, np.full((n, 3 - part.shape[1]), np.inf, np.float32)], axis=1)
        return part

    d_home, d_away = three(home), three(away)
    dh, da = d_home[:, 0], d_away[:, 0]
    near = np.minimum(dh, da)
    somebody = known & np.isfinite(near)

    def shown(values, limit):
        return np.where(known, np.minimum(values, limit), np.nan)

    f["near"] = shown(near, 30.0)
    with np.errstate(invalid="ignore"):
        f["diff"] = np.where(somebody, np.clip(da - dh, -20.0, 20.0), np.nan)
    for k in range(3):
        f["home{}".format(k + 1)] = shown(d_home[:, k], 40.0)
        f["away{}".format(k + 1)] = shown(d_away[:, k], 40.0)

    # +1 home nearer, -1 away nearer, 0 unknown or an exact tie (so swapping teams flips the sign)
    sign = np.where(somebody, (dh < da).astype(np.float64) - (da < dh), 0.0)
    control = sign * (near < CONTROL_M)
    tight = sign * (near < TIGHT_M)
    for w_s in WINDOWS_S:
        w = max(1, int(round(w_s * fps)))
        f["sign_past_{}".format(w_s)] = _past(sign, w)
        f["sign_next_{}".format(w_s)] = _future(sign, w)
        f["ctrl_past_{}".format(w_s)] = _past(control, w)
        f["ctrl_next_{}".format(w_s)] = _future(control, w)
    for w_s in (1.0, 4.0):
        w = int(round(w_s * fps))
        f["tight_past_{}".format(w_s)] = _past(tight, w)
        f["tight_next_{}".format(w_s)] = _future(tight, w)

    limit = 30.0
    home_since = np.minimum(_since(control > 0) * seconds, limit)
    away_since = np.minimum(_since(control < 0) * seconds, limit)
    home_until = np.minimum(_until(control > 0) * seconds, limit)
    away_until = np.minimum(_until(control < 0) * seconds, limit)
    f["home_since"], f["away_since"] = home_since, away_since
    f["home_until"], f["away_until"] = home_until, away_until
    f["since_diff"] = away_since - home_since       # > 0: home touched it more recently
    f["until_diff"] = away_until - home_until       # > 0: home touches it sooner

    # -- how the ball moves and where it is -----------------------------------------------------
    ball_v = _velocity(filled, fps)
    f["ball_speed"] = np.minimum(np.sqrt((ball_v * ball_v).sum(axis=1)), 40.0)
    kick = np.full(n, np.nan, dtype=np.float32)      # change of velocity over 4 frames
    step = ball_v[4:] - ball_v[:-4]
    kick[2:-2] = np.sqrt((step * step).sum(axis=1))
    f["ball_kick"] = np.minimum(kick, 40.0)
    # The delivered ball track is heavily smoothed, so a pass shows up as a velocity change spread
    # over about a second. Compare the velocity shortly before with the velocity shortly after:
    # 0.2-0.6 s either side of the frame.
    swing = np.full(n, np.nan, dtype=np.float32)
    if n > 30:
        after = (filled[30:] - filled[20:-10]) * (fps / 10.0)       # frames t+5 .. t+15
        before = (filled[10:-20] - filled[:-30]) * (fps / 10.0)     # frames t-15 .. t-5
        change = after - before
        swing[15:-15] = np.sqrt((change * change).sum(axis=1))
    f["ball_swing"] = np.minimum(swing, 40.0)
    # A track bridged by the tracker is a straight line at constant speed: no kick at all for
    # seconds. The average kick around the frame tells the model when it is looking at one.
    f["kick_around_2"] = _nan_mean(np.asarray(kick, dtype=np.float64), -fps, fps + 1)

    # A touch: the ball changes velocity while a player is close, so that player played it.
    # Two touches in a row by one team is what "control" looks like in tracking data.
    with np.errstate(invalid="ignore"):
        touch = somebody & (swing > TOUCH_SWING) & (near < CONTROL_M)
    touch_home, touch_away = touch & (sign > 0), touch & (sign < 0)
    th_since = np.minimum(_since(touch_home) * seconds, limit)
    ta_since = np.minimum(_since(touch_away) * seconds, limit)
    th_until = np.minimum(_until(touch_home) * seconds, limit)
    ta_until = np.minimum(_until(touch_away) * seconds, limit)
    f["home_touch_since"], f["away_touch_since"] = th_since, ta_since
    f["home_touch_until"], f["away_touch_until"] = th_until, ta_until
    f["touch_since_diff"] = ta_since - th_since
    f["touch_until_diff"] = ta_until - th_until
    touches = touch_home.astype(np.float64) - touch_away
    for w_s in (2.0, 6.0):
        w = int(round(w_s * fps))
        f["touch_past_{}".format(w_s)] = _past(touches, w)
        f["touch_next_{}".format(w_s)] = _future(touches, w)
    player_v = _velocity(players, fps)
    nearest_v = player_v[index, nearest_slot]
    relative = ball_v - nearest_v
    with np.errstate(invalid="ignore"):
        f["near_rel_speed"] = np.where(somebody, np.minimum(np.sqrt((relative * relative).sum(axis=1)), 30.0), np.nan)
        f["near_speed"] = np.where(somebody, np.minimum(np.sqrt((nearest_v * nearest_v).sum(axis=1)), 12.0), np.nan)

    bx, by = filled[:, 0], filled[:, 1]
    edge = geometry.edge_distance(filled, pitch)
    f["ball_x"] = np.clip(bx, -10.0, length + 10.0)
    f["ball_mid_x"] = np.minimum(np.abs(bx - length / 2), 62.0)
    f["ball_mid_y"] = np.minimum(np.abs(by - width / 2), 44.0)
    f["ball_edge"] = np.clip(edge, -10.0, 34.0)
    f["ball_centre"] = np.minimum(np.sqrt((bx - length / 2) ** 2 + (by - width / 2) ** 2), 70.0)
    corner_x = np.minimum(np.abs(bx), np.abs(bx - length))
    corner_y = np.minimum(np.abs(by), np.abs(by - width))
    f["ball_corner"] = np.minimum(np.sqrt(corner_x ** 2 + corner_y ** 2), 70.0)
    f["ball_goal"] = np.minimum(np.sqrt(corner_x ** 2 + (by - width / 2) ** 2), 70.0)

    # -- is the ball seen, and for how long has it (not) been --------------------------------------
    change = np.flatnonzero(vis[1:] != vis[:-1]) + 1
    bounds = np.concatenate(([0], change, [n]))
    lengths = np.diff(bounds).astype(np.float64)
    run = np.repeat(np.arange(len(lengths)), np.diff(bounds))
    f["vis"] = vis.astype(np.float32)
    f["run_len"] = np.minimum(lengths[run] * seconds, CLIP_S)
    f["run_into"] = np.minimum((index - bounds[run]) * seconds, CLIP_S)
    f["run_left"] = np.minimum((bounds[run + 1] - 1 - index) * seconds, CLIP_S)
    f["run_prev"] = np.minimum(np.concatenate(([np.nan], lengths[:-1]))[run] * seconds, CLIP_S)
    f["run_next"] = np.minimum(np.concatenate((lengths[1:], [np.nan]))[run] * seconds, CLIP_S)

    # For frames inside a gap: where and how fast the ball was when lost, where it came back.
    raw_edge = geometry.edge_distance(ball, pitch)
    back = np.full(n, np.nan, dtype=np.float32)
    step = ball[3:] - ball[:-3]
    back[3:] = np.sqrt((step * step).sum(axis=1)) * (fps / 3.0)
    before, after = bounds[:-1] - 1, bounds[1:]
    is_gap = ~vis[bounds[:-1]]
    has_before, has_after = is_gap & (before >= 0), is_gap & (after < n)
    b_idx, a_idx = np.maximum(before, 0), np.minimum(after, n - 1)

    def per_gap(values, valid):
        return np.where(valid, values, np.nan)[run]

    f["gap_edge_before"] = np.clip(per_gap(raw_edge[b_idx], has_before), -10.0, 34.0)
    f["gap_edge_after"] = np.clip(per_gap(raw_edge[a_idx], has_after), -10.0, 34.0)
    f["gap_speed_before"] = np.minimum(per_gap(back[b_idx], has_before), 40.0)
    jump = ball[a_idx] - ball[b_idx]
    jump = np.sqrt((jump * jump).sum(axis=1))
    f["gap_jump"] = np.minimum(per_gap(jump, has_before & has_after), 110.0)
    with np.errstate(all="ignore"):
        f["gap_implied_speed"] = np.minimum(per_gap(jump / (lengths * seconds), has_before & has_after), 40.0)

    seen = vis.astype(np.float64)
    for w_s in (2.0, 10.0):
        w = int(round(w_s * fps))
        f["vis_past_{}".format(w_s)] = _past(seen, w)
        f["vis_next_{}".format(w_s)] = _future(seen, w)
    f["vis_around_60"] = _nan_mean(seen, -30 * fps, 30 * fps + 1)

    # -- what the players are doing (needs no ball) ------------------------------------------------
    speed = geometry.speed_of(players, fps, half_span=5)
    speed[speed > MAX_PLAYER_SPEED] = np.nan
    count = (~np.isnan(speed)).sum(axis=1)
    ordered = np.sort(speed, axis=1)                         # NaN sorts last
    rows = index

    def pick(position):
        return ordered[rows, np.clip(position, 0, speed.shape[1] - 1)]

    with np.errstate(invalid="ignore"):
        median = np.where(count >= 6, 0.5 * (pick((count - 1) // 2) + pick(count // 2)), np.nan)
        fast = np.where(count >= 6, pick((0.9 * (count - 1)).astype(np.int64)), np.nan)
        walkers = np.where(count >= 6, (speed < WALK_SPEED).sum(axis=1) / np.maximum(count, 1), np.nan)
    half = fps // 2
    tempo = _nan_mean(median, -half, half + 1)
    f["tempo"] = tempo
    f["tempo_past_3"] = _nan_mean(median, -3 * fps, 1)
    f["tempo_next_3"] = _nan_mean(median, 1, 3 * fps + 1)
    f["tempo_around_10"] = _nan_mean(median, -5 * fps, 5 * fps + 1)
    f["tempo_around_30"] = _nan_mean(median, -15 * fps, 15 * fps + 1)
    f["speed_p90"] = _nan_mean(fast, -half, half + 1)
    f["walkers"] = _nan_mean(walkers, -half, half + 1)

    shape = {}
    for side, columns in (("home", home), ("away", away)):
        x, y = players[:, columns, 0], players[:, columns, 1]
        present = ~np.isnan(x)
        k = present.sum(axis=1)
        with np.errstate(all="ignore"):
            cx, cy = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
            sx, sy = np.nanstd(x, axis=1), np.nanstd(y, axis=1)
            own_half = ((x < length / 2) if side == "home" else (x > length / 2)).sum(axis=1) / np.maximum(k, 1)
        own_half = np.where(k > 0, own_half, np.nan)
        # second-deepest player = the defensive line (the deepest is normally the goalkeeper)
        depth = np.sort(x if side == "home" else -x, axis=1)
        line = depth[:, 1] if depth.shape[1] > 1 else np.full(n, np.nan)
        line = line if side == "home" else -line
        shape[side] = (cx, cy, sx, sy, line)
        f[side + "_n"] = k.astype(np.float32)
        f[side + "_cx"], f[side + "_sx"], f[side + "_sy"] = cx, sx, sy
        f[side + "_line"] = line
        f[side + "_own_half"] = own_half
        drift = np.full(n, np.nan)
        drift[12:-12] = (cx[24:] - cx[:-24]) * (fps / 24.0)
        f[side + "_drift"] = np.clip(drift, -10.0, 10.0)
        f["ball_from_" + side + "_cx"] = bx - cx
        f["ball_from_" + side + "_line"] = bx - line
    f["cx_gap"] = shape["away"][0] - shape["home"][0]
    f["sx_diff"] = shape["home"][2] - shape["away"][2]
    f["sy_diff"] = shape["home"][3] - shape["away"][3]
    f["ball_from_cy"] = np.abs(by - 0.5 * (shape["home"][1] + shape["away"][1]))

    to_touchline = np.minimum(players[:, :, 1], width - players[:, :, 1])
    with np.errstate(all="ignore"):
        f["touchline_min"] = np.clip(np.nanmin(to_touchline, axis=1), -5.0, 34.0)
        f["touchline_n"] = (to_touchline < 1.0).sum(axis=1).astype(np.float32)

    return {name: np.round(np.asarray(values, dtype=np.float64), 3).astype(np.float32)
            for name, values in f.items()}


def compute(tracking):
    """(X, names): X is float32 (n_frames, n_features); rows outside the halves are NaN."""
    ball, players = geometry.to_metres(tracking)
    out, names = None, None
    for start, stop in tracking.periods:
        with warnings.catch_warnings():
            # frames with no tracked player at all give "mean of empty slice": NaN is what we want
            warnings.simplefilter("ignore", category=RuntimeWarning)
            columns = _period(ball[start:stop], players[start:stop], tracking.team,
                              tracking.pitch, tracking.fps)
        if out is None:
            names = list(columns)
            out = np.full((tracking.n_frames, len(names)), np.nan, dtype=np.float32)
        out[start:stop] = np.stack([columns[name] for name in names], axis=1)
    return out, names


def swap_teams(tracking):
    """The same recording with home and away exchanged (used to make the model symmetric)."""
    return tracking.replace(team=(1 - tracking.team).astype(tracking.team.dtype),
                            team_names=tracking.team_names[::-1])
