"""Summary statistics of a Tracking: how often and how the ball and the players go missing.

Used for two things:
  * scripts/profile_atd.py prints it for the competition match, so the training simulator can be
    made to fail the same way the real tracking fails;
  * training code computes it for simulated tracking, to compare the two.

Everything here is an aggregate over thousands of frames (counts, shares, medians). It never
returns or prints a frame, a position, or anything that could rebuild the tracking.
"""

import numpy as np

from . import baseline, geometry, smart_tagging
from .tracking import AWAY, HOME

GAP_BINS_S = (0, 0.2, 0.5, 1, 2, 3, 5, 10, 20, 40, 90, np.inf)       # ball gaps / visible runs
STILL_BINS_S = (1, 2, 5, 10, 20, np.inf)                              # stationary-ball runs
NEAR_BINS_M = (0, 0.5, 1, 2, 4, 8, np.inf)                            # ball to nearest player
SPEED_BINS = (0, 0.5, 2, 5, 10, 15, 22, np.inf)                       # ball speed, m/s
EDGE_BINS_M = (-np.inf, -3, -1, 0, 1, 2, 5, 15, np.inf)               # ball to nearest line
TEMPO_BINS = (0, 0.6, 1.0, 1.4, 1.8, 2.4, 3.2, np.inf)                # median player speed, m/s
PERCENTILES = (5, 25, 50, 75, 95)
STILL_SPEED = 0.5      # m/s: below this the ball counts as stationary
MAX_PLAYER_SPEED = 12  # m/s: faster "movement" is a tracking jump, not a player


def _labels(bins, unit=""):
    out = []
    for low, high in zip(bins[:-1], bins[1:]):
        if np.isinf(high):
            out.append("{:g}+{}".format(low, unit))
        elif np.isinf(low):
            out.append("<{:g}{}".format(high, unit))
        else:
            out.append("{:g}-{:g}{}".format(low, high, unit))
    return out


def _pct(values, percentiles=PERCENTILES):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [float("nan")] * len(percentiles)
    return [float(v) for v in np.percentile(values, percentiles)]


def _median(values):
    return _pct(values, (50,))[0]


def _share(mask):
    mask = np.asarray(mask)
    return float(mask.mean()) if mask.size else float("nan")


def tempo_of(players, fps):
    """(n,) median speed (m/s) of the tracked players, averaged over 1 s. NaN if too few players.

    Low tempo = everybody walking = a stoppage is likely. It needs no ball, which makes it the
    one signal that still works when the ball is missing.
    """
    speed = geometry.speed_of(players, fps, half_span=5)
    speed[speed > MAX_PLAYER_SPEED] = np.nan
    enough = (~np.isnan(speed)).sum(axis=1) >= 6
    tempo = np.full(len(players), np.nan, dtype=np.float32)
    with np.errstate(all="ignore"):
        tempo[enough] = np.nanmedian(speed[enough], axis=1)
    # centred moving average that ignores NaN
    good = ~np.isnan(tempo)
    total = np.concatenate(([0.0], np.cumsum(np.where(good, tempo, 0.0))))
    count = np.concatenate(([0], np.cumsum(good)))
    index = np.arange(len(tempo))
    low, high = np.maximum(0, index - fps // 2), np.minimum(len(tempo), index + fps // 2 + 1)
    n = count[high] - count[low]
    with np.errstate(all="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan).astype(np.float32)


class _Period:
    """Per-frame context for one half (all arrays are views or copies of that half only)."""

    def __init__(self, tracking, start, stop):
        fps = tracking.fps
        ball, players = geometry.to_metres(tracking)
        self.fps = fps
        self.ball = ball[start:stop]
        self.players = players[start:stop]
        self.vis = ~np.isnan(self.ball).any(axis=1)
        distances = geometry.team_distances(self.ball, self.players, tracking.team, k=1)
        self.d_home, self.d_away = distances[:, 0, 0], distances[:, 1, 0]
        self.near = np.minimum(self.d_home, self.d_away)
        self.edge = geometry.edge_distance(self.ball, tracking.pitch)
        self.speed = geometry.speed_of(self.ball, fps, half_span=2)
        back = np.full(len(self.ball), np.nan, dtype=np.float32)   # speed over the last 3 frames
        step = self.ball[3:] - self.ball[:-3]
        back[3:] = np.sqrt((step * step).sum(axis=1)) * (fps / 3.0)
        self.speed_back = back
        self.tempo = tempo_of(self.players, fps)


def _histogram(lengths, bins, fps, n_frames):
    """[(count, share of frames)] of run lengths (in frames) over bins given in seconds."""
    seconds = np.asarray(lengths) / float(fps)
    out = []
    for low, high in zip(bins[:-1], bins[1:]):
        inside = (seconds >= low) & (seconds < high)
        out.append((int(inside.sum()), float(np.asarray(lengths)[inside].sum() / max(n_frames, 1))))
    return out


def _hazard(values, exposed, lost, bins, fps):
    """[(seconds of exposure, losses, losses per second)] by bin of a context value."""
    out = []
    for low, high in zip(bins[:-1], bins[1:]):
        inside = exposed & (values >= low) & (values < high)
        seconds = inside.sum() / float(fps)
        losses = int((inside & lost).sum())
        out.append((float(seconds), losses, float(losses / seconds) if seconds > 0 else float("nan")))
    return out


def measure(tracking, tags=None):
    """Nested dict of aggregate statistics over the halves of a Tracking."""
    fps = tracking.fps
    periods = [_Period(tracking, start, stop) for start, stop in tracking.periods]
    n_graded = sum(len(p.vis) for p in periods)
    cat = lambda name: np.concatenate([getattr(p, name) for p in periods])  # noqa: E731
    vis, near, edge, speed, tempo = cat("vis"), cat("near"), cat("edge"), cat("speed"), cat("tempo")
    stats = {"n_graded": n_graded, "fps": fps, "n_slots": tracking.n_slots}

    # -- ball visibility -------------------------------------------------------------------------
    stats["missing_share"] = _share(~vis)
    stats["missing_share_per_half"] = [_share(~p.vis) for p in periods]
    gap_lengths, run_lengths = [], []
    gaps = {key: [] for key in ("length", "edge_before", "edge_after", "near_before",
                                "near_after", "speed_before", "jump", "tempo")}
    exposed, lost = [], []
    for p in periods:
        starts, lengths = geometry.runs_of(~p.vis)
        gap_lengths.append(lengths)
        run_lengths.append(geometry.runs_of(p.vis)[1])
        # a visible frame is "exposed" if the next frame is in the same half; "lost" if the ball
        # is gone on that next frame
        e = p.vis.copy()
        e[-1] = False
        l = np.zeros(len(p.vis), dtype=bool)
        l[:-1] = p.vis[:-1] & ~p.vis[1:]
        exposed.append(e)
        lost.append(l)
        for s, n in zip(starts, lengths):
            if s == 0 or s + n >= len(p.vis):
                continue                     # touches the edge of the half: no before / after
            before, after = s - 1, s + n
            gaps["length"].append(n)
            gaps["edge_before"].append(p.edge[before])
            gaps["edge_after"].append(p.edge[after])
            gaps["near_before"].append(p.near[before])
            gaps["near_after"].append(p.near[after])
            gaps["speed_before"].append(p.speed_back[before])
            step = p.ball[after] - p.ball[before]
            gaps["jump"].append(float(np.sqrt((step * step).sum())))
            window = p.tempo[s:s + n]
            gaps["tempo"].append(float(np.nanmean(window)) if np.isfinite(window).any()
                                 else float("nan"))
    gap_lengths, run_lengths = np.concatenate(gap_lengths), np.concatenate(run_lengths)
    exposed, lost = np.concatenate(exposed), np.concatenate(lost)
    stats["gap_hist"] = _histogram(gap_lengths, GAP_BINS_S, fps, n_graded)
    stats["run_hist"] = _histogram(run_lengths, GAP_BINS_S, fps, n_graded)

    stats["loss_per_s"] = float(lost.sum() / max(exposed.sum() / fps, 1e-9))
    speed_back = cat("speed_back")
    stats["hazard_near"] = _hazard(near, exposed, lost, NEAR_BINS_M, fps)
    stats["hazard_speed"] = _hazard(speed_back, exposed, lost, SPEED_BINS, fps)
    stats["hazard_edge"] = _hazard(edge, exposed, lost, EDGE_BINS_M, fps)
    stats["hazard_tempo"] = _hazard(tempo, exposed, lost, TEMPO_BINS, fps)
    # by zone of the pitch: thirds along x (rows) by thirds along y (columns)
    ball_all = cat("ball")
    zone_x = np.clip(np.nan_to_num(ball_all[:, 0] / tracking.pitch[0] * 3), 0, 2).astype(np.int64)
    zone_y = np.clip(np.nan_to_num(ball_all[:, 1] / tracking.pitch[1] * 3), 0, 2).astype(np.int64)
    stats["hazard_zone"] = [[_hazard(np.zeros(len(vis)), exposed & (zone_x == i) & (zone_y == j),
                                     lost, (-1, 1), fps)[0] for j in range(3)] for i in range(3)]

    gaps = {key: np.asarray(value, dtype=np.float64) for key, value in gaps.items()}
    seconds = gaps["length"] / fps
    rows = []
    for low, high in zip(GAP_BINS_S[:-1], GAP_BINS_S[1:]):
        inside = (seconds >= low) & (seconds < high)
        g = {key: value[inside] for key, value in gaps.items()}
        with np.errstate(all="ignore"):
            implied = g["jump"] / (g["length"] / fps)
        rows.append({
            "n": int(inside.sum()),
            "edge_before_lt2": _share(g["edge_before"] < 2), "edge_after_lt2": _share(g["edge_after"] < 2),
            "near_before": _median(g["near_before"]), "near_after": _median(g["near_after"]),
            "speed_before": _median(g["speed_before"]), "jump": _median(g["jump"]),
            "implied_speed": _median(implied), "jump_gt20": _share(g["jump"] > 20),
            "tempo": _pct(g["tempo"], (25, 50, 75)),
            "tempo_hist": [int(((g["tempo"] >= a) & (g["tempo"] < b)).sum())
                           for a, b in zip(TEMPO_BINS[:-1], TEMPO_BINS[1:])],
        })
    stats["gap_context"] = rows
    stats["tempo_all"] = _pct(tempo)
    stats["tempo_visible"] = _pct(tempo[vis])
    stats["tempo_missing"] = _pct(tempo[~vis])
    stats["tempo_hist_visible"] = [_share((tempo[vis] >= a) & (tempo[vis] < b))
                                   for a, b in zip(TEMPO_BINS[:-1], TEMPO_BINS[1:])]
    stats["tempo_hist_missing"] = [_share((tempo[~vis] >= a) & (tempo[~vis] < b))
                                   for a, b in zip(TEMPO_BINS[:-1], TEMPO_BINS[1:])]

    # -- where the ball is, and how it moves, when it is seen ------------------------------------
    ball = cat("ball")[vis]
    length, width = tracking.pitch
    stats["ball_outside"] = {
        "any": _share(edge[vis] < 0), "x_low": _share(ball[:, 0] < 0),
        "x_high": _share(ball[:, 0] > length), "y_low": _share(ball[:, 1] < 0),
        "y_high": _share(ball[:, 1] > width),
        "depth_m": _pct(-edge[vis][edge[vis] < 0], (50, 90, 99)),
    }
    stats["ball_edge_hist"] = [_share((edge[vis] >= a) & (edge[vis] < b))
                               for a, b in zip(EDGE_BINS_M[:-1], EDGE_BINS_M[1:])]
    stats["ball_speed"] = _pct(speed[vis], (10, 25, 50, 75, 90, 99, 99.9))
    stats["ball_speed_hist"] = [_share((speed[vis] >= a) & (speed[vis] < b))
                                for a, b in zip(SPEED_BINS[:-1], SPEED_BINS[1:])]
    stats["ball_step_gt2m"] = float(np.mean([
        _share(np.sqrt((np.diff(p.ball, axis=0) ** 2).sum(axis=1))[p.vis[1:] & p.vis[:-1]] > 2.0)
        for p in periods]))
    jitter = []
    for p in periods:
        second = p.ball[2:] - 2 * p.ball[1:-1] + p.ball[:-2]
        jitter.append(np.sqrt((second * second).sum(axis=1)))
    stats["ball_jitter_m"] = _median(np.concatenate(jitter))

    still_rows = [[0, 0.0, [], []] for _ in STILL_BINS_S[:-1]]
    for p in periods:
        with np.errstate(invalid="ignore"):
            still = p.vis & (p.speed < STILL_SPEED)
        for s, n in zip(*geometry.runs_of(still)):
            sec = n / fps
            for row, (low, high) in zip(still_rows, zip(STILL_BINS_S[:-1], STILL_BINS_S[1:])):
                if low <= sec < high:
                    row[0] += 1
                    row[1] += n / n_graded
                    row[2].append(p.edge[s])
                    row[3].append(np.nanmean(p.tempo[s:s + n]) if np.isfinite(p.tempo[s:s + n]).any()
                                  else np.nan)
    stats["still_runs"] = [{"n": row[0], "frame_share": row[1],
                            "edge_lt2": _share(np.asarray(row[2]) < 2),
                            "tempo": _median(row[3])} for row in still_rows]

    # -- ball and players together ---------------------------------------------------------------
    stats["nearest_m"] = _pct(near[vis], (10, 25, 50, 75, 90, 99))
    d_home, d_away = cat("d_home")[vis], cat("d_away")[vis]
    both = np.isfinite(d_home) & np.isfinite(d_away)
    stats["margin_m"] = _pct(np.abs(d_home - d_away)[both], (10, 25, 50, 75, 90))
    flips, visible_s = 0, 0.0
    for p in periods:
        nearest = np.where(p.d_home <= p.d_away, HOME, AWAY)[p.vis & np.isfinite(p.near)]
        flips += int((np.diff(nearest) != 0).sum())
        visible_s += len(nearest) / fps
    stats["nearest_team_flips_per_min"] = float(flips / max(visible_s / 60.0, 1e-9))

    # -- players ---------------------------------------------------------------------------------
    players = cat("players")
    seen = ~np.isnan(players[:, :, 0])
    home, away = seen[:, tracking.team == HOME].sum(axis=1), seen[:, tracking.team == AWAY].sum(axis=1)
    stats["players_home"] = _pct(home)
    stats["players_away"] = _pct(away)
    stats["players_mean"] = [float(home.mean()), float(away.mean())]
    total = home + away
    stats["players_total_share"] = {"<16": _share(total < 16), "16-19": _share((total >= 16) & (total < 20)),
                                    "20-22": _share((total >= 20) & (total <= 22)), ">22": _share(total > 22),
                                    "home<8": _share(home < 8), "away<8": _share(away < 8),
                                    "home>11": _share(home > 11), "away>11": _share(away > 11)}
    tracklets, steps, second_diffs = [], [], []
    for p in periods:
        visible = ~np.isnan(p.players[:, :, 0])
        for slot in range(visible.shape[1]):
            tracklets.append(geometry.runs_of(visible[:, slot])[1])
        step = np.sqrt((np.diff(p.players, axis=0) ** 2).sum(axis=2))
        steps.append(step[np.isfinite(step)])
        second = p.players[2:] - 2 * p.players[1:-1] + p.players[:-2]
        second = np.sqrt((second * second).sum(axis=2))
        second_diffs.append(second[np.isfinite(second)])
    tracklets = np.concatenate(tracklets) if tracklets else np.zeros(0)
    steps, second_diffs = np.concatenate(steps), np.concatenate(second_diffs)
    stats["tracklets"] = {"n": int(len(tracklets)), "seconds": _pct(tracklets / fps, (10, 25, 50, 75, 90)),
                          "frame_share_under_2s": float(tracklets[tracklets < 2 * fps].sum()
                                                        / max(tracklets.sum(), 1))}
    stats["player_step_gt_half_m"] = _share(steps > 0.5)
    stats["player_jitter_m"] = _median(second_diffs)
    player_edge = geometry.edge_distance(players, tracking.pitch)
    stats["player_outside"] = _share(player_edge[seen] < 0)
    stats["player_speed_fd"] = _pct(np.concatenate([
        geometry.speed_of(p.players, fps, 5)[~np.isnan(p.players[:, :, 0])] for p in periods]),
        (50, 90, 99))
    if tracking.speed is not None:
        channel = np.concatenate([tracking.speed[start:stop] for start, stop in tracking.periods])
        finite = np.concatenate([geometry.speed_of(p.players, fps, 2) for p in periods])
        have = seen & np.isfinite(channel)
        stats["speed_channel"] = {"coverage": float(have.sum() / max(seen.sum(), 1)),
                                  "percentiles": _pct(channel[have], (50, 90, 99, 100)),
                                  "abs_diff_to_positions": _median(np.abs(channel - finite)[have])}
    sides = []
    for p in periods:
        x = p.players[:, :, 0] / length
        with np.errstate(all="ignore"):
            sides.append([float(np.nanmean(x[:, tracking.team == HOME])),
                          float(np.nanmean(x[:, tracking.team == AWAY]))])
    stats["team_mean_x"] = sides

    if tags is not None:
        stats["tags"] = _measure_tags(tracking, tags, periods)
    return stats


def _measure_tags(tracking, tags, periods):
    """How the Smart Tagging intervals line up with the tracking (aggregates per code / type)."""
    fps = tracking.fps
    n = tracking.n_frames
    vis = np.zeros(n, dtype=bool)
    tempo = np.full(n, np.nan, dtype=np.float32)
    graded = np.zeros(n, dtype=bool)
    reappear = []                                    # frames where a >= 2 s ball gap ends
    for (start, stop), p in zip(tracking.periods, periods):
        vis[start:stop], tempo[start:stop], graded[start:stop] = p.vis, p.tempo, True
        starts, lengths = geometry.runs_of(~p.vis)
        reappear.extend(start + s + m for s, m in zip(starts, lengths) if m >= 2 * fps)
    reappear = np.array(sorted(reappear))
    nearest = baseline.nearest_team(tracking)

    def window(a, b):
        a, b = max(0, a), min(n, b)
        return slice(a, max(a, b))

    out = {"by_code": [], "by_kind": []}
    covered = np.zeros(n, dtype=bool)
    for code in sorted({tag.code for tag in tags}):
        group = [tag for tag in tags if tag.code == code]
        agree = total = frames = 0
        inside = np.zeros(n, dtype=bool)
        for tag in group:
            inside[window(tag.start, tag.stop)] = True
            if tag.team in (HOME, AWAY):
                part = nearest[window(tag.start, tag.stop)]
                part = part[part != baseline.NOBODY]
                agree += int((part == tag.team).sum())
                total += len(part)
        inside &= graded
        if code != smart_tagging.SET_PIECES:
            covered |= inside
        out["by_code"].append({
            "code": code, "n": len(group),
            "with_team": sum(tag.team in (HOME, AWAY) for tag in group),
            "seconds": _pct([(tag.stop - tag.start) / fps for tag in group], (10, 50, 90)),
            "graded_share": float(inside.sum() / max(graded.sum(), 1)),
            "ball_visible": _share(vis[inside]),
            "nearest_is_tag_team": float(agree / total) if total else float("nan"),
        })
    out["phase_coverage"] = float(covered.sum() / max(graded.sum(), 1))

    ten = 10 * fps
    for kind in sorted({tag.kind for tag in tags if tag.code == smart_tagging.SET_PIECES}):
        group = [tag for tag in tags if tag.code == smart_tagging.SET_PIECES and tag.kind == kind]
        before = [window(tag.start - ten, tag.start) for tag in group]
        during = [window(tag.start, tag.stop) for tag in group]
        after = [window(tag.stop, tag.stop + ten) for tag in group]
        offsets = []
        for tag in group:
            if len(reappear):
                nearest_end = reappear[np.argmin(np.abs(reappear - tag.start))]
                if abs(nearest_end - tag.start) <= 15 * fps:
                    offsets.append((nearest_end - tag.start) / fps)
        out["by_kind"].append({
            "kind": kind, "n": len(group),
            "with_team": sum(tag.team in (HOME, AWAY) for tag in group),
            "seconds": _pct([(tag.stop - tag.start) / fps for tag in group], (10, 50, 90)),
            "ball_visible": [_share(np.concatenate([vis[w] for w in part])) for part in (before, during, after)],
            "tempo": [_median(np.concatenate([tempo[w] for w in part])) for part in (before, during, after)],
            "reappear_n": len(offsets), "reappear_offset_s": _pct(offsets, (25, 50, 75)),
        })
    return out


def measure_window(tracking, start, stop):
    """Ball visibility, tempo and player counts over one stretch of frames (aggregates)."""
    part = tracking.replace(ball=tracking.ball[start:stop], players=tracking.players[start:stop],
                            periods=[(0, stop - start)])
    _, players = geometry.to_metres(part)
    tempo = tempo_of(players, tracking.fps)
    seen = (~np.isnan(players[:, :, 0])).sum(axis=1)
    vis = part.ball_visible()
    return {"seconds": (stop - start) / tracking.fps, "ball_visible": _share(vis),
            "visible_runs_s": _pct(geometry.runs_of(vis)[1] / tracking.fps, (50, 90)),
            "tempo": _pct(tempo, (25, 50, 75)), "players": _pct(seen, (25, 50, 75))}


def measure_camera(tracking, matrices, valid, image=(1920, 1080)):
    """Does the tactical camera follow the play? Aggregates from the per-frame homography.

    The homography maps pitch (0..1) to pixels, so its inverse tells which point of the pitch is
    at the centre of the image ("focus") and how wide a stretch of pitch the image shows. If the
    focus moves with the ball, it says where the action is even when the ball is not detected.
    """
    fps = tracking.fps
    length, width = tracking.pitch
    graded = tracking.in_periods() & valid
    inverse = np.full_like(matrices, np.nan)
    inverse[valid] = np.linalg.inv(matrices[valid])

    def to_pitch(px, py):
        point = inverse @ np.array([px, py, 1.0])
        with np.errstate(all="ignore"):
            return np.stack([point[:, 0] / point[:, 2] * length, point[:, 1] / point[:, 2] * width], axis=1)

    w, h = image
    focus = to_pitch(w / 2.0, h / 2.0)
    left, right = to_pitch(0.0, h / 2.0), to_pitch(float(w), h / 2.0)
    view = np.sqrt(((right - left) ** 2).sum(axis=1))
    ball, players = geometry.to_metres(tracking)
    vis = tracking.ball_visible()
    focus_speed = geometry.speed_of(focus.astype(np.float32), fps, half_span=5)

    def in_image(xy):
        x, y = xy[..., 0] / length, xy[..., 1] / width
        m = matrices.reshape(len(matrices), 1, 3, 3) if xy.ndim == 3 else matrices
        den = m[..., 2, 0] * x + m[..., 2, 1] * y + m[..., 2, 2]
        with np.errstate(all="ignore"):
            u = (m[..., 0, 0] * x + m[..., 0, 1] * y + m[..., 0, 2]) / den
            v = (m[..., 1, 0] * x + m[..., 1, 1] * y + m[..., 1, 2]) / den
            return (u >= 0) & (u <= w) & (v >= 0) & (v <= h) & (den > 0)

    stats = {"valid_share": _share(valid[tracking.in_periods()])}
    g = graded
    stats["focus_x"] = _pct(focus[g, 0], (5, 25, 50, 75, 95))
    stats["focus_y"] = _pct(focus[g, 1], (5, 25, 50, 75, 95))
    stats["view_width"] = _pct(view[g], (5, 25, 50, 75, 95))
    stats["focus_speed"] = _pct(focus_speed[g], (25, 50, 75, 95))
    stats["focus_speed_visible"] = _pct(focus_speed[g & vis], (25, 50, 75, 95))
    stats["focus_speed_missing"] = _pct(focus_speed[g & ~vis], (25, 50, 75, 95))
    stats["focus_static_share"] = [_share(focus_speed[g & vis] < 0.3), _share(focus_speed[g & ~vis] < 0.3)]
    both = g & vis
    offset = ball[both] - focus[both]
    stats["ball_to_focus"] = _pct(np.sqrt((offset ** 2).sum(axis=1)), (10, 25, 50, 75, 90))
    stats["ball_to_focus_x"] = _pct(np.abs(offset[:, 0]), (25, 50, 75, 90))
    stats["ball_to_focus_y"] = _pct(np.abs(offset[:, 1]), (25, 50, 75, 90))
    lags = []
    index = np.flatnonzero(both)
    for lag in (-50, -25, -12, 0, 12, 25, 50):
        other = index + lag
        ok = (other >= 0) & (other < len(focus))
        ok[ok] &= graded[other[ok]]
        lags.append((lag / fps, _median(np.abs(ball[index[ok], 0] - focus[other[ok], 0]))))
    stats["lag_x"] = lags
    bx, fx = ball[both, 0], focus[both, 0]
    stats["corr_x"] = float(np.corrcoef(bx, fx)[0, 1]) if len(bx) > 2 else float("nan")
    stats["ball_in_image"] = _share(in_image(ball)[both])
    seen = ~np.isnan(players[:, :, 0])
    inside = in_image(players)
    stats["players_in_image"] = float((inside & seen)[g].sum() / max(seen[g].sum(), 1))
    # how much of the pitch the image shows: share of a 21 x 13 grid of pitch points inside it
    gx, gy = np.meshgrid(np.linspace(0, length, 21), np.linspace(0, width, 13))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    sample = np.flatnonzero(g)[::25]
    cover = []
    for frame in sample:
        m = matrices[frame]
        x, y = grid[:, 0] / length, grid[:, 1] / width
        den = m[2, 0] * x + m[2, 1] * y + m[2, 2]
        with np.errstate(all="ignore"):
            u = (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / den
            v = (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / den
        cover.append(float(((u >= 0) & (u <= w) & (v >= 0) & (v <= h) & (den > 0)).mean()))
    stats["pitch_shown"] = _pct(cover, (5, 25, 50, 75, 95))
    return stats


def render_camera(stats):
    lines = ["\n[11] tactical camera, from the homography (valid on {:.1%} of graded frames)".format(
        stats["valid_share"])]
    add = lines.append
    add("    pitch point at the image centre ('focus'), m p5/25/50/75/95:  x {}   y {}".format(
        _row(stats["focus_x"], "{:.1f}"), _row(stats["focus_y"], "{:.1f}")))
    add("    width of pitch shown at mid-height, m p5/25/50/75/95: {}".format(_row(stats["view_width"], "{:.1f}")))
    add("    share of the pitch inside the image p5/25/50/75/95: {}".format(_row(stats["pitch_shown"], "{:.0%}")))
    add("    focus speed m/s p25/50/75/95: all {}   ball visible {}   ball missing {}".format(
        _row(stats["focus_speed"], "{:.2f}"), _row(stats["focus_speed_visible"], "{:.2f}"),
        _row(stats["focus_speed_missing"], "{:.2f}")))
    add("    focus nearly still (< 0.3 m/s): {:.1%} of ball-visible frames, {:.1%} of ball-missing frames".format(
        *stats["focus_static_share"]))
    add("    ball to focus, m p10/25/50/75/90: {}   along x p25/50/75/90: {}   along y: {}".format(
        _row(stats["ball_to_focus"], "{:.1f}"), _row(stats["ball_to_focus_x"], "{:.1f}"),
        _row(stats["ball_to_focus_y"], "{:.1f}")))
    add("    correlation of ball x and focus x: {:.3f}".format(stats["corr_x"]))
    add("    median |ball x(t) - focus x(t + lag)|, m: " + "  ".join(
        "{:+.1f}s {:.1f}".format(lag, value) for lag, value in stats["lag_x"]))
    add("    ball inside the image: {:.1%} of visible frames   tracked players inside the image: {:.1%}".format(
        stats["ball_in_image"], stats["players_in_image"]))
    return "\n".join(lines)


# -- printing --------------------------------------------------------------------------------------

def _row(values, form="{:>7.1%}"):
    return " ".join(form.format(v) for v in values)


def render(stats):
    """The statistics as plain text. Aggregates only."""
    fps = stats["fps"]
    gap_names = _labels(GAP_BINS_S, "s")
    lines = []
    add = lines.append
    add("TRACKING PROFILE (aggregates only: no frames, no positions)")
    add("frames in the halves: {}   track slots: {}".format(stats["n_graded"], stats["n_slots"]))

    add("\n[1] ball missing: {:.1%} of frames   per half: {}".format(
        stats["missing_share"], _row(stats["missing_share_per_half"])))
    add("    ball lost {:.2f} times per second of visible ball".format(stats["loss_per_s"]))
    add("    {:<10} {:>7} {:>9}   | {:>9} {:>9}".format("length", "gaps", "frames", "vis.runs", "frames"))
    for name, (gn, gs), (rn, rs) in zip(gap_names, stats["gap_hist"], stats["run_hist"]):
        add("    {:<10} {:>7} {:>9.2%}   | {:>9} {:>9.2%}".format(name, gn, gs, rn, rs))

    add("\n[2] what a gap looks like, by gap length (medians; 'edge' = within 2 m of a line or off the pitch)")
    add("    {:<10} {:>6} {:>8} {:>8} {:>9} {:>9} {:>8} {:>7} {:>8} {:>7} {:>17}".format(
        "length", "gaps", "edge.bef", "edge.aft", "near.bef m", "near.aft m", "v.bef m/s", "jump m",
        "impl m/s", ">20m", "tempo q25/50/75"))
    for name, row in zip(gap_names, stats["gap_context"]):
        add("    {:<10} {:>6} {:>8.0%} {:>8.0%} {:>9.1f} {:>9.1f} {:>8.1f} {:>7.1f} {:>8.1f} {:>7.0%}   {:>4.1f} {:>4.1f} {:>4.1f}".format(
            name, row["n"], row["edge_before_lt2"], row["edge_after_lt2"], row["near_before"],
            row["near_after"], row["speed_before"], row["jump"], row["implied_speed"],
            row["jump_gt20"], *row["tempo"]))
    add("    gaps by tempo during the gap (median player speed, m/s): columns " + " | ".join(_labels(TEMPO_BINS)))
    for name, row in zip(gap_names, stats["gap_context"]):
        add("    {:<10} {}".format(name, " ".join("{:>6}".format(v) for v in row["tempo_hist"])))

    add("\n[3] tempo = median player speed (m/s), percentiles {}".format(PERCENTILES))
    add("    all frames      {}".format(_row(stats["tempo_all"], "{:>6.2f}")))
    add("    ball visible    {}".format(_row(stats["tempo_visible"], "{:>6.2f}")))
    add("    ball missing    {}".format(_row(stats["tempo_missing"], "{:>6.2f}")))
    add("    share of frames by tempo bin: " + " | ".join(_labels(TEMPO_BINS)))
    add("    ball visible    {}".format(_row(stats["tempo_hist_visible"])))
    add("    ball missing    {}".format(_row(stats["tempo_hist_missing"])))

    add("\n[4] how often the ball is lost (losses per second of visible ball), by what was true just before")
    for title, key, bins, unit in (("nearest player", "hazard_near", NEAR_BINS_M, "m"),
                                   ("ball speed", "hazard_speed", SPEED_BINS, "m/s"),
                                   ("distance to a line", "hazard_edge", EDGE_BINS_M, "m"),
                                   ("tempo", "hazard_tempo", TEMPO_BINS, "m/s")):
        add("    {:<19}{}".format(title, " ".join("{:>9}".format(v) for v in _labels(bins, unit))))
        add("    {:<19}{}".format("  seconds seen", " ".join("{:>9.0f}".format(r[0]) for r in stats[key])))
        add("    {:<19}{}".format("  losses per s", " ".join("{:>9.2f}".format(r[2]) for r in stats[key])))

    add("    by zone (rows: thirds along x; columns: thirds along y): seconds seen / losses per s")
    for row in stats["hazard_zone"]:
        add("      " + "   ".join("{:>6.0f} / {:.2f}".format(cell[0], cell[2]) for cell in row))

    outside = stats["ball_outside"]
    add("\n[5] the ball when it is seen")
    add("    off the pitch: {:.2%} of visible frames  (x<0 {:.2%}, x>1 {:.2%}, y<0 {:.2%}, y>1 {:.2%}); "
        "depth m p50/p90/p99: {}".format(outside["any"], outside["x_low"], outside["x_high"],
                                          outside["y_low"], outside["y_high"],
                                          _row(outside["depth_m"], "{:.1f}")))
    add("    distance to nearest line: " + " | ".join(_labels(EDGE_BINS_M, "m")))
    add("                              " + _row(stats["ball_edge_hist"]))
    add("    speed m/s p10/25/50/75/90/99/99.9: {}".format(_row(stats["ball_speed"], "{:.1f}")))
    add("    speed bins: " + " | ".join(_labels(SPEED_BINS)))
    add("                " + _row(stats["ball_speed_hist"]))
    add("    frame-to-frame steps over 2 m: {:.3%}   jitter (median 2nd difference): {:.3f} m".format(
        stats["ball_step_gt2m"], stats["ball_jitter_m"]))
    add("    stationary runs (speed < {} m/s):".format(STILL_SPEED))
    for name, row in zip(_labels(STILL_BINS_S, "s"), stats["still_runs"]):
        add("      {:<8} runs {:>5}  frames {:>6.2%}  at an edge {:>5.0%}  tempo {:>4.1f}".format(
            name, row["n"], row["frame_share"], row["edge_lt2"], row["tempo"]))

    add("\n[6] ball and players")
    add("    nearest player to ball, m p10/25/50/75/90/99: {}".format(_row(stats["nearest_m"], "{:.2f}")))
    add("    |home - away| nearest distance, m p10/25/50/75/90: {}".format(_row(stats["margin_m"], "{:.2f}")))
    add("    nearest team changes {:.1f} times per minute of visible ball".format(
        stats["nearest_team_flips_per_min"]))

    add("\n[7] players")
    add("    tracked per frame, percentiles {}: home {}   away {}   means {:.1f} / {:.1f}".format(
        PERCENTILES, _row(stats["players_home"], "{:.0f}"), _row(stats["players_away"], "{:.0f}"),
        *stats["players_mean"]))
    add("    share of frames: " + "  ".join("{} {:.1%}".format(k, v) for k, v in stats["players_total_share"].items()))
    t = stats["tracklets"]
    add("    tracklets: {}  length s p10/25/50/75/90: {}  frames in tracklets under 2 s: {:.1%}".format(
        t["n"], _row(t["seconds"], "{:.1f}"), t["frame_share_under_2s"]))
    add("    steps over 0.5 m in one frame: {:.3%}   jitter: {:.3f} m   off the pitch: {:.2%}".format(
        stats["player_step_gt_half_m"], stats["player_jitter_m"], stats["player_outside"]))
    add("    speed from positions, m/s p50/90/99: {}".format(_row(stats["player_speed_fd"], "{:.1f}")))
    if "speed_channel" in stats:
        c = stats["speed_channel"]
        add("    speed channel: on {:.1%} of tracked players; p50/90/99/max: {}; median |channel - "
            "from positions|: {:.2f}".format(c["coverage"], _row(c["percentiles"], "{:.1f}"),
                                             c["abs_diff_to_positions"]))
    for half, (home_x, away_x) in enumerate(stats["team_mean_x"], start=1):
        add("    half {}: mean x of home players {:.3f}, of away players {:.3f}".format(half, home_x, away_x))

    if "tags" in stats:
        tags = stats["tags"]
        add("\n[8] Smart Tagging vs tracking   (team-tagged phases cover {:.1%} of graded frames)".format(
            tags["phase_coverage"]))
        add("    {:<30} {:>4} {:>5} {:>17} {:>8} {:>9} {:>14}".format(
            "code", "n", "team", "seconds p10/50/90", "graded", "ball vis", "nearest=tagged"))
        for row in tags["by_code"]:
            add("    {:<30} {:>4} {:>5}   {:>5.1f} {:>5.1f} {:>5.1f} {:>8.1%} {:>9.1%} {:>14.1%}".format(
                row["code"], row["n"], row["with_team"], *row["seconds"], row["graded_share"],
                row["ball_visible"], row["nearest_is_tag_team"]))
        add("    set pieces by type: ball visible and tempo in the 10 s before / inside / 10 s after the tag;")
        add("    'reappear' = the nearest end of a 2 s+ ball gap, in seconds relative to the tag start (q25/50/75)")
        for row in tags["by_kind"]:
            add("    {:<22} n {:>3} team {:>3}  len s {:>4.1f} {:>4.1f} {:>4.1f}  ball vis {} tempo {}  "
                "reappear n {:>3}: {}".format(
                    row["kind"], row["n"], row["with_team"], *row["seconds"],
                    _row(row["ball_visible"], "{:>4.0%}"), _row(row["tempo"], "{:>4.1f}"),
                    row["reappear_n"], _row(row["reappear_offset_s"], "{:>5.1f}")))
    return "\n".join(lines)
