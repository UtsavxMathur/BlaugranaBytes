"""Render clean public tracking the way an automatic tracker might deliver it.

The public games have the true ball position on every frame. An automatic tracker does not: it
loses the ball (in the air, among players, when it lies still, when it leaves the picture), it
may bridge short gaps with a straight line, it smooths the track, it follows the ball for part
of a stoppage, and it loses players the camera does not show. degrade() applies that chain:
decide when the detector sees the ball -> apply the flat-pitch error to airborne balls -> bridge
gaps under fill_s -> smooth -> drop players out of view.

A Regime holds the knobs. NONE OF THEM IS SET FROM THE PROVIDED MATCH. Training draws every knob,
for every copy of every game, from a wide range written down in advance (Regime.random): from a
near-perfect tracker to a poor one, and from a ball never followed in a stoppage to one followed
most of the time. The model has to cope with all of them. Validation uses three fixed regimes
across those ranges: low, mid and high damage (VALIDATION). The ranges are generic statements
about automatic trackers and cameras, not measurements of any tracking.
"""

import dataclasses

import numpy as np

from possession import geometry

H, A, D, X = 0, 1, 2, 3


@dataclasses.dataclass
class Regime:
    """The knobs of degrade(). The defaults are the middle of the ranges in random() (the "mid"
    validation regime); none of them is taken from the provided match."""
    fill_s: float = 3.0                 # gaps shorter than this are bridged with a straight line
    smooth_s: float = 0.25              # smoothing of the delivered ball track (std in seconds)
    ball_noise_m: float = 0.25          # detection noise before smoothing (m)
    # A tracker may keep following the ball after play stops (it rolls on, someone fetches it)
    # and pick it up again before the restart, in the hands or at the feet of the taker.
    # Mean seconds, drawn per stoppage:
    linger_in_s: float = 2.5            # tracked after play stops with the ball on the pitch
    linger_out_s: float = 1.5           # tracked after the ball leaves the pitch ...
    far_side: float = 0.5               # ... if it left on the side the camera can see
    early_chance: float = 0.5           # chance the ball is tracked before the restart at all,
    early_s: float = 3.0                # and for how long (mean); the same knobs exist for
    early_goal_kick: tuple = (0.5, 3.0)  # goal kicks and corners (chance, mean seconds), drawn
    early_corner: tuple = (0.5, 3.0)     # from the same ranges: no kind is assumed to differ
    # In between, the ball is thrown back, passed to the taker, carried: it moves.
    hops: tuple = (0.45, 0.35, 0.20)    # chance of 1, 2 or 3 passes on its way to the taker
    seen_move: float = 0.5              # chance that such a pass is tracked (mid-stoppage)
    seen_hold: float = 0.25             # share of the time it is tracked while someone has it
    # Share of stoppages during which it is tracked from start to finish.
    seen_goal_kick: float = 0.25
    seen_corner: float = 0.25
    seen_other: float = 0.25
    slow_lost: float = 0.5              # a ball lying or held still in play is lost this often
    fail_per_min: float = 4.0           # detection failures per minute of live play
    fail_median_s: float = 1.75         # their median length (log-normal: 1 in 9 is over 3x this)
    air_height: float = 2.5             # a ball above this height (m) is "in the air" ...
    air_lost: float = 0.5               # ... and such a stretch is undetected with this chance
    air_never: float = 9.5              # a stretch that rises above this is never detected
    camera_height: float = 25.0         # camera position, for the flat-pitch error (m)
    camera_back: float = 35.0           # ... distance behind the near touchline (m)
    view_half_m: float = 50.0           # players further than this from the action along x ...
    view_lost: float = 0.5              # ... are untracked this share of the time
    player_missing: float = 0.05        # random loss of single players
    blackout: float = 0.05              # share of time when half the players are lost at once

    def seen_chance(self, kind):
        return {"goal_kick": self.seen_goal_kick, "corner": self.seen_corner}.get(kind, self.seen_other)

    def early(self, kind):
        """(chance, mean seconds) that the ball is tracked again before a restart of this kind."""
        return {"goal_kick": self.early_goal_kick, "corner": self.early_corner}.get(
            kind, (self.early_chance, self.early_s))

    @classmethod
    def random(cls, rng):
        """A regime drawn from wide ranges fixed in advance (see the module docstring)."""
        return cls(fill_s=rng.uniform(0.0, 6.0),                 # from no bridging to 6 s
                   smooth_s=rng.uniform(0.0, 0.5), ball_noise_m=rng.uniform(0.0, 0.5),
                   linger_in_s=rng.uniform(0.0, 5.0), linger_out_s=rng.uniform(0.0, 3.0),
                   far_side=rng.uniform(0.0, 1.0),
                   early_chance=rng.uniform(0.0, 1.0), early_s=rng.uniform(0.0, 6.0),
                   early_goal_kick=(rng.uniform(0.0, 1.0), rng.uniform(0.0, 6.0)),
                   early_corner=(rng.uniform(0.0, 1.0), rng.uniform(0.0, 6.0)),
                   seen_move=rng.uniform(0.0, 1.0), seen_hold=rng.uniform(0.0, 0.5),
                   seen_goal_kick=rng.uniform(0.0, 0.5), seen_corner=rng.uniform(0.0, 0.5),
                   seen_other=rng.uniform(0.0, 0.5),
                   slow_lost=rng.uniform(0.0, 1.0),
                   fail_per_min=rng.uniform(0.0, 8.0), fail_median_s=rng.uniform(0.5, 3.0),
                   air_height=rng.uniform(1.0, 4.0), air_lost=rng.uniform(0.0, 1.0),
                   air_never=rng.uniform(4.0, 15.0),
                   camera_height=rng.uniform(10.0, 40.0), camera_back=rng.uniform(10.0, 60.0),
                   view_half_m=rng.uniform(30.0, 70.0), view_lost=rng.uniform(0.0, 1.0),
                   player_missing=rng.uniform(0.0, 0.1), blackout=rng.uniform(0.0, 0.1))


# Fixed regimes for validation, across the ranges of Regime.random: little damage and little
# tracking in stoppages; the middle of every range; much damage and much tracking in stoppages.
VALIDATION = {
    "low": Regime(fill_s=1.0, smooth_s=0.1, ball_noise_m=0.1, linger_in_s=1.0, linger_out_s=0.5,
                  early_chance=0.3, early_s=1.5, early_goal_kick=(0.3, 1.5),
                  early_corner=(0.3, 1.5), seen_move=0.2, seen_hold=0.05, seen_goal_kick=0.1,
                  seen_corner=0.1, seen_other=0.1, slow_lost=0.2, fail_per_min=1.5,
                  fail_median_s=1.0, air_height=3.0, air_lost=0.2, air_never=12.0,
                  view_lost=0.2, player_missing=0.02, blackout=0.02),
    "mid": Regime(),
    "high": Regime(fill_s=5.0, smooth_s=0.4, ball_noise_m=0.4, linger_in_s=4.0, linger_out_s=2.5,
                   far_side=0.8, early_chance=0.8, early_s=5.0, early_goal_kick=(0.8, 5.0),
                   early_corner=(0.8, 5.0), seen_move=0.8, seen_hold=0.4, seen_goal_kick=0.4,
                   seen_corner=0.4, seen_other=0.4, slow_lost=0.8, fail_per_min=7.0,
                   fail_median_s=2.5, air_height=1.5, air_lost=0.8, air_never=6.0,
                   view_lost=0.8, player_missing=0.08, blackout=0.08)}


def runs_mask(n, share, mean_run, rng):
    """Boolean (n,): True in runs of geometric length (mean mean_run) covering about `share`."""
    if share <= 0:
        return np.zeros(n, dtype=bool)
    share = min(share, 0.999)
    mean_gap = max(mean_run * (1.0 - share) / share, 1.0)
    pairs = int(n / (mean_run + mean_gap)) * 2 + 100
    lengths = np.empty(2 * pairs, dtype=np.int64)
    lengths[0::2] = rng.geometric(1.0 / mean_gap, pairs)
    lengths[1::2] = rng.geometric(1.0 / max(mean_run, 1.0), pairs)
    offset = int(rng.integers(0, int(mean_run + mean_gap) + 1))
    mask = np.repeat(np.tile([False, True], pairs), lengths)[offset:offset + n]
    if len(mask) < n:
        mask = np.concatenate([mask, np.zeros(n - len(mask), dtype=bool)])
    return mask


def _gaussian(values, sigma):
    """Gaussian smoothing along axis 0 that repeats the end values."""
    if sigma <= 0 or len(values) < 3:
        return values
    radius = max(1, int(round(3 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([np.repeat(values[:1], radius, axis=0), values,
                             np.repeat(values[-1:], radius, axis=0)])
    return np.stack([np.convolve(padded[:, k], kernel, mode="valid")
                     for k in range(values.shape[1])], axis=1)


def restart_kind(position, pitch):
    """'corner', 'goal_kick' or 'other', from where the ball is when play restarts."""
    length, width = pitch
    x, y = position
    to_goal_line, to_touchline = min(x, length - x), min(y, width - y)
    if to_goal_line < 2.5 and to_touchline < 2.5:
        return "corner"
    if 3.0 <= to_goal_line <= 8.0 and abs(y - width / 2.0) < 11.0:
        return "goal_kick"
    return "other"


def _dead_story(rolling, target, steps, taker, regime, rng, fps, tau):
    """The ball's path through one stoppage, and when it is with a player or on its way.

    rolling  (m, 2) where it would be if it only rolled on from where it went dead
    steps    (m, slots, 2) the players' real positions during the stoppage
    taker    slot of the player who restarts play, or None

    It rolls to rest, then goes -- in one to three passes, thrown or kicked at 5-14 m/s -- via
    players near it to the taker, and moves with whoever has it. Returns (path, moving, held):
    the positions, and boolean masks of the frames spent in a pass and in someone's possession.
    """
    m = len(rolling)
    path = rolling.copy()
    moving, held = np.zeros(m, dtype=bool), np.zeros(m, dtype=bool)
    rest = min(m - 1, int(rng.uniform(1.5, 3.0) * tau) + 1)       # frames until it lies still
    path[rest:] = rolling[rest]
    if taker is None or m - rest < fps:
        path[m - 1] = target
        return path, moving, held

    # Passes cluster at the two ends of a stoppage: the ball is kicked away or fetched right
    # after play stops, and it reaches the taker shortly before the restart.
    n_hops = 1 + int(rng.choice(len(regime.hops), p=regime.hops))
    low, high = rest, m - int(0.4 * fps)
    times = [high - int(rng.uniform(0.6, 5.6) * fps)]                       # to the taker
    if n_hops >= 2:
        times.append(rest + int(rng.uniform(0.5, 4.0) * fps))               # away from the spot
    if n_hops >= 3:
        times.append(int(rng.integers(low, max(high, low + 1))))
    times = np.sort(np.clip(times, low, max(high, low)))
    position, cursor = path[rest].copy(), rest
    for k, start in enumerate(times):
        start = max(int(start), cursor)
        if k == n_hops - 1:
            slot = taker
        else:                                   # someone within 30 m of the ball
            away = steps[start] - position
            near = np.flatnonzero(np.sqrt((away * away).sum(axis=1)) < 30.0)
            slot = int(rng.choice(near)) if len(near) else taker
        with_player = steps[:, slot] + rng.normal(0.0, 0.3, 2)
        with_player[np.isnan(with_player[:, 0])] = target
        gap = with_player[start] - position
        flight = int(np.clip(np.sqrt((gap * gap).sum()) / rng.uniform(5.0, 14.0) * fps, 3, 3 * fps))
        stop = min(m, start + flight)
        if stop > start:
            weight = np.linspace(0.0, 1.0, stop - start)[:, None]
            path[start:stop] = position * (1 - weight) + with_player[stop - 1] * weight
            moving[start:stop] = True
        end = int(times[k + 1]) if k + 1 < n_hops else m
        if end > stop:
            path[stop:end] = with_player[stop:end]
            held[stop:end] = True
        position, cursor = path[max(min(end, m) - 1, 0)].copy(), min(end, m)
    return path, moving, held


def _stoppage_paths(ball, players, labels, periods, regime, rng, fps, pitch):
    """Where the ball really is during stoppages, and whether the detector sees it.

    The public games do not say (DFL parks the ball where it went dead, Metrica has no position),
    so _dead_story() writes it with the players' real movements. What the detector sees of it:

      * the ball rolling on, for a while after play stops;
      * some of the passes on its way to the taker, and part of the time someone has it;
      * often the last seconds before the restart, when the taker has it.

    For some stoppages it is tracked from start to finish.
    """
    path = ball.copy()
    seen = ~np.isnan(ball[:, 0])
    for start, stop in periods:
        for first, length in zip(*geometry.runs_of(labels[start:stop] == D)):
            first += start
            last = after = first + length             # after = the first live frame
            before = first - 1
            seen[first:last] = False
            path[first:last] = np.nan
            if before < start or after >= stop or np.isnan(ball[before, 0]) \
                    or np.isnan(ball[after, 0]):
                continue                              # nothing to anchor a story on: stays unseen
            origin, target = ball[before], ball[after]
            kind = restart_kind(target, pitch)
            # The ball does not stop the instant play stops: it keeps the motion it had and
            # slows over a second or two. (The public files slow or jump it in the last frames
            # before a stoppage, so its velocity is read 0.4-0.8 s earlier.)
            a10, a20 = max(before - 10, start), max(before - 20, start)
            velocity = (ball[a10] - ball[a20]) / max(a10 - a20, 1)             # metres per frame
            if np.isnan(velocity).any():
                velocity = np.zeros(2)
            speed = np.sqrt((velocity * velocity).sum())
            if speed * fps > 15.0:
                velocity = velocity * (15.0 / (speed * fps))
            tau = rng.uniform(0.5, 1.5) * fps
            t = np.arange(1, length + 1)[:, None]
            rolling = origin + velocity * tau * (1.0 - np.exp(-t / tau))

            gap = players[after] - target             # the taker stands at the ball at the restart
            distance = np.sqrt((gap * gap).sum(axis=1))
            taker = None
            if np.isfinite(distance).any() and np.nanmin(distance) < 4.0:
                taker = int(np.nanargmin(distance))
            story, moving, held = _dead_story(rolling, target, players[first:last], taker,
                                              regime, rng, fps, tau)
            path[first:last] = story

            if rng.random() < regime.seen_chance(kind):
                seen[first:last] = True               # tracked from start to finish
                continue
            visible = np.zeros(length, dtype=bool)
            # (a tracker keeps a rolling ball longer than a still one, hence + tau)
            if geometry.edge_distance(origin, pitch) > 1.0:
                keep = rng.exponential(regime.linger_in_s) + tau / fps      # stopped on the pitch
            elif rng.random() < regime.far_side:
                keep = rng.exponential(regime.linger_out_s) + tau / fps     # rolled out in view
            else:
                keep = rng.uniform(-0.6, 0.3)                        # left the camera's view
            keep = int(round(keep * fps))
            if keep > 0:
                visible[:keep] = True
            elif keep < 0:
                seen[max(start, first + keep):first] = False          # lost before it went dead
            for a, n in zip(*geometry.runs_of(moving)):                # passes on the way
                if rng.random() < regime.seen_move:
                    visible[a:a + n] = True
            visible |= held & runs_mask(length, regime.seen_hold, 2 * fps, rng)
            chance, mean = regime.early(kind)
            back = rng.exponential(mean) if rng.random() < chance else rng.uniform(-0.4, 0.5)
            back = int(round(back * fps))
            if back > 0:
                visible[max(0, length - back):] = True
            elif back < 0:
                seen[after:min(stop, after - back)] = False           # found a moment late
            seen[first:last] = visible
    return path, seen


def degrade(tracking, labels, regime, seed):
    """The tracking as an automatic tracker with these knobs would deliver it. labels: codes per
    frame."""
    rng = np.random.default_rng(seed)
    fps, n = tracking.fps, tracking.n_frames
    length, width = tracking.pitch
    scale = np.array([length, width])
    live = (labels == H) | (labels == A)
    periods = tracking.periods

    # -- the ball: where it is, and when the detector sees it -------------------------------------
    ball = tracking.ball * scale
    path, seen = _stoppage_paths(ball, tracking.players * scale, labels, periods, regime, rng,
                                 fps, tracking.pitch)

    live_frames = np.flatnonzero(live)
    n_fail = rng.poisson(regime.fail_per_min * len(live_frames) / fps / 60.0)
    if n_fail and len(live_frames):
        starts = rng.choice(live_frames, size=n_fail)
        lengths = np.minimum(rng.lognormal(np.log(regime.fail_median_s * fps), 1.0, n_fail),
                             40 * fps)
        for start, run in zip(starts, lengths.astype(np.int64)):
            seen[start:start + max(run, 1)] = False

    # A ball that lies or is held still (a goalkeeper's hands, a foot on the ball, a ball placed
    # for a free kick) may be lost: a detector that relies on motion does not see it.
    step = np.full(n, np.nan)
    step[5:] = np.sqrt(((path[5:] - path[:-5]) ** 2).sum(axis=1)) * (fps / 5.0)
    with np.errstate(invalid="ignore"):
        slow = seen & (step < 0.7)
    for start, run in zip(*geometry.runs_of(slow)):
        if run >= fps and rng.random() < regime.slow_lost:
            begin = start + int(rng.uniform(0.5, 2.0) * fps)
            seen[begin:start + run + int(rng.uniform(0.0, 1.0) * fps)] = False

    observed = path.copy()
    if tracking.ball_z is not None:
        height = np.nan_to_num(tracking.ball_z.astype(np.float64))
        height[~live] = 0.0
        high = live & (height > regime.air_height)
        for start, run in zip(*geometry.runs_of(high)):
            too_high = height[start:start + run].max() > regime.air_never
            if too_high or rng.random() < regime.air_lost:
                seen[max(0, start - 2):start + run + 2] = False
        # A detected airborne ball is drawn where the line of sight from the camera meets the
        # ground: pushed away from the camera by z / (camera height - z) of its distance.
        camera = np.array([length / 2.0, -regime.camera_back])
        height = np.clip(height, 0.0, min(regime.air_never, regime.camera_height - 5.0))
        observed = path + (height / (regime.camera_height - height))[:, None] * (path - camera)

    # -- what is delivered: detections, gaps under fill_s bridged, the result smoothed -------------
    delivered = np.full((n, 2), np.nan)
    ok = seen & ~np.isnan(observed[:, 0])
    delivered[ok] = observed[ok] + rng.normal(0.0, regime.ball_noise_m, size=(int(ok.sum()), 2))
    limit = int(round(regime.fill_s * fps))
    sigma = regime.smooth_s * fps
    for start, stop in periods:
        part = delivered[start:stop]
        for first, run in zip(*geometry.runs_of(np.isnan(part[:, 0]))):
            if run < limit and first > 0 and first + run < len(part):
                weight = (np.arange(1, run + 1) / (run + 1.0))[:, None]
                part[first:first + run] = part[first - 1] * (1 - weight) + part[first + run] * weight
        for first, run in zip(*geometry.runs_of(~np.isnan(part[:, 0]))):
            part[first:first + run] = _gaussian(part[first:first + run], sigma)

    # -- players: those far from the action are out of the camera's view ---------------------------
    players = tracking.players.copy()
    focus = path[:, 0].copy()
    known = ~np.isnan(focus)
    index = np.maximum.accumulate(np.where(known, np.arange(n), 0))
    focus = np.where(np.maximum.accumulate(known), focus[index], length / 2.0)
    focus = _gaussian(focus[:, None], 1.5 * fps)[:, 0]             # the camera pans smoothly
    with np.errstate(invalid="ignore"):
        outside = np.abs(players[:, :, 0] * length - focus[:, None]) > regime.view_half_m
    blackout = runs_mask(n, regime.blackout, 6 * fps, rng)
    for slot in range(tracking.n_slots):
        lost = outside[:, slot] & runs_mask(n, regime.view_lost, 8 * fps, rng)
        lost |= runs_mask(n, regime.player_missing, 4 * fps, rng)
        lost |= blackout & runs_mask(n, 0.5, 6 * fps, rng)
        players[lost, slot] = np.nan

    return tracking.replace(ball=delivered / scale, players=players, ball_z=None, speed=None)
