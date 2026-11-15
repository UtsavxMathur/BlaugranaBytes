"""Lining a broadcast recording up with the tactical one (possession/broadcast.py), on made-up play
with a known answer: a time shift with a cut and a two-frame jump, mirrored axes, close-ups
without a pitch and a replay of earlier play."""

import numpy as np

from possession import broadcast
from possession.tracking import Tracking

PITCH = (105.3, 68.0)


def made_up_play(n, seed=0, n_players=22):
    """(ball (n, 2), players (n, s, 2)) in 0..1: players wander, the ball goes from one to the next."""
    rng = np.random.default_rng(seed)
    scale = np.array(PITCH)
    pos = rng.uniform([10, 5], [95, 63], size=(n_players, 2))
    vel = np.zeros((n_players, 2))
    players = np.empty((n, n_players, 2))
    for f in range(n):
        vel = 0.98 * vel + rng.normal(0.0, 0.4, vel.shape)
        pos = pos + vel / 25.0
        out = (pos < [2, 2]) | (pos > scale - 2)
        vel[out] *= -1
        pos = np.clip(pos, [2, 2], scale - 2)
        players[f] = pos
    ball = np.empty((n, 2))
    f, carrier = 0, 0
    while f < n:
        hold = int(rng.uniform(2, 6) * 25)
        ball[f:f + hold] = players[f:f + hold, carrier] + 0.5
        f += hold
        nxt = int(rng.integers(n_players))
        travel = min(20, n - f)
        if travel > 0:
            share = np.linspace(0, 1, travel)[:, None]
            ball[f:f + travel] = ((1 - share) * players[f:f + travel, carrier]
                                  + share * players[f:f + travel, nxt])
        f += travel
        carrier = nxt
    return ball / scale, players / scale


def scenario():
    """(tactical, broadcast, homography valid, true shift, true ball) of a made-up recording pair."""
    rng = np.random.default_rng(1)
    n_t, n_b = 32000, 26000
    ball, players = made_up_play(n_t)
    seen = np.ones(n_t, dtype=bool)
    for s in rng.integers(0, n_t, 60):
        seen[s:s + int(rng.uniform(5, 20) * 25)] = False       # the tactical ball goes missing
    t_ball = np.where(seen[:, None], ball, np.nan)
    tactical = Tracking(ball=t_ball, players=players, team=np.repeat([0, 1], 11).astype(np.int8),
                        pitch=PITCH, periods=[(0, n_t)])

    b = np.arange(n_b)
    true_shift = np.where(b < 12000, 1500, np.where(b < 18000, 3500, 3502))
    t = b + true_shift
    shown = t.copy()
    valid = np.ones(n_b, dtype=bool)
    for a in (3000, 9000, 15500, 21000):
        valid[a:a + 300] = False                                 # close-ups: no pitch
    shown[6000:6250] = t[6000:6250] - 600                        # a replay of 24 s earlier
    scale = np.array(PITCH)
    focus = ball[shown] * scale
    pos = players[shown] * scale
    in_view = ((np.abs(pos[..., 0] - focus[:, None, 0]) < 22)
               & (np.abs(pos[..., 1] - focus[:, None, 1]) < 14) & valid[:, None])
    pos = pos + rng.normal(0.0, 0.1, pos.shape)
    pos[~in_view] = np.nan
    b_players = np.full((n_b, 30, 2), np.nan)
    b_players[:, 4:26] = 1.0 - pos / scale                      # the broadcast's axes are mirrored
    b_ball = np.where((valid & (rng.random(n_b) < 0.7))[:, None], 1.0 - ball[shown], np.nan)
    feed = Tracking(ball=b_ball, players=b_players, team=np.zeros(30, dtype=np.int8),
                    pitch=PITCH, periods=[(0, n_b)])
    return tactical, feed, valid, true_shift, ball


def test_alignment_recovers_shift_axes_and_pictures():
    tactical, feed, valid, true_shift, ball = scenario()
    n_t, n_b = tactical.n_frames, feed.n_frames
    result = broadcast.align(tactical, feed, valid)

    assert broadcast.FLIPS[result.flip] == (True, True)
    expected = broadcast.tactical_map(true_shift, n_t)
    both = (expected >= 0) & (result.t_to_b >= 0)
    assert both.mean() > 0.75
    assert (np.abs(result.t_to_b[both] - expected[both]) <= 2).mean() > 0.99
    runs = [(s["low"], s["high"]) for s in result.stretches]
    assert len(runs) == 2                                   # the cut; the 2-frame jump is drift
    assert abs(runs[0][0] - 1500) <= 1 and abs(runs[0][1] - 1500) <= 1
    assert abs(runs[1][0] - 3500) <= 1 and abs(runs[1][1] - 3502) <= 1

    state_at = np.full(n_b, -1)
    state_at[result.t_to_b[result.t_to_b >= 0]] = result.state[result.t_to_b >= 0]
    assert (state_at[~valid] == broadcast.OTHER).mean() > 0.95
    replay = state_at[6020:6230]
    assert (replay == broadcast.REPLAY).mean() > 0.6
    assert not (replay == broadcast.LIVE).any()
    normal = valid.copy()
    normal[5900:6350] = False
    counted = normal & (state_at >= 0)
    enough = (~np.isnan(feed.players[..., 0])).sum(axis=1) >= broadcast.MIN_PLAYERS
    assert (state_at[counted & enough] == broadcast.LIVE).mean() > 0.99
    assert not (state_at[counted] == broadcast.REPLAY).any()

    live = result.state == broadcast.LIVE
    close = np.abs(result.ball[live] - ball[live]).max(axis=1) < 0.01
    assert close[~np.isnan(result.ball[live, 0])].mean() > 0.95


def test_tactical_map_fills_frames_a_growing_shift_skips():
    shift = np.array([10] * 5 + [12] * 5)
    out = broadcast.tactical_map(shift, 30)
    assert list(out[10:15]) == [0, 1, 2, 3, 4]
    assert list(out[15:17]) == [4, 4]                     # skipped by the +2 step, filled
    assert list(out[17:22]) == [5, 6, 7, 8, 9]
    assert (out[:10] == -1).all() and (out[22:] == -1).all()


def test_dead_evidence_is_calibrated_by_stage_and_ignores_short_stretches():
    fps = broadcast.SECOND
    n = 20000
    state = np.full(n, broadcast.LIVE, dtype=np.int8)
    p_dead = np.full(n, 0.05)
    for start in range(1000, 19000, 1500):          # twelve 20-s stretches without live pictures
        state[start:start + 20 * fps] = broadcast.OTHER
        p_dead[start:start + 17 * fps] = 0.9        # the model mostly agrees ...
        p_dead[start + 17 * fps:start + 20 * fps] = 0.3   # ... less so just before play resumes
    state[600:600 + fps] = broadcast.REPLAY         # one second only: too short to count
    extra, rows = broadcast.dead_evidence(state, p_dead, [(0, n)])

    assert extra[600:600 + fps].sum() == 0
    assert (extra[state == broadcast.LIVE] == 0).all()
    costs = [row["cost"] for row in rows]
    assert all(row["calibrated"] for row in rows)
    assert costs == sorted(costs) and costs[0] > 0          # weaker towards the end of a stretch
    assert costs[-1] == round(100 * np.log(broadcast.MAX_RATIO))
    start = 1000
    assert extra[start + 20 * fps - 1] == costs[0]           # last frame: "last 2 s" stage
    assert extra[start] == costs[-1]                         # first frame: "earlier" stage


def test_live_cost_only_changes_frames_the_model_was_unsure_of():
    from possession import model as model_module
    prob = np.tile([0.97, 0.02, 0.01], (500, 1))             # sure it is H ...
    prob[200:300] = [0.45, 0.05, 0.50]                       # ... except a doubtful stretch
    extra = np.zeros(500, dtype=np.int64)
    extra[150:350] = 250                                     # evidence over both kinds of frame
    prices = {"flip": 1000, "stop": 4000, "start": 4000, "cap": 600}
    plain = model_module.decode_periods(prob, [(0, 500)], prices)
    with_evidence = model_module.decode_periods(prob, [(0, 500)], prices, live_cost=extra)
    assert (plain == 0).all()
    assert (with_evidence[200:300] == 2).all()
    assert (with_evidence[:200] == 0).all() and (with_evidence[300:] == 0).all()
