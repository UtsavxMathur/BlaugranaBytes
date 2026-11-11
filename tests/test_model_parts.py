"""Decoder, features, simulator and tree export, checked on small synthetic cases."""

import itertools

import numpy as np
import pytest

from possession import decode, features, geometry, model
from possession.tracking import Tracking
from training import simulate


def path_cost(path, cost, flip, stop, start):
    """(total cost, number of label changes) of one label sequence, the slow obvious way."""
    total, changes = int(cost[0, path[0]]), 0
    for t in range(1, len(path)):
        a, b = path[t - 1], path[t]
        if a != b:
            changes += 1
            total += flip if decode.D not in (a, b) else (stop if b == decode.D else start)
        total += int(cost[t, b])
    return total, changes


@pytest.mark.parametrize("seed", range(6))
def test_viterbi_finds_the_cheapest_sequence(seed):
    rng = np.random.default_rng(seed)
    n = 7
    cost = rng.integers(0, 60, size=(n, 3))
    flip, stop, start = (int(v) for v in rng.integers(5, 80, size=3))
    best = min(path_cost(p, cost, flip, stop, start) for p in itertools.product(range(3), repeat=n))
    labels = decode.viterbi(cost, flip, stop, start)
    assert path_cost(labels, cost, flip, stop, start) == best


def test_viterbi_grid_equals_one_setting_at_a_time():
    rng = np.random.default_rng(0)
    cost = rng.integers(0, 300, size=(400, 3))
    flips, stops, starts = np.array([10, 200, 900]), np.array([50, 50, 700]), np.array([5, 400, 400])
    together = decode.viterbi(cost, flips, stops, starts)
    for k in range(3):
        alone = decode.viterbi(cost, int(flips[k]), int(stops[k]), int(starts[k]))
        np.testing.assert_array_equal(together[k], alone)


def test_viterbi_removes_flicker_but_keeps_real_changes():
    # 100 frames clearly H, a 2-frame blip of A, 100 frames H, then 100 frames clearly A
    prob = np.full((302, 3), 0.02)
    prob[:, decode.H] = 0.96
    prob[100:102] = [0.02, 0.96, 0.02]
    prob[202:] = [0.02, 0.96, 0.02]
    labels = decode.viterbi(decode.frame_costs(prob), 1500, 3000, 3000)
    assert (labels[:202] == decode.H).all() and (labels[202:] == decode.A).all()


def test_viterbi_can_force_the_first_label():
    prob = np.tile([0.9, 0.05, 0.05], (50, 1))
    labels = decode.viterbi(decode.frame_costs(prob), 100, 100, 100, first=decode.D)
    assert labels[0] == decode.D and labels[-1] == decode.H


def synthetic_tracking(n=600, seed=0):
    """Two teams of 5 walking about, a ball that follows a home player and is sometimes lost."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(0.2, 0.8, size=(1, 10, 2))
    players = start + np.cumsum(rng.normal(0, 0.0008, size=(n, 10, 2)), axis=0)
    players[:, :5, 0] -= 0.1          # home sits nearer x = 0
    ball = players[:, 2] + rng.normal(0, 0.002, size=(n, 2))
    ball[200:230] = np.nan
    ball[400:520] = np.nan
    team = np.array([0] * 5 + [1] * 5, dtype=np.int8)
    return Tracking(ball=ball, players=players, team=team, pitch=(105.0, 68.0), periods=[(0, n)])


def test_features_are_finite_and_complete():
    tracking = synthetic_tracking()
    X, names = features.compute(tracking)
    assert X.shape == (tracking.n_frames, len(names)) and len(set(names)) == len(names)
    assert not np.isinf(X).any()
    assert not np.isnan(X[:, names.index("vis")]).any()
    assert np.isnan(X[410, names.index("near")])            # long gap: no ball position
    assert not np.isnan(X[210, names.index("near")])        # short gap: bridged


def test_swapping_teams_flips_signed_features():
    tracking = synthetic_tracking()
    plain, names = features.compute(tracking)
    swapped, _ = features.compute(features.swap_teams(tracking))
    for name in ("diff", "sign_past_2.0", "ctrl_next_1.0", "since_diff", "touch_past_2.0"):
        a, b = plain[:, names.index(name)], swapped[:, names.index(name)]
        ok = ~np.isnan(a)
        np.testing.assert_allclose(a[ok], -b[ok], atol=2e-3, err_msg=name)
    for name in ("near", "tempo", "run_len", "ball_edge"):
        a, b = plain[:, names.index(name)], swapped[:, names.index(name)]
        ok = ~np.isnan(a)
        np.testing.assert_allclose(a[ok], b[ok], atol=2e-3, err_msg=name)
    np.testing.assert_allclose(plain[:, names.index("home1")], swapped[:, names.index("away1")],
                               atol=2e-3, equal_nan=True)


def test_combine_swaps_home_and_away_exactly():
    rng = np.random.default_rng(1)
    d1, d2, t1, t2 = rng.normal(size=(4, 50))
    plain = model.combine(d1, d2, t1, t2)
    swapped = model.combine(d2, d1, t2, t1)      # what the model sees with the teams exchanged
    np.testing.assert_allclose(plain[:, 0], swapped[:, 1])
    np.testing.assert_allclose(plain[:, 2], swapped[:, 2])
    np.testing.assert_allclose(plain.sum(axis=1), 1.0)


def moving_ball_game(n=30000, seed=0):
    """A ball wandering over the pitch, 8 players, and stoppages of varied length."""
    rng = np.random.default_rng(seed)
    ball = 0.5 + np.cumsum(rng.normal(0, 0.002, size=(n, 2)), axis=0)
    ball = np.abs((ball % 2.0) - 1.0) * 0.9 + 0.05            # reflect to stay inside the pitch
    tracking = Tracking(ball=ball, players=rng.random((1, 8, 2)) + np.zeros((n, 1, 1)),
                        team=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8),
                        pitch=(105.0, 68.0), periods=[(0, n)])
    labels = np.where(simulate.runs_mask(n, 0.4, 400, np.random.default_rng(seed + 1)), 2, 0)
    return tracking, labels.astype(np.int8)


def test_simulator_delivers_what_the_real_tracker_delivers():
    tracking, labels = moving_ball_game()
    regime = simulate.Regime()
    damaged = simulate.degrade(tracking, labels, regime, seed=3)

    # no short gaps: everything under fill_s is bridged (gaps touching the ends cannot be)
    starts, lengths = geometry.runs_of(~damaged.ball_visible())
    inner = (starts > 0) & (starts + lengths < tracking.n_frames)
    assert len(lengths[inner]) > 5
    assert lengths[inner].min() >= int(regime.fill_s * tracking.fps)

    # most stoppages lose the ball, most live play keeps it, some players go missing
    missing = ~damaged.ball_visible()
    assert missing[labels == 2].mean() > 0.4 > missing[labels == 0].mean()
    assert 0.01 < np.isnan(damaged.players[:, :, 0]).mean() < 0.3

    again = simulate.degrade(tracking, labels, regime, seed=3)
    np.testing.assert_array_equal(damaged.ball, again.ball)       # same seed, same result
    assert not np.isnan(tracking.ball).any()                      # the original is untouched


def test_tag_evidence_only_where_the_model_allows_it():
    from possession import tags
    from possession.smart_tagging import Tag
    fps, n = 25, 20000
    prob = np.tile([0.45, 0.45, 0.10], (n, 1))                    # the model: live everywhere ...
    prob[5000:9000] = [0.05, 0.05, 0.90]                          # ... except one long stoppage
    codes = np.where(prob[:, 2] > 0.5, 2, 0).astype(np.int8)
    items = [Tag("goals", 4400, 8900, 0, "", ""),                 # goal, then its kick-off
             Tag("set pieces", 8875, 9000, -1, "kick off (after goal)", ""),
             Tag("goals", 12000, 15000, 0, "", ""),               # a goal tag the model disputes
             Tag("set pieces", 14975, 15100, -1, "kick off (after goal)", "")]
    goal_windows, candidates = tags.goal_stretches(items, prob, fps)
    assert candidates == 2 and len(goal_windows) == 1
    assert goal_windows[0][0] == 4400 + 20 * fps and 8900 < goal_windows[0][1] < 8875 + 4 * fps

    # restarts: the ball is unseen, then struck 6 s after each tag starts (5 tags of one kind)
    speed = np.full(n, np.nan)
    codes = np.zeros(n, dtype=np.int8)
    pieces = []
    for k, start in enumerate((1000, 4000, 7000, 10000, 13000)):
        strike = start + 6 * fps
        speed[strike - 200:strike] = 1.0                     # carried to the spot ...
        speed[strike:strike + 100] = 12.0                    # ... then struck
        pieces.append(Tag("set pieces", start, start + 875, 0, "corner kick", ""))
        if k < 4:
            codes[strike - 400:strike - 200] = 2             # the model found the stoppage, but
    windows, report = tags.restart_stretches(pieces, codes, speed, fps)   # ended it too early
    n_tags, found, median, spread, used = report["corner kick"]
    assert (n_tags, found, used) == (5, 5, True) and abs(median - 6.0) < 0.1 and spread < 0.1
    assert report["counts"] == {"joined": 4, "alone": 1, "fallback": 0}
    first = min(windows)
    assert first == (1000 + 6 * fps - 201, 1000 + 6 * fps - int(0.5 * fps))   # from the model's D
    assert tags.strike_frame(np.full(500, 1.0), 100, fps) is None             # never struck
    assert tags.strike_frame(np.full(500, 12.0), 100, fps) is None            # fast all along: play
    twice = np.full(600, 1.0)
    twice[150:200], twice[320:400] = 12.0, 12.0                               # a return, then the kick
    assert tags.strike_frames(twice, 100, fps) == [150, 320]
    assert tags.strike_frame(twice, 100, fps, expected=8.0) == 320


def test_tree_export_matches_scikit_learn(tmp_path):
    ensemble = pytest.importorskip("sklearn.ensemble")
    from possession import trees
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 6)).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan
    y = (np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) ** 2 + rng.normal(0, 0.3, 4000)) > 0.3
    estimator = ensemble.HistGradientBoostingClassifier(max_iter=30, early_stopping=False,
                                                        random_state=0).fit(X, y)
    exported = trees.export(estimator)
    np.testing.assert_allclose(trees.raw_score(exported, X), estimator.decision_function(X),
                               rtol=0, atol=1e-10)

    saved = model.Model(names=list("abcdef"), dead=exported, team=exported,
                        prices={"flip": 1, "stop": 2, "start": 3}, info={})
    path = str(tmp_path / "m.npz")
    saved.save(path)
    loaded = model.Model.load(path)
    assert loaded.names == saved.names and loaded.prices == saved.prices
    np.testing.assert_array_equal(trees.raw_score(loaded.team, X), trees.raw_score(exported, X))
