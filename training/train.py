"""Train the model on the public games, and measure it on games it has not seen.

    python training/train.py              # cross-validation, decoder prices, final model
    python training/train.py --quick      # small and fast, to check that everything runs
    python training/train.py --steps tune # reuse saved predictions, only redo the prices

Steps:
  rows    every public game is damaged (training/simulate.py) under a few random regimes, turned
          into features, and thinned to every 4th frame (neighbouring frames are near-duplicates).
  cv      5 folds, split BY GAME: train on 8 games, predict the 2 held out, under three fixed
          damage regimes. Nothing a fold predicts was seen in its training.
  tune    choose the decoder prices on those held-out predictions, by the official spellF1.
  final   train on all 10 games and write models/possession.npz.

Needs scikit-learn (requirements-train.txt). Making the submission does not.
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import baseline, decode, features, model as model_module, trees  # noqa: E402
from possession.submission import pm  # noqa: E402
from training import dataset, simulate  # noqa: E402

WORK_DIR = os.path.join(ROOT, ".cache", "train")
FOLDS = [["dfl-J03WMX", "metrica-1"], ["dfl-J03WN1", "metrica-2"], ["dfl-J03WOH", "metrica-3"],
         ["dfl-J03WOY", "dfl-J03WPY"], ["dfl-J03WQQ", "dfl-J03WR9"]]
COPIES = 3            # rendered copies of each training game (odd copies have the teams swapped)
THIN = 4              # keep every 4th frame for training
SPELL_WEIGHT_S = 12.0 # segments shorter than this many seconds are up-weighted (0 = off)
WEIGHT_CAP = 8.0      # ... by at most this factor
H, A, D, X = 0, 1, 2, 3

# Decoder settings tried on the held-out predictions (see possession/decode.py).
CAPS = (200, 600, 1200)
FLIPS = (100, 300, 1000, 3000)
STOPS = (1000, 4000, 12000)
STARTS = (1000, 4000, 12000)


def seed_of(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def swap_labels(labels):
    out = labels.copy()
    out[labels == H], out[labels == A] = A, H
    return out


def spell_weights(labels, fps, reference_s, most):
    """A weight per frame that makes short segments count as much as long ones.

    The ranking metric counts every possession once, however long it is, while a classifier
    trained on frames sees a 40 s possession 40 times more often than a 1 s one. Frames of a
    segment (a run of one label) shorter than reference_s get weight reference_s / length, capped
    at `most`; all other frames get 1.
    """
    weights = np.ones(len(labels), dtype=np.float32)
    if reference_s <= 0:
        return weights
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(labels)]))
    seconds = np.diff(bounds) / float(fps)
    per_run = np.clip(reference_s / np.maximum(seconds, 1e-9), 1.0, most)
    return np.repeat(per_run, np.diff(bounds)).astype(np.float32)


def training_rows(game, copy, spell_weight=0.0):
    """(X, labels, weights, names) for one damaged copy of a game, thinned."""
    rng = np.random.default_rng(seed_of(game.name, "regime", copy))
    regime = simulate.Regime.random(rng)
    tracking, labels = game.tracking, game.labels
    if copy % 2 == 1:
        tracking, labels = features.swap_teams(tracking), swap_labels(labels)
    damaged = simulate.degrade(tracking, labels, regime, seed_of(game.name, "damage", copy))
    X, names = features.compute(damaged)
    rows = np.flatnonzero(tracking.in_periods() & (labels != X_CODE))[copy % THIN::THIN]
    weights = spell_weights(labels, tracking.fps, spell_weight, WEIGHT_CAP)
    return X[rows], labels[rows], weights[rows], names


X_CODE = X  # (the name X is used for feature matrices below)


def fit(X, y, weight, quick):
    from sklearn.ensemble import HistGradientBoostingClassifier
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=60 if quick else 300, max_leaf_nodes=31, max_depth=8,
        min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)
    return estimator.fit(X, y, sample_weight=weight)


def fit_heads(parts, quick):
    """Train the two ensembles on a list of (X, labels, game index) parts."""
    X = np.concatenate([p[0] for p in parts])
    labels = np.concatenate([p[1] for p in parts])
    weight = np.concatenate([p[2] / p[2].sum() for p in parts])   # every game copy counts the same
    weight *= len(weight) / weight.sum()
    dead = fit(X, labels == D, weight, quick)
    live = labels != D
    team = fit(X[live], labels[live] == H, weight[live], quick)
    return dead, team


def held_out_probabilities(game, regime_name, dead, team):
    """(probabilities (n, 3), baseline codes (n,)) for one held-out game under one regime."""
    regime = simulate.VALIDATION[regime_name]
    damaged = simulate.degrade(game.tracking, game.labels, regime, seed_of(game.name, regime_name))
    inside = damaged.in_periods()
    plain, _ = features.compute(damaged)
    swapped, _ = features.compute(features.swap_teams(damaged))
    prob = np.full((damaged.n_frames, 3), np.nan, dtype=np.float32)
    prob[inside] = model_module.combine(
        dead.decision_function(plain[inside]), dead.decision_function(swapped[inside]),
        team.decision_function(plain[inside]), team.decision_function(swapped[inside]))
    return prob, pm.label_codes(baseline.predict(damaged))


def top_features(estimator, names, k=12):
    """[(feature, share of the total split gain)]: what the ensemble relies on most."""
    gain = np.zeros(len(names))
    for stage in estimator._predictors:
        nodes = stage[0].nodes
        split = ~nodes["is_leaf"].astype(bool)
        np.add.at(gain, nodes["feature_idx"][split], nodes["gain"][split])
    order = np.argsort(-gain)[:k]
    return [(names[i], float(gain[i] / gain.sum())) for i in order]


def check_export(estimator, X):
    """Our numpy tree walk must give scikit-learn's own scores."""
    sample = X[:: max(1, len(X) // 5000)]
    ours = trees.raw_score(trees.export(estimator), sample)
    theirs = estimator.decision_function(sample)
    worst = float(np.abs(ours - theirs).max())
    if worst > 1e-9:
        raise AssertionError("tree export disagrees with scikit-learn by {}".format(worst))
    return worst


# -- steps -----------------------------------------------------------------------------------------

def step_rows(games, quick, spell_weight=0.0):
    rows, names = {}, None
    for name in games:
        game = dataset.load(name, verbose=False)
        started = time.time()
        rows[name] = []
        for copy in range(1 if quick else COPIES):
            X, labels, weights, names = training_rows(game, copy, spell_weight)
            rows[name].append((X, labels, weights))
        print("  rows {:<12} {:>7} rows  ({:.0f} s)".format(
            name, sum(len(r[1]) for r in rows[name]), time.time() - started), flush=True)
    return rows, names


def step_cv(rows, quick):
    os.makedirs(WORK_DIR, exist_ok=True)
    for fold, held_out in enumerate(FOLDS):
        started = time.time()
        parts = [part for name in rows if name not in held_out for part in rows[name]]
        dead, team = fit_heads(parts, quick)
        if fold == 0:
            print("  tree export check: max difference {:.1e}".format(
                check_export(dead, parts[0][0])), flush=True)
        for name in held_out:
            game = dataset.load(name, verbose=False)
            for regime_name in simulate.VALIDATION:
                prob, base = held_out_probabilities(game, regime_name, dead, team)
                np.savez(os.path.join(WORK_DIR, "oof_{}_{}.npz".format(name, regime_name)),
                         prob=prob, baseline=base)
        print("  fold {} (held out {}) done in {:.0f} s".format(fold, ", ".join(held_out),
                                                             time.time() - started), flush=True)


def load_oof(games):
    """{(game, regime): (prob, baseline codes, truth codes, periods)}"""
    out = {}
    for name in games:
        game = dataset.load(name, verbose=False)
        for regime_name in simulate.VALIDATION:
            path = os.path.join(WORK_DIR, "oof_{}_{}.npz".format(name, regime_name))
            if not os.path.isfile(path):
                raise IOError("{} missing: run the cv step first".format(path))
            with np.load(path) as z:
                out[(name, regime_name)] = (z["prob"].astype(np.float64), z["baseline"],
                                            game.labels, game.tracking.periods)
    return out


def pooled_scores(cells):
    """Official scores for several (truth, prediction, periods) cells joined into one match."""
    gap = np.array([X_CODE], dtype=np.int8)
    truth = np.concatenate([part for t, _, periods in cells for s, e in periods
                            for part in (t[s:e], gap)])
    pred = np.concatenate([part for _, p, periods in cells for s, e in periods
                           for part in (p[s:e], gap)])
    return pm.evaluate(truth, pred)


def decode_grid(prob, periods, grid):
    """(c, n) codes for c decoder settings (cap, flip, stop, start) at once (X outside the halves)."""
    grid = np.asarray(grid)
    labels = np.full((len(grid), len(prob)), X_CODE, dtype=np.int8)
    for cap in np.unique(grid[:, 0]):
        which = np.flatnonzero(grid[:, 0] == cap)
        for start, stop in periods:
            labels[which, start:stop] = decode.viterbi(
                decode.frame_costs(prob[start:stop], cap), grid[which, 1], grid[which, 2],
                grid[which, 3])
    return labels


def step_tune(games, grid=None):
    """Decode the held-out predictions with every setting and score each one.

    Returns (grid, sources, regimes, score[setting, source, regime], oof, decoded).
    """
    oof = load_oof(games)
    if grid is None:
        grid = [(c, f, s, t) for c in CAPS for f in FLIPS for s in STOPS for t in STARTS]
    decoded = {key: decode_grid(prob, periods, grid) for key, (prob, _, _, periods) in oof.items()}

    sources = sorted({name.split("-")[0] for name, _ in oof})
    regimes = list(simulate.VALIDATION)
    gap = np.array([X_CODE], dtype=np.int8)
    score = np.zeros((len(grid), len(sources), len(regimes)))
    for i_source, source in enumerate(sources):
        for i_regime, regime_name in enumerate(regimes):
            keys = [key for key in oof if key[0].startswith(source) and key[1] == regime_name]
            truth = np.concatenate([p for key in keys for s, e in oof[key][3]
                                    for p in (oof[key][2][s:e], gap)])
            runs = pm.valid_runs(truth)
            for k in range(len(grid)):
                pred = np.concatenate([p for key in keys for s, e in oof[key][3]
                                       for p in (decoded[key][k][s:e], gap)])
                score[k, i_source, i_regime] = pm.spell_scores(truth, pred, runs)["mean_f1"]
    return grid, sources, regimes, score, oof, decoded


def report(grid, sources, regimes, score, oof, decoded, best):
    print("\nheld-out results at cap {} / flip {} / stop {} / start {}".format(*grid[best]))
    print("{:<8} {:<7} {:>9} {:>9}   {:>9} {:>9}   {:>12} {:>9}".format(
        "source", "regime", "spellF1", "baseline", "macroF1", "baseline", "poss p/true", "D-F1"))
    for source in sources:
        for regime_name in regimes:
            keys = [key for key in oof if key[0].startswith(source) and key[1] == regime_name]
            ours = pooled_scores([(oof[k][2], decoded[k][best], oof[k][3]) for k in keys])
            base = pooled_scores([(oof[k][2], oof[k][1], oof[k][3]) for k in keys])
            print("{:<8} {:<7} {:>9.4f} {:>9.4f}   {:>9.4f} {:>9.4f}   {:>5}/{:<6} {:>9.3f}".format(
                source, regime_name, ours["spell_f1_mean"], base["spell_f1_mean"],
                ours["macro_f1"], base["macro_f1"], ours["n_pred_spells"], ours["n_gt_spells"],
                ours["per_class"]["D"]["f1"]))


def step_final(rows, names, prices, quick, path, spell_weight=0.0):
    parts = [part for name in rows for part in rows[name]]
    dead, team = fit_heads(parts, quick)
    check_export(dead, parts[0][0])
    check_export(team, parts[0][0])
    info = {"games": sorted(rows), "copies": 1 if quick else COPIES, "thin": THIN,
            "spell_weight_s": float(spell_weight),
            "rows": int(sum(len(p[1]) for p in parts)), "quick": bool(quick),
            "top_features_dead": top_features(dead, names),
            "top_features_team": top_features(team, names)}
    # Summary of every feature on the training rows, kept in the model file so that
    # generate_submission.py --drift can show where the real match looks different.
    everything = np.concatenate([p[0] for p in parts])
    with np.errstate(all="ignore"):
        info["feature_stats"] = {
            "percentiles": list(model_module.DRIFT_PERCENTILES),
            "values": np.round(np.nanpercentile(everything, model_module.DRIFT_PERCENTILES,
                                                axis=0).T.astype(np.float64), 3).tolist(),
            "missing": np.round(np.isnan(everything).mean(axis=0), 4).tolist()}
    for head in ("dead", "team"):
        print("  what the '{}' ensemble relies on: {}".format(head, ", ".join(
            "{} {:.0%}".format(name, share) for name, share in info["top_features_" + head])))
    model_module.Model(names=names, dead=trees.export(dead), team=trees.export(team),
                       prices=prices, info=info).save(path)
    print("\nwrote {}  ({:.1f} MB)".format(os.path.relpath(path, ROOT), os.path.getsize(path) / 1e6))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", nargs="+", default=["cv", "tune", "final"],
                        choices=["cv", "tune", "final"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=model_module.DEFAULT_PATH)
    parser.add_argument("--spell-weight", type=float, default=SPELL_WEIGHT_S,
                        help="up-weight segments shorter than this many seconds (0 = off)")
    parser.add_argument("--weight-cap", type=float, default=None,
                        help="largest factor a short segment is up-weighted by")
    parser.add_argument("--work-dir", default=None, help="where held-out predictions are kept")
    args = parser.parse_args()
    if args.weight_cap:
        global WEIGHT_CAP
        WEIGHT_CAP = args.weight_cap
    if args.work_dir:
        global WORK_DIR
        WORK_DIR = args.work_dir
        os.makedirs(WORK_DIR, exist_ok=True)
    games = [name for fold in FOLDS for name in fold]

    rows = names = None
    if "cv" in args.steps or "final" in args.steps:
        print("building training rows ...")
        rows, names = step_rows(games, args.quick, args.spell_weight)
    if "cv" in args.steps:
        print("cross-validation ...")
        step_cv(rows, args.quick)

    prices_path = os.path.join(WORK_DIR, "prices.json")
    if "tune" in args.steps:
        print("choosing decoder prices on held-out predictions ...")
        grid, sources, regimes, score, oof, decoded = step_tune(games)
        # The possession-change price is judged on the Metrica games (their labelling convention
        # is the competition's); the stoppage prices on all games. Mean over the three regimes.
        overall = score.mean(axis=(1, 2))
        best = int(overall.argmax())
        report(grid, sources, regimes, score, oof, decoded, best)
        prices = dict(zip(("cap", "flip", "stop", "start"), (int(v) for v in grid[best])))
        np.save(os.path.join(WORK_DIR, "grid.npy"), np.array(grid))
        with open(prices_path, "w") as handle:
            json.dump(prices, handle)
        np.save(os.path.join(WORK_DIR, "grid_scores.npy"), score)
    if "final" in args.steps:
        with open(prices_path) as handle:
            prices = json.load(handle)
        print("training the final model on all games ...")
        step_final(rows, names, prices, args.quick, args.out, args.spell_weight)
    return 0


if __name__ == "__main__":
    sys.exit(main())
