"""Combine the "dead?" ensemble of one model with the "home?" ensemble of another.

    python training/train.py --out .cache/models/possession_v1.npz
    python training/train.py --labels controlled --work-dir .cache/train_controlled \\
        --out .cache/models/possession_controlled.npz
    python training/combine.py              # writes models/possession.npz

The controlled-possession convention (training/ownership.py) changes who owns the ball, never
whether it is dead. So the "dead?" ensemble should not change with it. But train.py weights its
training frames by the length of the segment they sit in, and with longer possessions the live
frames weigh less, which shifts the "dead?" ensemble too. This keeps the "dead?" ensemble of the
model trained on the labels as the files give them, and takes the "home?" ensemble of the model
trained on controlled labels.

Both models' held-out predictions come from the same folds and the same simulated tracking, so
the combined held-out predictions follow exactly:

    P(D) from the first model,  P(H | alive) from the second

and the decoder prices are chosen on them like train.py chooses them (the official spellF1,
averaged over the data sources and simulator settings). Public games only.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import model as model_module  # noqa: E402
from training import simulate, train  # noqa: E402


def combine_held_out(dead_dir, team_dir, out_dir, games):
    os.makedirs(out_dir, exist_ok=True)
    for name in games:
        for regime in simulate.VALIDATION:
            file = "oof_{}_{}.npz".format(name, regime)
            with np.load(os.path.join(dead_dir, file)) as z:
                p_dead, baseline = z["prob"][:, 2].astype(np.float64), z["baseline"]
            with np.load(os.path.join(team_dir, file)) as z:
                team = z["prob"].astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                p_home = team[:, 0] / (team[:, 0] + team[:, 1])
            prob = np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], axis=1)
            np.savez(os.path.join(out_dir, file), prob=prob.astype(np.float32), baseline=baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead-model", default=os.path.join(ROOT, ".cache", "models", "possession_v1.npz"),
                        help="model trained on the labels as given (python training/train.py)")
    parser.add_argument("--dead-work-dir", default=os.path.join(ROOT, ".cache", "train"))
    parser.add_argument("--team-model",
                        default=os.path.join(ROOT, ".cache", "models", "possession_controlled.npz"),
                        help="model trained on controlled labels (train.py --labels controlled)")
    parser.add_argument("--team-work-dir", default=os.path.join(ROOT, ".cache", "train_controlled"))
    parser.add_argument("--work-dir", default=os.path.join(ROOT, ".cache", "train_hybrid"))
    parser.add_argument("--labels", default="controlled", help="convention the prices are chosen on")
    parser.add_argument("--out", default=model_module.DEFAULT_PATH)
    args = parser.parse_args()

    dead, team = model_module.Model.load(args.dead_model), model_module.Model.load(args.team_model)
    if dead.names != team.names:
        raise SystemExit("the two models were trained on different features")
    games = [name for fold in train.FOLDS for name in fold]
    combine_held_out(args.dead_work_dir, args.team_work_dir, args.work_dir, games)

    train.WORK_DIR, train.LABELS = args.work_dir, args.labels
    grid, sources, regimes, score, oof, decoded = train.step_tune(games)
    best = int(score.mean(axis=(1, 2)).argmax())
    train.report(grid, sources, regimes, score, oof, decoded, best)
    prices = dict(zip(("cap", "flip", "stop", "start"), (int(v) for v in grid[best])))
    with open(os.path.join(args.work_dir, "prices.json"), "w") as handle:
        json.dump(prices, handle)

    info = dict(team.info, combined={
        "dead": os.path.relpath(args.dead_model, ROOT), "dead_labels": dead.info.get("labels", "source"),
        "team": os.path.relpath(args.team_model, ROOT), "team_labels": team.info.get("labels", "source")})
    model_module.Model(names=dead.names, dead=dead.dead, team=team.team, prices=prices,
                       info=info).save(args.out)
    print("\nwrote {} with decoder prices {}".format(os.path.relpath(args.out, ROOT), prices))
    return 0


if __name__ == "__main__":
    sys.exit(main())
