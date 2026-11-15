"""Score held-out predictions in detail, and compare two versions fairly.

    python training/benchmark.py                        # main's model on the saved predictions
    python training/benchmark.py --save main            # ... and keep the result as "main"
    python training/benchmark.py --work-dir D --model-file M --save new --compare main

Reads the held-out predictions that training/train.py saves (.cache/train/oof_*.npz: every
public game predicted by a model that never saw it, under the three simulator settings of
training/simulate.py VALIDATION), decodes them with a model file's prices, and reports for each
data source and setting:

    the official scores: spellF1 at every IoU threshold, macroF1, D-F1, possession counts
    why possessions were missed or invented, and how far off their edges are (possession/errors.py)

A score sheet is kept as .cache/benchmarks/<name>.json. --compare puts two sheets side by side
game by game: a difference is only called real when it is larger than twice its standard error
across games.

Public games only: nothing here reads the match data.
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import errors, model as model_module  # noqa: E402
from possession.submission import pm  # noqa: E402
from training import dataset, simulate, train  # noqa: E402

SHEETS = os.path.join(ROOT, ".cache", "benchmarks")
GAP = np.array([pm.X], dtype=np.int8)
IOUS = ("0.3", "0.4", "0.5", "0.6", "0.7")


def official(truth, pred):
    """The official scores that matter here, from evaluation/possession_metrics.py."""
    r = pm.evaluate(truth, pred, bootstrap_n=0)
    return {"spell_f1": r["spell_f1_mean"],
            "per_iou": {"{:.1f}".format(k): v["f1"] for k, v in r["spell_f1"].items()},
            "macro_f1": r["macro_f1"], "d_f1": r["per_class"]["D"]["f1"],
            "n_pred": r["n_pred_spells"], "n_true": r["n_gt_spells"]}


def joined(parts):
    """One long (truth, prediction) pair from several games, an X frame between their halves."""
    truth = np.concatenate([p for t, _, runs in parts for s, e in runs for p in (t[s:e], GAP)])
    pred = np.concatenate([p for _, q, runs in parts for s, e in runs for p in (q[s:e], GAP)])
    return truth, pred


def run(work_dir, prices):
    """Score sheet (dict) for the held-out predictions in work_dir, decoded with `prices`."""
    train.WORK_DIR = work_dir
    games = [name for fold in train.FOLDS for name in fold]
    oof = train.load_oof(games)
    sources = sorted({name.split("-")[0] for name in games})
    rows = {}
    for (name, regime), (prob, base, truth, periods) in oof.items():
        pred = model_module.decode_periods(prob, periods, prices)
        # The halves, not pm.valid_runs(truth): in the DFL files the halves are back to back
        # with no unlabelled frame between them, and must not be scored as one run.
        rows[(name, regime)] = (truth, pred, base, [tuple(p) for p in periods])

    sheet = {"cells": {}, "games": {}, "detail": {}}
    for (name, regime), (truth, pred, _, runs) in rows.items():
        score = official(*joined([(truth, pred, runs)]))
        sheet["games"].setdefault(name, {})[regime] = {
            "spell_f1": score["spell_f1"], "macro_f1": score["macro_f1"]}
    for source in sources:
        taxonomies, edge_lists = [], []
        for regime in simulate.VALIDATION:
            keys = [k for k in rows if k[0].startswith(source) and k[1] == regime]
            cell = official(*joined([(rows[k][0], rows[k][1], rows[k][3]) for k in keys]))
            cell["baseline_spell_f1"] = official(
                *joined([(rows[k][0], rows[k][2], rows[k][3]) for k in keys]))["spell_f1"]
            sheet["cells"]["{}/{}".format(source, regime)] = cell
            for k in keys:
                taxonomies.append(errors.taxonomy(rows[k][0], rows[k][1], rows[k][3]))
                edge_lists.append(errors.edges(rows[k][0], rows[k][1], rows[k][3]))
        sheet["detail"][source] = {"taxonomy": errors.merge_taxonomies(taxonomies),
                                   "edges": errors.summarise_edges(errors.merge_edges(edge_lists))}
    return sheet


def report(sheet):
    print("\nheld-out results, decoder prices {}, scored against {} labels".format(
        sheet["prices"], sheet.get("labels", "source")))
    print("  {:<8} {:<10} {:>8}  {}  {:>8} {:>6}  {:>11}  {:>8}".format(
        "source", "setting", "spellF1", "  ".join("@" + t for t in IOUS), "macroF1", "D-F1",
        "poss / true", "baseline"))
    for key, cell in sheet["cells"].items():
        source, regime = key.split("/")
        print("  {:<8} {:<10} {:>8.4f}  {}  {:>8.4f} {:>6.3f}  {:>5}/{:<5}  {:>8.4f}".format(
            source, regime, cell["spell_f1"], "  ".join(
                "{:>4.2f}".format(cell["per_iou"][t]) for t in IOUS),
            cell["macro_f1"], cell["d_f1"], cell["n_pred"], cell["n_true"],
            cell["baseline_spell_f1"]))
    for source, detail in sheet["detail"].items():
        print("\n  {} (all three settings together)".format(source))
        for line in errors.describe(detail["taxonomy"], detail["edges"], indent="    "):
            print(line)


def per_game(sheet):
    """{game: spellF1 averaged over the settings}"""
    return {name: float(np.mean([v["spell_f1"] for v in regimes.values()]))
            for name, regimes in sheet["games"].items()}


def compare(new, old):
    print("\n{} against {} (paired by game; spellF1 averaged over the three settings)".format(
        new["name"], old["name"]))
    print("  {:<8} {:>8} {:>8} {:>8}   {:>12}   {:>10}   {}".format(
        "source", old["name"][:8], new["name"][:8], "change", "games better", "noise 2 SE",
        "verdict"))
    a, b = per_game(old), per_game(new)
    games = sorted(set(a) & set(b))
    for source in sorted({g.split("-")[0] for g in games}) + ["all"]:
        chosen = [g for g in games if source == "all" or g.startswith(source)]
        diff = np.array([b[g] - a[g] for g in chosen])
        noise = 2 * diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        mean = float(diff.mean())
        verdict = ("likely real" if abs(mean) > noise and abs(mean) >= 0.005 else
                   "within noise")
        print("  {:<8} {:>8.4f} {:>8.4f} {:>+8.4f}   {:>5} of {:<3}   {:>10.4f}   {}".format(
            source, np.mean([a[g] for g in chosen]), np.mean([b[g] for g in chosen]), mean,
            int((diff > 0).sum()), len(diff), noise, verdict))
    print("\n  by setting:  {:<18} {:>8} {:>8} {:>8}   macroF1 change".format(
        "", old["name"][:8], new["name"][:8], "change"))
    for key in new["cells"]:
        if key in old["cells"]:
            o, n = old["cells"][key], new["cells"][key]
            print("  {:<31} {:>8.4f} {:>8.4f} {:>+8.4f}   {:>+8.4f}".format(
                key, o["spell_f1"], n["spell_f1"], n["spell_f1"] - o["spell_f1"],
                n["macro_f1"] - o["macro_f1"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-dir", default=train.WORK_DIR,
                        help="where training/train.py saved the held-out predictions")
    parser.add_argument("--model-file", default=model_module.DEFAULT_PATH,
                        help="model whose decoder prices are used")
    parser.add_argument("--save", metavar="NAME", help="keep the result as a score sheet")
    parser.add_argument("--compare", metavar="NAME", help="compare with a saved score sheet")
    parser.add_argument("--labels", default="source", choices=dataset.CONVENTIONS,
                        help="the label convention to score against (training/ownership.py)")
    parser.add_argument("--games", default="base", choices=sorted(train.GAME_SETS),
                        help="base: the 7 DFL and 3 Metrica games; all: also the 20 SkillCorner")
    args = parser.parse_args()

    started = time.time()
    train.LABELS, train.FOLDS = args.labels, train.GAME_SETS[args.games]
    prices = model_module.Model.load(args.model_file).prices
    sheet = run(args.work_dir, prices)
    sheet.update(name=args.save or "current", prices=prices, labels=args.labels,
                 created=datetime.date.today().isoformat(),
                 work_dir=os.path.relpath(args.work_dir, ROOT),
                 model_file=os.path.relpath(args.model_file, ROOT))
    report(sheet)
    if args.save:
        os.makedirs(SHEETS, exist_ok=True)
        path = os.path.join(SHEETS, args.save + ".json")
        with open(path, "w") as handle:
            json.dump(sheet, handle, indent=1, sort_keys=True,
                      default=lambda value: value.item())     # numpy scalars
        print("\nsaved score sheet {}".format(os.path.relpath(path, ROOT)))
    if args.compare:
        with open(os.path.join(SHEETS, args.compare + ".json")) as handle:
            compare(sheet, json.load(handle))
    print("\ndone in {:.0f} s".format(time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
