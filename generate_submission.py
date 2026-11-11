"""Team Blaugrana Bytes: from the data bundle to the submission CSV in one command.

    python generate_submission.py                     # writes "outputs/Blaugrana Bytes.csv"
    python generate_submission.py --check-public
    python generate_submission.py --model baseline --compare evaluation/results/baseline_smooth50.csv

Reads data/tactical/ (or $MTAH_DATA_DIR), writes the CSV, runs the official format checks on it
and prints its SHA-256, so two runs (or a fresh clone) can be compared byte for byte.

  --model possession   our model (models/possession.npz, trained on public games only).
  --model baseline     the organisers' baseline (nearest player + majority vote). Its full-match
                       score is published (spellF1 0.2433), so it is the known floor.
  --compare FILE       % of graded frames on which the output agrees with another submission.
                       With --model baseline against baseline_smooth50.csv this must be 100.00%:
                       it proves the reader, the frame indexing and home/away are right.
  --check-public       score on the public 5 minutes. A check that the pipeline is correct, NOT
                       a way to choose settings (RULES.md): 17 possessions cannot rank methods.
  --drift              print every feature's percentiles on this match next to the training
                       games' (aggregates only): shows where the simulator misses reality.
  --no-tags            do not use the Smart Tagging file (see possession/tags.py for its use).
"""

import argparse
import os
import sys
import time

import numpy as np

from possession import baseline, competition, geometry, model as model_module, submission
from possession import tags as tags_module
from possession.submission import pm
from possession.tracking import show_path

MODELS = ("possession", "baseline")
PUBLIC_BASELINE_SPELL_F1 = 0.3256  # EVALUATION.md: baseline_smooth50 on the public window


def second_half_wait(tracking):
    """[(start, stop)]: the graded frames of the second half before its kick-off, which are dead
    ball by definition (FAQ.md). The kick-off frame is the organisers' own anchor in games.py."""
    from games import GAME, GAMES, anchor_video_frame
    anchor = GAMES[GAME].get("kickoff_2nd")
    if not anchor or len(tracking.periods) < 2:
        return []
    start, stop = tracking.periods[1]
    kickoff = anchor_video_frame(anchor)          # the frame the ball is struck on
    return [(start, kickoff + 1)] if start <= kickoff < stop else []


def predict(model_name, tracking, model_path, tags=None):
    """(label letters for every frame, frame probabilities or None)."""
    if model_name == "baseline":
        return baseline.predict(tracking), None
    if not os.path.isfile(model_path):
        raise SystemExit("{} not found. Train it (python training/train.py) or use "
                         "--model baseline.".format(show_path(model_path)))
    model = model_module.Model.load(model_path)
    print("  model trained on {} public games, decoder prices {}".format(
        len(model.info.get("games", [])), model.prices))
    prob = model_module.probabilities(model, tracking)
    forced = second_half_wait(tracking)
    codes = model_module.decode_periods(prob, tracking.periods, model.prices, forced_dead=forced)
    if tags is not None:
        # Second pass with Smart Tagging evidence, where the first pass allows it (possession/tags.py)
        fps = tracking.fps
        ball, _ = geometry.to_metres(tracking)
        speed = geometry.speed_of(ball, fps, half_span=2)
        goals, n_goals = tags_module.goal_stretches(tags, prob, fps)
        restarts, report = tags_module.restart_stretches(tags, codes, speed, fps)
        again = model_module.decode_periods(prob, tracking.periods, model.prices,
                                            forced_dead=forced + goals + restarts)
        inside = tracking.in_periods()
        count = lambda c: len(pm.possession_spells(c, list(tracking.periods)))  # noqa: E731
        print("  Smart Tagging: {} of {} goal-to-kick-off stretches set to dead".format(
            len(goals), n_goals))
        print("  Smart Tagging, restarts (when the ball is first struck after a set-piece tag):")
        print(tags_module.describe(report))
        print("  ... together this changed {:.2%} of graded frames; possessions {} -> {}; D share "
              "{:.1%} -> {:.1%}".format((again[inside] != codes[inside]).mean(), count(codes),
                                        count(again), (codes[inside] == pm.D).mean(),
                                        (again[inside] == pm.D).mean()))
        codes = again
    return submission.to_letters(codes), prob


def describe(labels, tracking, prob):
    """Aggregate facts about the prediction (counts and shares only), for sanity checks."""
    codes = pm.label_codes(labels)
    spells = pm.possession_spells(codes, list(tracking.periods))
    seconds = np.array([stop - start for start, stop, _ in spells]) / float(tracking.fps)
    inside = np.concatenate([codes[start:stop] for start, stop in tracking.periods])
    stoppages = np.concatenate([geometry.runs_of(codes[start:stop] == pm.D)[1]
                                for start, stop in tracking.periods]) / float(tracking.fps)
    print("  possessions: {}   mean {:.1f} s   median {:.1f} s   under 2 s {:.0%}   over 20 s {:.0%}"
          .format(len(seconds), seconds.mean(), np.median(seconds), (seconds < 2).mean(),
                  (seconds > 20).mean()))
    print("  per half: {}".format("  ".join(
        "{} possessions".format(len(pm.possession_spells(codes, [period])))
        for period in tracking.periods)))
    print("  stoppages: {}   median {:.1f} s   under 3 s {:.0%}   D on {:.1%} of graded frames"
          .format(len(stoppages), np.median(stoppages) if len(stoppages) else 0.0,
                  (stoppages < 3).mean() if len(stoppages) else 0.0, (inside == pm.D).mean()))
    print("  home share of live frames: {:.1%}".format(
        (inside == pm.H).sum() / max(((inside == pm.H) | (inside == pm.A)).sum(), 1)))

    # How the labels relate to the ball being seen: share of frames labelled D, by how long the
    # ball gap they sit in is. (Shows whether "ball unseen" is being read as "ball dead".)
    seen = tracking.ball_visible()
    bins = (0, 1, 3, 10, 30, np.inf)
    dead_frames, all_frames = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    for start, stop in tracking.periods:
        dead = codes[start:stop] == pm.D
        for s, n in zip(*geometry.runs_of(~seen[start:stop])):
            k = int(np.searchsorted(bins, n / float(tracking.fps), side="right")) - 1
            dead_frames[k] += dead[s:s + n].sum()
            all_frames[k] += n
    visible = np.concatenate([seen[start:stop] for start, stop in tracking.periods])
    print("  labelled D: {:.1%} of ball-visible frames; of ball-missing frames by gap length: {}"
          .format((inside[visible] == pm.D).mean(), "  ".join(
              "{:g}-{:g} s {:.0%}".format(a, b, d / max(n, 1)) if np.isfinite(b)
              else "{:g}+ s {:.0%}".format(a, d / max(n, 1))
              for a, b, d, n in zip(bins[:-1], bins[1:], dead_frames, all_frames))))
    if prob is not None:
        graded = prob[tracking.in_periods()]
        print("  model confidence: top probability over 0.9 on {:.1%} of graded frames, under 0.6 "
              "on {:.1%}".format((graded.max(axis=1) > 0.9).mean(), (graded.max(axis=1) < 0.6).mean()))
        same = pm.label_codes(baseline.predict(tracking))[tracking.in_periods()] == inside
        print("  agrees with the organisers' baseline on {:.1%} of graded frames".format(same.mean()))


def compare(labels, other_path, graded):
    other = pm.read_label_csv(other_path)["label"].astype(str).str.strip().str.upper().values
    same = labels[graded] == other[graded]
    print("\nagreement with {} on graded frames: {:.2f}%  ({} of {} differ)".format(
        show_path(other_path), 100.0 * same.mean(), int((~same).sum()), int(graded.sum())))


def check_public(labels, model_name, tracking):
    truth = pm.label_codes(pm.read_label_csv(competition.PUBLIC_GT))
    result = pm.evaluate(truth, pm.label_codes(labels))
    print("\npublic 5-minute check (pipeline correctness only -- never choose settings by it):")
    print("  spellF1 {:.4f}   macroF1 {:.4f}   possessions {} (truth {})".format(
        result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
        result["n_gt_spells"]))
    # Do our three labels mean what the truth's mean? Per class, and where each true class went.
    confusion = np.array(result["confusion"], dtype=float)[:3, :3]
    for k, name in enumerate(("H", "A", "D")):
        row = result["per_class"][name]
        went = confusion[k] / max(confusion[k].sum(), 1)
        print("  true {}: {:>5} frames  precision {:.2f}  recall {:.2f}  F1 {:.2f}   labelled "
              "H {:.0%} / A {:.0%} / D {:.0%}".format(name, row["support"], row["precision"],
                                                       row["recall"], row["f1"], *went))
    codes = pm.label_codes(labels)
    window = truth != pm.X
    stops_true = len(geometry.runs_of(truth[window] == pm.D)[0])
    stops_ours = len(geometry.runs_of(codes[window] == pm.D)[0])
    print("  stoppages in the window: {} (truth {})".format(stops_ours, stops_true))

    # What does the truth's "dead" look like in the tracking? (CHALLENGE.md: the public truth is
    # where to see what the corrected tracking means by dead.) Is the ball tracked while dead,
    # and do we lose the start or the end of a stoppage?
    seen = tracking.ball_visible()
    for name, mask in (("dead", window & (truth == pm.D)),
                       ("live", window & ((truth == pm.H) | (truth == pm.A)))):
        ours_dead = codes[mask] == pm.D
        tracked = seen[mask]
        print("  true {} frames: ball tracked on {:.0%}; we label them dead {:.0%} of the time when "
              "it is tracked, {:.0%} when it is not".format(
                  name, tracked.mean(), ours_dead[tracked].mean() if tracked.any() else 0.0,
                  ours_dead[~tracked].mean() if (~tracked).any() else 0.0))
    fps = float(tracking.fps)
    first = int(np.flatnonzero(window)[0])
    starts, lengths = geometry.runs_of(truth[window] == pm.D)
    for start, length in zip(starts + first, lengths):
        part = codes[start:start + length] == pm.D
        tracked = seen[start:start + length]
        if part.any():
            where = np.flatnonzero(part)
            late, early = where[0] / fps, (length - 1 - where[-1]) / fps
            text = "we call {:.0%} of it dead, starting {:.1f} s late and ending {:.1f} s early".format(
                part.mean(), late, early)
        else:
            text = "we call none of it dead"
        runs = geometry.runs_of(~tracked)[1]
        print("  a true stoppage of {:.1f} s: ball tracked on {:.0%} of it (longest untracked stretch "
              "{:.1f} s); {}".format(length / fps, tracked.mean(),
                                     (runs.max() if len(runs) else 0) / fps, text))
    if model_name == "baseline":
        ok = abs(result["spell_f1_mean"] - PUBLIC_BASELINE_SPELL_F1) < 5e-4
        print("  organisers report {:.4f} for this baseline: {}".format(
            PUBLIC_BASELINE_SPELL_F1, "MATCH" if ok else "MISMATCH -- tell Claude"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--model-file", default=model_module.DEFAULT_PATH)
    parser.add_argument("--data-dir", default=None,
                        help="data bundle root (default: ./data or $MTAH_DATA_DIR)")
    parser.add_argument("--out", default=submission.DEFAULT_OUT)
    parser.add_argument("--no-cache", action="store_true", help="re-parse the ATD from scratch")
    parser.add_argument("--compare", metavar="CSV", help="agreement with another submission")
    parser.add_argument("--check-public", action="store_true",
                        help="score on the public 5 minutes (pipeline check only)")
    parser.add_argument("--no-tags", action="store_true",
                        help="ignore the Smart Tagging file (model output only)")
    parser.add_argument("--drift", action="store_true",
                        help="print each feature's percentiles here next to training's")
    args = parser.parse_args()

    started = time.time()
    tracking = competition.load_tracking(args.data_dir, use_cache=not args.no_cache)
    graded = tracking.in_periods()
    print("  ball visible on {:.1f}% of graded frames (DATA.md says 59.3%)".format(
        100.0 * tracking.ball_visible()[graded].mean()))

    print("model: {}".format(args.model))
    tags = None
    if args.model == "possession" and not args.no_tags:
        try:
            tags, _ = competition.load_tags(tracking, args.data_dir)
        except IOError:
            print("  (no Smart Tagging file found: using the model alone)")
    letters, prob = predict(args.model, tracking, args.model_file, tags)
    labels = submission.finalize(letters, tracking.periods)

    path = submission.write(args.out, labels)
    reason = submission.check(path, competition.n_frames())
    if reason:
        print("\nFAIL  official format check: {}".format(reason))
        return 1

    counts = {name: int(np.sum(labels == name)) for name in ("H", "A", "D", "X")}
    print("\nwrote {}  (official format check: OK)".format(show_path(path)))
    print("  labels  H {H}  A {A}  D {D}  X {X}".format(**counts))
    describe(labels, tracking, prob)
    print("  sha256  {}".format(submission.sha256(path)))

    if args.drift and args.model == "possession":
        print("\nfeatures on this match vs the training games (aggregates; most different first):")
        print(model_module.drift_report(model_module.Model.load(args.model_file), tracking))
    if args.compare:
        compare(labels, args.compare, graded)
    if args.check_public:
        check_public(labels, args.model, tracking)
    print("\ndone in {:.0f} s".format(time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
