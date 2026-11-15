"""Where a labelling goes wrong, possession by possession. Aggregates only.

The official spellF1 counts how many possessions were found at each IoU threshold. This says why
the others were not, and how far off the edges of the found ones are:

    true possessions     good (IoU >= 0.7), edges off (0.3 to 0.7) or missed (under 0.3); a
                         missed one was called dead, given to the other team, swallowed by a
                         longer possession of the same team, or broken into pieces
    predicted ones       matched (IoU >= 0.3) or invented: inside a stoppage, inside the other
                         team's possession, or a piece of a real possession
    edges                start and end errors of matched possessions, split by what happens at
                         that edge in the truth: a restart after dead ball, a stoppage, or a
                         change of team

IoU here is the best same-team overlap, without the one-to-one matching of the official score,
which is what makes the reasons readable. Used by training/benchmark.py (public games) and by
generate_submission.py --check-public (the public 5 minutes).
"""

import numpy as np

from .submission import pm

GOOD_IOU, LOOSE_IOU = 0.7, 0.3          # the highest and the lowest threshold of spellF1
LENGTH_BINS_S = (0, 2, 4, 8, 16, np.inf)
MISSED_REASONS = ("called dead", "other team", "inside a longer one", "in pieces")
INVENTED_REASONS = ("in a stoppage", "in the other team's", "piece of a real one")
EDGE_KINDS = ("start after a restart", "start after a change", "end at a stoppage",
              "end at a change")


def _spells(codes, runs):
    spells = pm.possession_spells(codes, runs)
    return np.array(spells, dtype=np.int64).reshape(-1, 3)


def _overlapping(spell, others):
    """Indices of `others` (sorted, non-overlapping spells) that overlap `spell` in time."""
    first = int(np.searchsorted(others[:, 1], spell[0], side="right"))
    last = int(np.searchsorted(others[:, 0], spell[1], side="left"))
    return range(first, last)


def _best(spell, others):
    """(best same-team IoU, index of that spell or -1, [(index, shared frames)] same team)."""
    best, partner, shared = 0.0, -1, []
    for j in _overlapping(spell, others):
        if others[j, 2] != spell[2]:
            continue
        inter = min(spell[1], others[j, 1]) - max(spell[0], others[j, 0])
        union = max(spell[1], others[j, 1]) - min(spell[0], others[j, 0])
        shared.append((j, inter))
        if inter / union > best:
            best, partner = inter / union, j
    return best, partner, shared


def _composition(codes, spell):
    """Share of the spell's frames labelled the same team, the other team and D in `codes`."""
    counts = np.bincount(codes[spell[0]:spell[1]].astype(np.int64), minlength=4)
    n = float(spell[1] - spell[0])
    same, other = counts[spell[2]], counts[1 - spell[2]]
    return same / n, other / n, counts[pm.D] / n


def taxonomy(truth, pred, runs=None):
    """Counts of what happened to every true and every predicted possession (dict)."""
    runs = pm.valid_runs(truth) if runs is None else runs
    true_spells, pred_spells = _spells(truth, runs), _spells(pred, runs)
    n_bins = len(LENGTH_BINS_S) - 1
    out = {"true": len(true_spells), "pred": len(pred_spells), "good": 0, "edges": 0,
           "missed": 0, "split": 0, "merged": 0,
           "missed_by": dict.fromkeys(MISSED_REASONS, 0),
           "invented": 0, "invented_by": dict.fromkeys(INVENTED_REASONS, 0),
           "by_length": {"edges_s": list(LENGTH_BINS_S), "true": [0] * n_bins,
                         "good": [0] * n_bins, "found": [0] * n_bins}}
    partners = []
    for spell in true_spells:
        iou, partner, shared = _best(spell, pred_spells)
        length = spell[1] - spell[0]
        k = int(np.searchsorted(LENGTH_BINS_S, length / 25.0, side="right")) - 1
        out["by_length"]["true"][k] += 1
        if iou >= GOOD_IOU:
            out["good"] += 1
            out["by_length"]["good"][k] += 1
        if iou >= LOOSE_IOU:
            out["by_length"]["found"][k] += 1
            out["edges"] += iou < GOOD_IOU
        else:
            out["missed"] += 1
            same, other, dead = _composition(pred, spell)
            if dead >= max(same, other):
                reason = "called dead"
            elif other >= same:
                reason = "other team"
            elif max([n for _, n in shared], default=0) >= 0.5 * length:
                reason = "inside a longer one"
            else:
                reason = "in pieces"
            out["missed_by"][reason] += 1
        out["split"] += sum(n >= 0.2 * length for _, n in shared) >= 2
        partners.append((partner, iou))
    for j in {p for p, _ in partners if p >= 0}:
        claims = sorted((iou for p, iou in partners if p == j), reverse=True)
        out["merged"] += len(claims) - 1
    for spell in pred_spells:
        iou, _, _ = _best(spell, true_spells)
        if iou >= LOOSE_IOU:
            continue
        out["invented"] += 1
        same, other, dead = _composition(truth, spell)
        reason = ("in a stoppage" if dead >= max(same, other) else
                  "in the other team's" if other >= same else "piece of a real one")
        out["invented_by"][reason] += 1
    return out


def edges(truth, pred, runs=None, fps=25):
    """{edge kind: [seconds]} start and end errors of matched possessions (+ = late)."""
    runs = pm.valid_runs(truth) if runs is None else runs
    true_spells, pred_spells = _spells(truth, runs), _spells(pred, runs)
    first_frames = {a for a, _ in runs}
    last_frames = {b for _, b in runs}
    out = {kind: [] for kind in EDGE_KINDS}
    for spell in true_spells:
        iou, partner, _ = _best(spell, pred_spells)
        if iou < LOOSE_IOU:
            continue
        p = pred_spells[partner]
        after_dead = spell[0] in first_frames or truth[spell[0] - 1] == pm.D
        into_dead = spell[1] in last_frames or truth[spell[1]] == pm.D
        out[EDGE_KINDS[0] if after_dead else EDGE_KINDS[1]].append((p[0] - spell[0]) / float(fps))
        out[EDGE_KINDS[2] if into_dead else EDGE_KINDS[3]].append((p[1] - spell[1]) / float(fps))
    return out


def merge_taxonomies(parts):
    """One taxonomy for several games: every count added up."""
    total = {}
    for key, value in parts[0].items():
        if key == "by_length":
            total[key] = {"edges_s": value["edges_s"]}
            for name in ("true", "good", "found"):
                total[key][name] = [sum(p[key][name][i] for p in parts)
                                    for i in range(len(value[name]))]
        elif isinstance(value, dict):
            total[key] = {k: sum(p[key][k] for p in parts) for k in value}
        else:
            total[key] = sum(p[key] for p in parts)
    return total


def merge_edges(parts):
    """One edges() result for several games."""
    return {kind: [v for part in parts for v in part[kind]] for kind in EDGE_KINDS}


def summarise_edges(edge_lists):
    """{kind: {n, median, p10, p90, within_1s}} from edges()."""
    out = {}
    for kind, values in edge_lists.items():
        v = np.asarray(values, dtype=float)
        out[kind] = ({"n": int(len(v)), "median": float(np.median(v)),
                      "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                      "within_1s": float(np.mean(np.abs(v) <= 1.0))} if len(v) else {"n": 0})
    return out


def describe(tax, edge_summary, indent="  "):
    """Text lines for a taxonomy and an edge summary."""
    true, pred = max(tax["true"], 1), max(tax["pred"], 1)
    lines = [
        "{}true possessions {}: good {:.0%}, edges off {:.0%}, missed {:.0%}  (split into 2+ {:.0%}, "
        "merged with a neighbour {:.0%})".format(
            indent, tax["true"], tax["good"] / true, tax["edges"] / true, tax["missed"] / true,
            tax["split"] / true, tax["merged"] / true),
        "{}  missed because: {}".format(indent, ", ".join(
            "{} {:.0%}".format(k, v / true) for k, v in tax["missed_by"].items())),
        "{}  found (IoU >= 0.3) by length: {}".format(indent, "  ".join(
            "{} {:.0%} of {}".format(_bin_name(tax["by_length"]["edges_s"], i), f / max(t, 1), t)
            for i, (f, t) in enumerate(zip(tax["by_length"]["found"], tax["by_length"]["true"])))),
        "{}predicted possessions {}: invented {:.0%} ({})".format(
            indent, tax["pred"], tax["invented"] / pred, ", ".join(
                "{} {:.0%}".format(k, v / pred) for k, v in tax["invented_by"].items())),
        "{}edges of found possessions, seconds (+ = late):  n, median, p10 / p90, within 1 s"
        .format(indent)]
    for kind, row in edge_summary.items():
        lines.append("{}  {:<22} {:>5}".format(indent, kind, row["n"]) + (
            "  {:>+6.2f}  {:>+6.2f} / {:<+6.2f}  {:>4.0%}".format(
                row["median"], row["p10"], row["p90"], row["within_1s"]) if row["n"] else ""))
    return lines


def _bin_name(edges_s, i):
    a, b = edges_s[i], edges_s[i + 1]
    return "{:g}-{:g}s".format(a, b) if np.isfinite(b) else "{:g}s+".format(a)
