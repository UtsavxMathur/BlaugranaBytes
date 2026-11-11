"""Does the simulator's output look like the competition tracking? Print both side by side.

    python training/calibrate.py                       # the default Regime on three public games
    python training/calibrate.py --games dfl-J03WOY    # one game

REAL holds aggregate statistics of the competition tracking: the output of
scripts/profile_atd.py and of generate_submission.py --drift, run by a team member on their own
machine (2026-11-11). They are counts, shares and percentiles over the whole match -- no frame,
no position. The simulator's knobs (training/simulate.py) were set by hand so that rendered
public games give similar numbers. Nothing here uses labels of the competition match.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import features, profile  # noqa: E402
from training import dataset, simulate  # noqa: E402

REAL_MINUTES = 158857 / 25 / 60.0
P5 = (5, 25, 50, 75, 95)
REAL = {
    "missing_share": 0.407,
    # ball gaps per match by length in seconds: <5, 5-10, 10-20, 20-40, 40-90, 90+
    "gaps": [18, 53, 35, 31, 11, 2],
    "runs": [34, 27, 23, 37, 20, 8],       # visible runs per match, same length groups
    # share of gaps during which the players' median speed is 1.4 m/s or more ("play went on")
    "gaps_fast": {"5-10": 23 / 53, "10-20": 10 / 34, "20-40": 4 / 31},
    "tempo_visible": [0.79, 1.26, 1.65, 2.56, 3.92],
    "tempo_missing": [0.24, 0.61, 0.99, 1.31, 2.23],
    # ball lost per second of visible ball, by ball speed just before (bins of profile.SPEED_BINS)
    "hazard_speed": [0.15, 0.08, 0.04, 0.02, 0.01, 0.00, 0.02],
    # ... and by distance to the nearest line (bins of profile.EDGE_BINS_M)
    "hazard_edge": [0.13, 0.18, 0.14, 0.15, 0.08, 0.05, 0.04, 0.02],
    "ball_outside": 0.0419, "ball_outside_depth": [1.9, 9.6, 20.5],
    "ball_edge_hist": [0.017, 0.010, 0.015, 0.021, 0.027, 0.111, 0.363, 0.435],
    "still_share": 0.0199,
    "nearest_m": [0.44, 0.89, 1.91, 4.06, 7.02, 16.35],
    "flips_per_min": 29.6,
    "players_home": [6, 10, 11, 11, 12],
    # percentiles (5, 25, 50, 75, 95) of model features over all graded frames
    "features": {
        "ball_kick": [0.1, 0.4, 0.9, 1.9, 3.7], "ball_speed": [0.5, 2.4, 5.1, 9.5, 17.4],
        "ball_swing": [0.3, 1.6, 4.1, 8.0, 15.2], "kick_around_2": [0.1, 0.6, 1.2, 1.8, 3.0],
        "home_touch_since": [0.0, 1.0, 6.7, 23.6, 30.0],
        "near": [0.3, 0.9, 1.9, 4.1, 9.6], "near_speed": [0.3, 1.2, 2.4, 4.0, 6.5],
        "near_rel_speed": [0.4, 1.4, 3.7, 8.3, 16.0], "ball_edge": [0.5, 6.1, 12.9, 21.1, 30.6],
        "run_len": [6.8, 20.8, 37.2, 60.0, 60.0], "run_prev": [0.2, 5.8, 11.0, 21.9, 51.1],
        "gap_edge_before": [-2.6, 0.7, 5.9, 11.1, 29.0], "gap_edge_after": [-6.6, 0.5, 6.4, 17.9, 33.7],
        "gap_speed_before": [0.1, 0.7, 1.8, 4.8, 9.2], "gap_jump": [1.8, 6.6, 14.0, 26.9, 45.4],
        "gap_implied_speed": [0.1, 0.2, 0.4, 0.9, 3.0],
        "tempo": [0.4, 1.0, 1.4, 2.0, 3.6], "tempo_around_30": [0.5, 1.2, 1.6, 2.0, 2.5],
        "walkers": [0.0, 0.1, 0.2, 0.4, 0.7], "speed_p90": [1.0, 1.5, 2.5, 3.7, 5.4],
        "home_n": [6.0, 10.0, 11.0, 11.0, 12.0], "cx_gap": [2.9, 10.0, 12.7, 14.9, 20.6],
        "home_sx": [6.7, 12.0, 15.3, 19.3, 24.5], "home_sy": [6.0, 9.7, 11.9, 14.9, 18.6],
        "sx_diff": [-15.6, -4.5, 0.7, 7.2, 17.7],
    },
}
GAP_GROUPS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 90), (90, np.inf))


def simulated(game, regime, seed=0):
    """The same aggregates for one rendered public game."""
    damaged = simulate.degrade(game.tracking, game.labels, regime, seed)
    stats = profile.measure(damaged)
    minutes = stats["n_graded"] / 25 / 60.0
    edges = profile.GAP_BINS_S
    counts = np.array([row[0] for row in stats["gap_hist"]], dtype=float)
    gaps = [counts[[i for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])) if a >= lo and b <= hi]].sum()
            * REAL_MINUTES / minutes for lo, hi in GAP_GROUPS]
    fast = {}
    labels = profile._labels(edges, "s")
    for key in REAL["gaps_fast"]:
        row = stats["gap_context"][labels.index(key + "s")]
        tempo_bins = np.array(row["tempo_hist"], dtype=float)
        fast[key] = tempo_bins[3:].sum() / max(tempo_bins.sum(), 1)
    X, names = features.compute(damaged)
    X = X[damaged.in_periods()]
    with np.errstate(all="ignore"):
        feats = {name: np.nanpercentile(X[:, names.index(name)], P5).tolist()
                 for name in REAL["features"]}
    truth = game.labels[damaged.in_periods()]
    missing = ~damaged.ball_visible()[damaged.in_periods()]
    # what each gap really is (only a labelled game can tell): mostly live play, or a stoppage?
    live_gaps, dead_gaps = np.zeros(len(GAP_GROUPS)), np.zeros(len(GAP_GROUPS))
    for start, stop in damaged.periods:
        part_missing = ~damaged.ball_visible()[start:stop]
        part_live = np.isin(game.labels[start:stop], (0, 1))
        from possession import geometry
        for first, run in zip(*geometry.runs_of(part_missing)):
            k = next(i for i, (lo, hi) in enumerate(GAP_GROUPS) if lo <= run / 25.0 < hi)
            if part_live[first:first + run].mean() >= 0.5:
                live_gaps[k] += REAL_MINUTES / minutes
            else:
                dead_gaps[k] += REAL_MINUTES / minutes
    # stoppages with no gap at all (the ball track runs straight through them)
    n_stops = n_hidden = 0
    for start, stop in damaged.periods:
        part_missing = ~damaged.ball_visible()[start:stop]
        for first, run in zip(*geometry.runs_of(game.labels[start:stop] == 2)):
            if run >= 75:
                n_stops += 1
                n_hidden += part_missing[first:first + run].mean() >= 0.3
    run_counts = np.array([row[0] for row in stats["run_hist"]], dtype=float)
    runs = [run_counts[[i for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])) if a >= lo and b <= hi]].sum()
            * REAL_MINUTES / minutes for lo, hi in GAP_GROUPS]
    return {
        "runs": runs,
        "live_gaps": live_gaps.tolist(), "dead_gaps": dead_gaps.tolist(),
        "stoppages_with_gap": n_hidden / max(n_stops, 1),
        "missing_share": stats["missing_share"], "gaps": gaps, "gaps_fast": fast,
        "tempo_visible": stats["tempo_visible"], "tempo_missing": stats["tempo_missing"],
        "hazard_speed": [row[2] for row in stats["hazard_speed"]],
        "hazard_edge": [row[2] for row in stats["hazard_edge"]],
        "ball_outside": stats["ball_outside"]["any"],
        "ball_outside_depth": stats["ball_outside"]["depth_m"],
        "ball_edge_hist": stats["ball_edge_hist"],
        "still_share": sum(row["frame_share"] for row in stats["still_runs"]),
        "nearest_m": stats["nearest_m"], "flips_per_min": stats["nearest_team_flips_per_min"],
        "players_home": stats["players_home"], "features": feats,
        # things only a labelled game can tell (not comparable with the real match)
        "dead_share": float(np.mean(truth == 2)),
        "missing_given_dead": float(missing[truth == 2].mean()),
        "missing_given_live": float(missing[np.isin(truth, (0, 1))].mean()),
    }


def show(real, sims, names):
    def line(label, key, form="{:.2f}", pick=None):
        def cell(source):
            value = source[key] if pick is None else source[key][pick]
            if isinstance(value, (list, tuple)):
                return " ".join(form.format(v) for v in value)
            return form.format(value)
        print("  {:<28} real  {}".format(label, cell(real)))
        for name, sim in zip(names, sims):
            print("  {:<28} {:<5} {}".format("", name[-5:], cell(sim)))

    line("ball missing", "missing_share", "{:.1%}")
    line("gaps <5|5-10|10-20|20-40|40-90|90+", "gaps", "{:>4.0f}")
    line("visible runs, same groups", "runs", "{:>4.0f}")
    for key in real["gaps_fast"]:
        line("gaps {} s, players running".format(key), "gaps_fast", "{:.0%}", key)
    line("tempo, ball visible", "tempo_visible")
    line("tempo, ball missing", "tempo_missing")
    line("losses/s by ball speed", "hazard_speed")
    line("losses/s by line distance", "hazard_edge")
    line("ball off pitch (of visible)", "ball_outside", "{:.1%}")
    line("  depth m p50/90/99", "ball_outside_depth", "{:.1f}")
    line("ball line-distance histogram", "ball_edge_hist", "{:.1%}")
    line("stationary ball share", "still_share", "{:.1%}")
    line("nearest player m", "nearest_m")
    line("nearest team flips per min", "flips_per_min", "{:.1f}")
    line("home players tracked", "players_home", "{:.0f}")
    for name in real["features"]:
        line("feature " + name, "features", "{:>6.1f}", name)
    print("  (labelled games only)")
    for name, sim in zip(names, sims):
        print("  {:<28} {:<5} gaps that are live play {}   stoppages {}   stoppages (3 s+) that leave a gap: {:.0%}".format(
            "", name[-5:], " ".join("{:>3.0f}".format(v) for v in sim["live_gaps"]),
            " ".join("{:>3.0f}".format(v) for v in sim["dead_gaps"]), sim["stoppages_with_gap"]))
    for name, sim in zip(names, sims):
        print("  {:<28} {:<5} dead {:.0%} of frames; ball missing on {:.0%} of dead frames and "
              "{:.1%} of live frames".format("", name[-5:], sim["dead_share"],
                                             sim["missing_given_dead"], sim["missing_given_live"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", nargs="+", default=["dfl-J03WOY", "dfl-J03WQQ", "metrica-2"])
    parser.add_argument("--regime", default="typical", choices=list(simulate.VALIDATION))
    args = parser.parse_args()
    regime = simulate.VALIDATION[args.regime]
    sims = [simulated(dataset.load(name, verbose=False), regime) for name in args.games]
    print("regime '{}': {}".format(args.regime, regime))
    show(REAL, sims, args.games)
    return 0


if __name__ == "__main__":
    sys.exit(main())
