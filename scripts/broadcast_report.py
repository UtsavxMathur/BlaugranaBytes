"""What the broadcast recording could add, measured on the real match. Aggregates only.

    python scripts/broadcast_report.py

The broadcast feed is not synchronised with the tactical one and no mapping is provided
(DATA.md). This lines the two up from their tracking (possession/broadcast.py), says how well
that worked, and then compares what the broadcast was showing with the labels the current
pipeline produces.

Everything printed is a count, a share, a percentile or a time shift -- nothing frame by frame --
so the output is safe to paste back (CLAUDE.md, section 0).
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import generate_submission  # noqa: E402
from possession import broadcast, competition, geometry, submission  # noqa: E402
from possession import model as model_module  # noqa: E402
from possession.broadcast import LIVE, OTHER, REPLAY, STATES  # noqa: E402
from possession.submission import pm  # noqa: E402
from possession.tracking import HOME  # noqa: E402

FPS = 25.0
GAP_BINS = (0, 1, 5, 10, 20, 40, 90, np.inf)


def clock(frames):
    """Frames as [-]m:ss.s"""
    tenths = int(round(abs(frames) / FPS * 10))
    return "{}{}:{:04.1f}".format("-" if frames < 0 else "", tenths // 600, (tenths % 600) / 10)


def pct(values, ps=(10, 50, 90), form="{:.2f}"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return "n/a"
    return " / ".join(form.format(v) for v in np.percentile(values, ps))


def share(part, whole):
    return float(np.sum(part)) / max(float(np.sum(whole)), 1.0)


def shares(state, mask):
    return "  ".join("{} {:.1%}".format(name, share(mask & (state == k), mask))
                     for k, name in enumerate(STATES))


def runs_in(mask, periods, min_frames=1):
    """[(start, stop)] maximal runs of True inside the periods, at least min_frames long."""
    out = []
    for a, b in periods:
        starts, lengths = geometry.runs_of(mask[a:b])
        out += [(a + s, a + s + n) for s, n in zip(starts, lengths) if n >= min_frames]
    return out


def gap_bin(frames):
    return int(np.searchsorted(GAP_BINS, frames / FPS, side="right")) - 1


def gap_names():
    return ["{:g}-{:g} s".format(a, b) if np.isfinite(b) else "{:g}+ s".format(a)
            for a, b in zip(GAP_BINS[:-1], GAP_BINS[1:])]


# -- sections -------------------------------------------------------------------------------------

def section_files(feed, valid, error, tactical):
    print("\n[1] the broadcast files")
    print("  {} broadcast frames ({} expected), {} track slots; teams {} / {} (tactical {} / {})"
          .format(feed.n_frames, broadcast.source_frames("broadcast", broadcast.GAME), feed.n_slots,
                  feed.team_names[0], feed.team_names[1], *tactical.team_names))
    print("  frame rate {} (tactical {}){}".format(
        feed.fps, tactical.fps, "" if feed.fps == tactical.fps else
        "   WARNING: a constant frame shift cannot line these up -- tell Claude"))
    seen = feed.ball_visible()
    print("  ball seen on {:.1%} of frames (DATA.md 50.5%), homography valid on {:.1%} (DATA.md "
          "65.8%)".format(seen.mean(), valid.mean()))
    print("  ball seen on {:.1%} of valid frames and on {:.1%} of the others".format(
        share(seen & valid, valid), share(seen & ~valid, ~valid)))
    count = (~np.isnan(feed.players[..., 0])).sum(axis=1)
    print("  tracked players per valid frame p10/p50/p90 {}; frames without a valid homography "
          "that still have players {:.1%}".format(pct(count[valid], form="{:.0f}"),
                                                  share((count > 0) & ~valid, ~valid)))
    print("  reprojection error on valid frames p50/p90/p99 {}".format(
        pct(error[valid], (50, 90, 99))))
    for name, xy in (("ball", feed.ball), ("players", feed.players.reshape(-1, 2))):
        print("  {:<8} x p1/p50/p99 {}   y p1/p50/p99 {}   (expected within 0..1)".format(
            name, pct(xy[:, 0], (1, 50, 99)), pct(xy[:, 1], (1, 50, 99))))


def section_coarse(al):
    print("\n[2] coarse alignment: ball tracks compared second by second, per 5 minutes of "
          "broadcast")
    for k, (clear, hits) in enumerate(al.flip_totals):
        print("  axes {:<18} clear chunks {:>3}, matching samples in them {:>6}{}".format(
            broadcast.FLIP_NAMES[k], clear, hits, "   <- used" if k == al.flip else ""))
    print("  {:>3}  {:>9}  {:>24}  {:>15}  {:>9}".format(
        "#", "broadcast", "tactical = broadcast +", "hits / both", "runner-up"))
    for i, row in enumerate(al.chunks):
        found = row["hits"] > 0
        print("  {:>3}  {:>9}  {:>8} {:>15}  {:>6} / {:<6}  {:>9}  {}".format(
            i, clock(row["start"]), row["offset"] if found else "-",
            "(" + clock(row["offset"]) + ")" if found else "", row["hits"], row["both"],
            row["runner_up"], "clear" if row["clear"] else "-"))


def section_fine(al):
    windows = al.windows
    usable = [w for w in windows if w is not None]
    good = broadcast.good_windows(windows)
    print("\n[3] fine alignment: players compared frame by frame, per 10 s of broadcast")
    print("  windows {}: with enough players {} ({:.0%}), good {} ({:.0%} of those)".format(
        len(windows), len(usable), share(len(usable), len(windows)), len(good),
        share(len(good), len(usable))))
    print("  player distance at the best shift (m) p10/p50/p90: good {}   all usable {}".format(
        pct([w["score"] for w in good]), pct([w["score"] for w in usable])))
    print("  sharpness (m worse 1 s either side) p10/p50/p90: good {}".format(
        pct([w["sharp"] for w in good])))
    print("  stretches with one shift (tactical frame = broadcast frame + shift):")
    for s in al.stretches:
        print("    broadcast {:>9} - {:>9}   shift {:>7} ({:>9}), varies {}..{}, {} windows, "
              "distance {:.2f} m".format(clock(s["first"]), clock(s["last"]), s["shift"],
                                         clock(s["shift"]), s["low"], s["high"], s["windows"],
                                         s["score"]))


def section_states(al, tactical):
    graded = tactical.in_periods()
    print("\n[4] what the broadcast shows at each graded tactical frame")
    print("  graded frames with a broadcast frame: {:.1%}".format(
        share(graded & (al.t_to_b >= 0), graded)))
    print("  all graded  {}".format(shares(al.state, graded)))
    for h, (a, b) in enumerate(tactical.periods, start=1):
        mask = np.zeros(tactical.n_frames, dtype=bool)
        mask[a:b] = True
        print("  half {}      {}".format(h, shares(al.state, mask)))
    for k in (REPLAY, OTHER):
        runs = runs_in(geometry.close_gaps(al.state == k, int(FPS)), tactical.periods, int(FPS))
        lengths = [(b - a) / FPS for a, b in runs]
        print("  {:<7} stretches of 1 s or more: {:>4}, length p10/p50/p90 {} s, longest {:.0f} s"
              .format(STATES[k], len(runs), pct(lengths, form="{:.1f}"),
                      max(lengths) if lengths else 0.0))


def section_agreement(al, tactical):
    graded = tactical.in_periods()
    live = graded & (al.state == LIVE)
    pitch = np.asarray(tactical.pitch)
    print("\n[5] how well the two trackings agree where the broadcast is live")
    print("  player distance on live frames (m) p10/p50/p90 {}".format(pct(al.distance[live])))
    both = live & tactical.ball_visible() & ~np.isnan(al.ball[:, 0])
    gap = (al.ball[both] - tactical.ball[both]) * pitch
    gap = np.sqrt((gap * gap).sum(axis=1))
    print("  both balls seen on {:.1%} of live frames; ball distance (m) p10/p50/p90 {}; within "
          "2 m {:.0%}, within 5 m {:.0%}".format(share(both, live), pct(gap),
                                                 share(gap < 2, len(gap)), share(gap < 5, len(gap))))


def section_ball(al, tactical):
    graded = tactical.in_periods()
    t_seen = tactical.ball_visible()
    b_seen = ~np.isnan(al.ball[:, 0])
    either = t_seen | b_seen
    print("\n[6] the ball on graded frames: tactical alone, and with the broadcast added")
    print("  seen by tactical {:.1%}, by the broadcast (live pictures) {:.1%}, by either {:.1%}"
          .format(share(graded & t_seen, graded), share(graded & b_seen, graded),
                  share(graded & either, graded)))
    print("  where the tactical ball is missing, the broadcast has it on {:.1%} of frames".format(
        share(graded & ~t_seen & b_seen, graded & ~t_seen)))
    frames, filled = np.zeros(len(GAP_BINS) - 1), np.zeros(len(GAP_BINS) - 1)
    for a, b in runs_in(~t_seen, tactical.periods):
        k = gap_bin(b - a)
        frames[k] += b - a
        filled[k] += b_seen[a:b].sum()
    names = gap_names()
    print("  {:<34}".format("tactical gap length") + "".join("{:>9}".format(n) for n in names))
    print("  {:<34}".format("  share of its frames filled") + "".join(
        "{:>9}".format("{:.0%}".format(f / n) if n else "-") for f, n in zip(filled, frames)))
    for title, seen in (("gaps per length, tactical alone", t_seen),
                        ("gaps per length, with broadcast", either)):
        counts = np.zeros(len(GAP_BINS) - 1, dtype=int)
        for a, b in runs_in(~seen, tactical.periods):
            counts[gap_bin(b - a)] += 1
        print("  {:<34}".format(title) + "".join("{:>9}".format(c) for c in counts))


def section_labels(al, tactical, codes, prob):
    graded = tactical.in_periods()
    print("\n[7] the broadcast against the labels of the current pipeline")
    for k, name in enumerate(STATES):
        m = graded & (al.state == k)
        print("  {:<8} {:>6.1%} of graded frames; we label them H {:>4.0%}  A {:>4.0%}  D {:>4.0%}"
              .format(name, share(m, graded), share(m & (codes == pm.H), m),
                      share(m & (codes == pm.A), m), share(m & (codes == pm.D), m)))
    dead = graded & (codes == pm.D)
    live = graded & ((codes == pm.H) | (codes == pm.A))
    print("  frames we label D    {}".format(shares(al.state, dead)))
    print("  frames we label H/A  {}".format(shares(al.state, live)))
    p_dead = np.nan_to_num(prob[:, 2], nan=0.5)
    sure_dead, sure_live = graded & (p_dead > 0.9), graded & (p_dead < 0.1)
    print("  model sure of dead (P > 0.9, {:.0%} of graded)  {}".format(
        share(sure_dead, graded), shares(al.state, sure_dead)))
    print("  model sure of live (P < 0.1, {:.0%} of graded)  {}".format(
        share(sure_live, graded), shares(al.state, sure_live)))
    print("  ratio of the two: " + "  ".join(
        "{} {}".format(name, "{:.2f}".format(share(sure_dead & (al.state == k), sure_dead)
                                             / share(sure_live & (al.state == k), sure_live))
                       if (sure_live & (al.state == k)).any() else "inf")
        for k, name in enumerate(STATES)))

    play = list(tactical.periods)
    for a, b in generate_submission.second_half_wait(tactical):   # dead by definition: leave out
        play = [(max(s, b), e) if s >= a and s < b else (s, e) for s, e in play]
    bad = (al.state == OTHER) | (al.state == REPLAY)
    stops = runs_in(codes == pm.D, play, int(2 * FPS))
    lags, leads, spills = [], [], []
    for a, b in stops:
        inner = np.flatnonzero(bad[a:b])
        if not len(inner):
            continue
        lags.append(inner[0] / FPS)
        leads.append((b - a - 1 - inner[-1]) / FPS)
        if bad[b - 1]:
            k = b
            while k < len(bad) and bad[k]:
                k += 1
            spills.append((k - b) / FPS)
    print("  our stoppages of 2 s or more after the kick-offs: {}; the broadcast leaves live "
          "pictures in {} of them ({:.0%})".format(len(stops), len(lags), share(len(lags), len(stops))))
    print("    from our stoppage start to the first non-live picture (s) p10/p50/p90  {}".format(
        pct(lags, form="{:.1f}")))
    print("    from the last non-live picture to our restart (s)         p10/p50/p90  {}".format(
        pct(leads, form="{:.1f}")))
    print("    still non-live when we restart play: {} stoppages, for another (s) p10/p50/p90 {}"
          .format(len(spills), pct(spills, form="{:.1f}")))

    lone, lone_p, soon = [], [], 0
    for a, b in runs_in(geometry.close_gaps(bad, int(FPS)), play, int(2 * FPS)):
        if (codes[a:b] == pm.D).any():
            continue
        lone.append((b - a) / FPS)
        lone_p.append(float(p_dead[a:b].mean()))
        soon += int((codes[max(0, a - int(15 * FPS)):a] == pm.D).any())
    print("  non-live stretches of 2 s or more that we label live throughout: {}, length (s) "
          "p10/p50/p90 {}; {} of them start within 15 s after one of our stoppages; model "
          "P(dead) over them p10/p50/p90 {}".format(len(lone), pct(lone, form="{:.1f}"), soon,
                                                   pct(lone_p)))

    pitch = np.asarray(tactical.pitch)
    ours = (codes == pm.H) | (codes == pm.A)
    for title, ball, rows in (
            ("tactical ball missing, broadcast ball seen", al.ball,
             np.flatnonzero(graded & ours & ~tactical.ball_visible() & ~np.isnan(al.ball[:, 0]))),
            ("for comparison, the tactical ball where seen", tactical.ball,
             np.flatnonzero(graded & ours & tactical.ball_visible()))):
        if not len(rows):
            print("  {}: no frames".format(title))
            continue
        near = geometry.team_distances(ball[rows] * pitch, tactical.players[rows] * pitch,
                                       tactical.team, k=1)[:, :, 0]
        close = near.min(axis=1) <= 3.0
        agree = (near.argmin(axis=1) == HOME) == (codes[rows] == pm.H)
        print("  {}: {} frames labelled H/A ({:.1%} of graded); a player within 3 m on {:.0%}; "
              "there the nearest player's team is our label on {:.0%}".format(
                  title, len(rows), share(len(rows), graded), close.mean(),
                  share(close & agree, close)))


def section_public(al, codes, truth_path):
    truth = pm.label_codes(pm.read_label_csv(truth_path))
    window = truth != pm.X
    print("\n[8] the public 5 minutes: a check, not a way to choose settings")
    dead = window & (truth == pm.D)
    live = window & ((truth == pm.H) | (truth == pm.A))
    print("  true dead frames  {}".format(shares(al.state, dead)))
    print("  true live frames  {}".format(shares(al.state, live)))
    if codes is not None:
        bad = (al.state == OTHER) | (al.state == REPLAY)
        missed, wrong = dead & (codes != pm.D), live & (codes == pm.D)
        print("  true dead frames we label live: {}, non-live pictures on {:.0%} of them".format(
            int(missed.sum()), share(missed & bad, missed)))
        print("  true live frames we label dead: {}, non-live pictures on {:.0%} of them".format(
            int(wrong.sum()), share(wrong & bad, wrong)))


def section_stand_in(al, tactical, path):
    """Only for stand-in bundles (tests/make_fake_bundle.py), which come with the true shift."""
    with np.load(path) as z:
        shift, shot = z["shift"], z["shot"]
    expected = broadcast.tactical_map(shift, tactical.n_frames)
    both = (expected >= 0) & (al.t_to_b >= 0)
    print("\n[stand-in] mapped {:.1%} of tactical frames (truth {:.1%}); within 2 frames of the "
          "true broadcast frame on {:.2%}".format((al.t_to_b >= 0).mean(), (expected >= 0).mean(),
                                                  share(both & (np.abs(al.t_to_b - expected) <= 2),
                                                        both)))
    kind = np.full(tactical.n_frames, -1)
    kind[both] = shot[expected[both]]
    for k, name in enumerate(("live", "replay", "no pitch")):
        print("  true {:<9} {}".format(name, shares(al.state, kind == k)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None,
                        help="data bundle root (default: ./data or $MTAH_DATA_DIR)")
    parser.add_argument("--no-model", action="store_true",
                        help="skip section 7, which runs the current pipeline for its labels")
    parser.add_argument("--no-cache", action="store_true", help="re-parse the ATD files")
    parser.add_argument("--truth", default=competition.PUBLIC_GT, help=argparse.SUPPRESS)
    parser.add_argument("--true-shift", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    started = time.time()
    tactical = competition.load_tracking(args.data_dir, use_cache=not args.no_cache)
    feed, valid, error = broadcast.load(args.data_dir, use_cache=not args.no_cache)
    section_files(feed, valid, error, tactical)

    print("\naligning the broadcast with the tactical recording ...")
    al = broadcast.align(tactical, feed, valid)
    print("  done after {:.0f} s".format(time.time() - started))
    section_coarse(al)
    section_fine(al)
    section_states(al, tactical)
    section_agreement(al, tactical)
    section_ball(al, tactical)

    codes = None
    if not args.no_model:
        print("\nrunning the current pipeline for its labels ...")
        tags = None
        try:
            tags, _ = competition.load_tags(tactical, args.data_dir)
        except IOError:
            print("  (no Smart Tagging file)")
        letters, prob = generate_submission.predict("possession", tactical,
                                                    model_module.DEFAULT_PATH, tags)
        codes = pm.label_codes(submission.finalize(letters, tactical.periods))
        section_labels(al, tactical, codes, prob)
    section_public(al, codes, args.truth)
    if args.true_shift:
        section_stand_in(al, tactical, args.true_shift)
    print("\ndone in {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
