"""Print summary statistics of the competition tracking. YOU run this, on your own machine.

    python scripts/profile_atd.py

Why: the model is trained on public games whose tracking is clean. To train it for the real
thing we damage that clean tracking the way the competition tracking is damaged -- and for that
we need to know HOW the real tracking fails: how long the ball gaps are, where the ball is when
it is lost, how many players are tracked, how the Smart Tagging lines up.

What it prints: counts, percentages and medians over the whole match. It never prints a frame, a
position, or anything that could rebuild the data. The same text is saved to
outputs/atd_profile.txt (gitignored). Whether you paste it to an AI assistant is your decision
under the competition's terms.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import competition, profile, smart_tagging  # noqa: E402
from possession.tracking import show_path  # noqa: E402

# Published in DATA.md / FAQ.md / starter/load_smart_tagging.py: the frame each half kicks off on.
PUBLISHED_KICKOFFS = {0: 11185, 1: 113742}


def kickoff_lines(tags, fps):
    """Do the 'Kick Off (Start)' tags contain the published kick-off frames?"""
    lines = ["\n[9] kick-off tags vs the published kick-off frames"]
    starts = [t for t in tags if t.code == smart_tagging.SET_PIECES
              and t.kind == smart_tagging.KICK_OFF_START]
    for period, frame in sorted(PUBLISHED_KICKOFFS.items()):
        mine = [t for t in starts if t.half.startswith("2") == (period == 1)]
        if not mine:
            lines.append("    half {}: no Kick Off (Start) tag".format(period + 1))
            continue
        tag = mine[0]
        lines.append("    half {}: tag starts {:+.1f} s and ends {:+.1f} s relative to the kick-off; "
                     "contains it: {}".format(period + 1, (tag.start - frame) / fps,
                                              (tag.stop - frame) / fps,
                                              tag.start <= frame < tag.stop))
    after_goal = [t for t in tags if t.kind == smart_tagging.KICK_OFF_GOAL]
    goals = [t for t in tags if t.code == "goals"]
    lines.append("    {} 'Kick Off (After Goal)' tags, {} GOALS tags".format(len(after_goal), len(goals)))
    for goal in goals:
        later = [t.start for t in after_goal if t.start >= goal.start]
        if later:
            lines.append("    a GOALS tag lasts {:.1f} s; the next kick-off tag starts {:.1f} s after "
                         "it starts".format((goal.stop - goal.start) / fps,
                                            (min(later) - goal.start) / fps))
    return lines


def dead_window_lines(tracking):
    """The tracking in stretches where the ball is certainly not in play.

    These windows are published (DATA.md, FAQ.md): before the match, halftime, after the final
    whistle, and the start of the graded second half before its kick-off. They show how the ball
    detector behaves when nothing is being played.
    """
    (h1, e1), (h2, e2) = tracking.periods
    windows = [("before the match", 0, h1), ("halftime", e1, h2),
               ("2nd half before kick-off", h2, PUBLISHED_KICKOFFS[1]),
               ("after the final whistle", e2, tracking.n_frames)]
    lines = ["\n[10] stretches with no play (published frame ranges)",
             "    {:<26} {:>8} {:>9} {:>20} {:>17} {:>17}".format(
                 "window", "seconds", "ball vis", "vis. runs s p50/90", "tempo q25/50/75",
                 "players q25/50/75")]
    for name, start, stop in windows:
        if stop <= start:
            continue
        w = profile.measure_window(tracking, start, stop)
        lines.append("    {:<26} {:>8.0f} {:>9.1%} {:>10.1f} {:>8.1f}   {:>5.1f} {:>5.1f} {:>5.1f}   "
                     "{:>5.0f} {:>5.0f} {:>5.0f}".format(name, w["seconds"], w["ball_visible"],
                                                          *w["visible_runs_s"], *w["tempo"], *w["players"]))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "atd_profile.txt"))
    args = parser.parse_args()

    tracking = competition.load_tracking(args.data_dir)
    try:
        tags, _ = competition.load_tags(tracking, args.data_dir)
    except IOError as error:
        print("  (no Smart Tagging file: {})".format(str(error).splitlines()[0]))
        tags = None

    text = profile.render(profile.measure(tracking, tags))
    if tags is not None:
        text += "\n" + "\n".join(kickoff_lines(tags, tracking.fps))
    text += "\n" + "\n".join(dead_window_lines(tracking))
    try:
        matrices, valid = competition.load_homography(args.data_dir)
        text += "\n" + profile.render_camera(profile.measure_camera(tracking, matrices, valid))
    except IOError as error:
        text += "\n\n[11] tactical camera: no homography file ({})".format(str(error).splitlines()[0])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write(text + "\n")
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)
    print("\nsaved to {}".format(show_path(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
