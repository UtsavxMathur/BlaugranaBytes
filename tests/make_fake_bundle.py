"""Build a stand-in for the competition bundle out of a PUBLIC game.

    python tests/make_fake_bundle.py <folder> [--game metrica-3]
    python generate_submission.py --data-dir <folder> ...
    python scripts/profile_atd.py --data-dir <folder>

Same file names, frame count, half positions and formats as the real bundle (DATA.md), but the
content is a public game, so the whole pipeline can be run end to end without the gated data. It
also writes the labels of the public game as truth.csv, to score the pipeline's output against.

Not a pytest test: it writes ~200 MB and takes about a minute.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import competition, submission  # noqa: E402
from training import dataset  # noqa: E402

N_SLOTS = 46
KICKOFF_2 = 113742           # published second-half kick-off (FAQ.md)
TEAM_TAGS = ("Barcelona", "Atletic Club")   # the spellings DATA.md lists for the Team tag


def place(game, render=True):
    """(ball, players, team, labels) on the competition's frame grid.

    render=True first passes the game through the simulator, so the stand-in has the same kind
    of ball gaps and smoothing as the real tracking.
    """
    n = competition.n_frames()
    (h1, e1), (h2, e2) = competition.periods()
    tracking, labels = game.tracking, game.labels
    if render:
        from training import simulate
        tracking = simulate.degrade(tracking, labels, simulate.VALIDATION["typical"], seed=0)
    order = np.argsort(tracking.team, kind="stable")[:N_SLOTS]
    ball = np.full((n, 2), np.nan)
    players = np.full((n, N_SLOTS, 2), np.nan)
    truth = np.full(n, 3, dtype=np.int8)
    truth[h1:e1] = 2
    truth[h2:e2] = 2                                   # anything not covered below is dead ball
    for (src_start, src_stop), (dst_start, dst_stop) in zip(tracking.periods,
                                                           ((h1, e1), (KICKOFF_2, e2))):
        m = min(src_stop - src_start, dst_stop - dst_start)
        ball[dst_start:dst_start + m] = tracking.ball[src_start:src_start + m]
        players[dst_start:dst_start + m, :len(order)] = tracking.players[src_start:src_start + m][:, order]
        part = labels[src_start:src_start + m].copy()
        part[part == 3] = 2
        truth[dst_start:dst_start + m] = part
    team = np.full(N_SLOTS, 1, dtype=np.int8)
    team[:len(order)] = tracking.team[order]
    return ball, players, team, truth


def write_atd(folder, ball, players, team):
    ids = ["T{}".format(i + 1) for i in range(N_SLOTS)]
    meta = ['<?xml version="1.0" encoding="utf-8"?>', "<main><Metadata>",
            "<GlobalConfig><FrameRate>25</FrameRate></GlobalConfig>",
            "<Sessions><Session><FieldSize><Width>105</Width><Height>68</Height></FieldSize>"
            "</Session></Sessions>",
            '<Teams><Team id="BAR"><Name>FC Barcelona</Name></Team>'
            '<Team id="ATH"><Name>Athletic Club</Name></Team></Teams><Players>']
    meta += ['<Player id="{}" teamId="{}"><Name>Track {}</Name><ShirtNumber>{}</ShirtNumber></Player>'
             .format(pid, "BAR" if side == 0 else "ATH", i + 1, i + 1)
             for i, (pid, side) in enumerate(zip(ids, team))]
    meta.append("</Players><Devices><Device><Sensors><Sensor><PlayerChannels>")
    for pid in ids:
        for channel in ("x", "y", "speed"):
            meta.append('<PlayerChannel id="{0}_{1}" playerId="{0}" channelId="{1}"/>'.format(pid, channel))
    meta.append("</PlayerChannels></Sensor></Sensors></Device></Devices></Metadata>"
                '<DataFormatSpecifications><DataFormatSpecification separator=":" startFrame="1" '
                'endFrame="{}"><StringRegister name="frameCount"/><SplitRegister separator=";">'
                .format(len(ball)))
    for pid in ids:
        meta.append('<SplitRegister separator=",">' + "".join(
            '<PlayerChannelRef playerChannelId="{}_{}"/>'.format(pid, c) for c in ("x", "y", "speed"))
            + "</SplitRegister>")
    meta.append('</SplitRegister><SplitRegister separator=","><BallChannelRef channelId="x"/>'
                '<BallChannelRef channelId="y"/><BallChannelRef channelId="z"/></SplitRegister>'
                "</DataFormatSpecification></DataFormatSpecifications></main>")
    with open(os.path.join(folder, "BAR-ATH_tactical_atd.xml"), "w") as handle:
        handle.write("\n".join(meta))

    empty = "NaN,NaN,NaN"
    with open(os.path.join(folder, "BAR-ATH_tactical_atd.txt"), "w") as handle:
        for f in range(len(ball)):
            row = players[f]
            groups = [empty if x != x else "{:.5f},{:.5f},0.00".format(x, y) for x, y in row]
            b = empty if ball[f, 0] != ball[f, 0] else "{:.5f},{:.5f},0.00".format(*ball[f])
            handle.write("{}:{}:{}\n".format(f + 1, ";".join(groups), b))


def write_tags(folder, truth, seed=0):
    """A plausible Smart Tagging file derived from the labels: kick-offs, some restarts, phases."""
    rng = np.random.default_rng(seed)
    fps = 25.0
    rows = []

    def add(code, start, stop, team=None, kind=None, half=None):
        labels = [("Team", "N/A" if team is None else TEAM_TAGS[team])]
        if half is not None:
            labels.append(("Half", half))
        if kind is not None:
            labels.append(("Type", kind))
        rows.append((start / fps, stop / fps, code, labels))

    (h1, e1), (_, e2) = competition.periods()
    change = np.flatnonzero(np.diff(truth) != 0) + 1
    bounds = np.concatenate(([0], change, [len(truth)]))
    for i, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        state = truth[start]
        half = "1st Half" if start < e1 else "2nd Half"
        # Timing as in the real file: a kick-off tag is 5 s long and starts about 3.4 s before
        # the kick-off; other set-piece tags are 35 s long and start about 6 s before the restart.
        if state in (0, 1) and start in (h1, KICKOFF_2):
            add("SET PIECES", start - 85, start + 40, None, "Kick Off (Start)", half)
        elif state in (0, 1) and i > 0 and truth[start - 1] == 2 and rng.random() < 0.5:
            kind = rng.choice(["Throw-in", "Corner Kick", "Goal Kick", "Free Kick"], p=[0.5, 0.2, 0.2, 0.1])
            begin = start - int(rng.uniform(4.0, 8.0) * 25)
            add("SET PIECES", begin, begin + 875, None if kind == "Throw-in" else int(state), kind, half)
        if state in (0, 1) and stop - start > 200 and rng.random() < 0.6:
            add("PROGRESSION", start + 25, stop, int(state), half=half)
            if rng.random() < 0.5:
                add("DEFENDING IN MIDDLE THIRD", start + 25, stop, 1 - int(state), half=half)

    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<file><ALL_INSTANCES>"]
    for index, (start, stop, code, labels) in enumerate(rows, start=1):
        lines.append("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>{}"
                     "</instance>".format(index, start, stop, code, "".join(
                         "<label><group>{}</group><text>{}</text></label>".format(g, t)
                         for g, t in labels)))
    lines.append("</ALL_INSTANCES></file>")
    with open(os.path.join(folder, "BAR-ATH_tactical_smart_tagging.xml"), "w") as handle:
        handle.write("\n".join(lines))
    return len(rows)


def write_homography(folder, ball, view_m=70.0, image=(1920, 1080), lag=10):
    """A made-up camera that follows the ball: a top-down view `view_m` metres wide, centred on
    the last known ball position, smoothed over `lag` frames. Pitch (0..1) -> pixels, as DATA.md."""
    length, width = 105.3, 68.0
    n = len(ball)
    focus = ball * (length, width)
    known = ~np.isnan(focus[:, 0])
    index = np.maximum.accumulate(np.where(known, np.arange(n), 0))
    focus = focus[index]
    focus[~np.maximum.accumulate(known)] = (length / 2, width / 2)
    kernel = np.ones(lag) / lag
    focus = np.stack([np.convolve(focus[:, k], kernel, mode="same") for k in (0, 1)], axis=1)
    w, h = image
    sx, sy = w / view_m, h / (view_m * h / w)
    with open(os.path.join(folder, "BAR-ATH_tactical_homography.csv"), "w") as handle:
        handle.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        for f in range(n):
            fx, fy = focus[f]
            handle.write("{},{:.6g},0,{:.6g},0,{:.6g},{:.6g},0,0,1,0.25,1\n".format(
                f, length * sx, w / 2 - fx * sx, width * sy, h / 2 - fy * sy))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("--game", default="metrica-3", choices=dataset.ALL_GAMES)
    parser.add_argument("--clean", action="store_true",
                        help="keep the public game's clean tracking (default: render it like the "
                             "real tracker, see training/simulate.py)")
    args = parser.parse_args()

    folder = os.path.join(args.folder, "tactical")
    os.makedirs(folder, exist_ok=True)
    ball, players, team, truth = place(dataset.load(args.game), render=not args.clean)
    write_atd(folder, ball, players, team)
    write_homography(folder, ball)
    n_tags = write_tags(folder, truth)
    submission.write(os.path.join(args.folder, "truth.csv"), submission.to_letters(truth))
    print("wrote a stand-in bundle from {} to {}  ({} tags)".format(args.game, args.folder, n_tags))


if __name__ == "__main__":
    main()
