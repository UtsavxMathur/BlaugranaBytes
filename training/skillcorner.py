"""Load a SkillCorner open-data match (public, MIT licence; see training/fetch.py) with labels.

SkillCorner's open data has 20 matches of the Australian A-League 2024/25: broadcast tracking at
10 frames per second, with the ball (and its height) and all 22 players on every frame where the
broadcast shows the pitch (players off screen are extrapolated by SkillCorner), and the player
and team in possession. Its phases of play cover exactly the time the ball is in play.

So the labels are:

    D       outside the phases of play
    H / A   inside them, the team in possession; while nobody has the ball the last team keeps
            it (a loose ball stays with the last team, as CHALLENGE.md says)
    X       outside the halves, and where the broadcast shows something else (a replay or a
            close-up) for longer than HOLE_S: never trained on, never scored

Shorter holes in the tracking are bridged with a straight line. Everything is then resampled to
25 frames per second, the rate of the competition tracking, by linear interpolation of the
positions and the nearest label.

Coordinates are metres from the centre spot, x along the pitch; we convert them to the 0..1
convention of Tracking.
"""

import csv
import json

import numpy as np

from possession import geometry
from possession.tracking import AWAY, HOME, Tracking

H, A, D, X = 0, 1, 2, 3
FPS_IN, FPS_OUT = 10, 25
HOLE_S = 3.0


def _bridge(values, missing, longest):
    """Linear interpolation over runs of `missing` rows no longer than `longest`, in place."""
    n = len(values)
    for start, length in zip(*geometry.runs_of(missing)):
        before, after = start - 1, start + length
        if length > longest or before < 0 or after >= n:
            continue
        share = (np.arange(1, length + 1) / float(length + 1)).reshape(
            (length,) + (1,) * (values.ndim - 1))
        values[start:after] = (1 - share) * values[before] + share * values[after]


def _resample(values, n_out):
    """10 fps -> 25 fps by linear interpolation along the first axis; NaN stays NaN."""
    source = np.arange(n_out) * (FPS_IN / float(FPS_OUT))
    low = np.floor(source).astype(np.int64)
    high = np.minimum(low + 1, len(values) - 1)
    share = (source - low).reshape((n_out,) + (1,) * (values.ndim - 1))
    return (1 - share) * values[low] + share * values[high]


def load(paths, verbose=True, seen_only=False):
    """(Tracking, labels) at 25 fps. paths: {"match.json": .., "phases_of_play.csv": ..,
    "tracking_extrapolated.jsonl": ..}.

    seen_only  keep only the players and ball the camera actually saw (is_detected), and drop
               SkillCorner's estimates for everything off screen.
    """
    meta = json.load(open(paths["match.json"]))
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    people = sorted(meta["players"], key=lambda p: (p["team_id"] != home_id, p["id"]))
    slot_of = {p["id"]: k for k, p in enumerate(people)}
    team = np.array([HOME if p["team_id"] == home_id else AWAY for p in people], dtype=np.int8)

    records = [json.loads(line) for line in open(paths["tracking_extrapolated.jsonl"])]
    n = max(r["frame"] for r in records) + 1
    ball = np.full((n, 2), np.nan)
    ball_z = np.full(n, np.nan)
    players = np.full((n, len(people), 2), np.nan)
    owner = np.full(n, -1, dtype=np.int8)
    shown = np.zeros(n, dtype=bool)          # the broadcast shows the pitch on this frame
    unknown_ids = 0
    for r in records:
        f = r["frame"]
        b = r.get("ball_data") or {}
        if b.get("x") is not None and b.get("y") is not None and (
                b.get("is_detected") or not seen_only):
            ball[f] = (b["x"], b["y"])
            if b.get("z") is not None:
                ball_z[f] = b["z"]
        for p in r.get("player_data") or []:
            k = slot_of.get(p.get("player_id"))
            if k is None:
                unknown_ids += 1
            elif p.get("x") is not None and p.get("y") is not None and (
                    p.get("is_detected") or not seen_only):
                players[f, k] = (p["x"], p["y"])
        shown[f] = bool(r.get("player_data"))
        owner[f] = {"home team": H, "away team": A}.get((r.get("possession") or {}).get("group"), -1)

    alive = np.zeros(n, dtype=bool)
    phase_team = np.full(n, -1, dtype=np.int8)
    with open(paths["phases_of_play.csv"]) as handle:
        for row in csv.DictReader(handle):
            a, b = int(row["frame_start"]), min(int(row["frame_end"]), n)
            alive[a:b] = True
            phase_team[a:b] = H if int(row["team_in_possession_id"]) == home_id else A

    halves = [(int(p["start_frame"]), min(int(p["end_frame"]) + 1, n)) for p in meta["match_periods"]]
    labels = np.full(n, X, dtype=np.int8)
    for a, b in halves:
        labels[a:b] = D
        state = -1
        for f in range(a, b):
            if not alive[f]:
                state = -1
                continue
            if owner[f] >= 0:
                state = owner[f]
            elif state < 0:
                state = phase_team[f]
            labels[f] = state

    longest = int(HOLE_S * FPS_IN)
    for values in (ball, ball_z, players):
        _bridge(values, ~shown, longest)
    gaps = ~shown.copy()
    for start, size in zip(*geometry.runs_of(~shown)):
        if size <= longest and start > 0 and start + size < n:
            gaps[start:start + size] = False         # a short cut away: the label stays
    labels[gaps] = X

    n_out = int((n - 1) * FPS_OUT / FPS_IN) + 1
    nearest = np.minimum(np.floor(np.arange(n_out) * (FPS_IN / float(FPS_OUT)) + 0.5).astype(
        np.int64), n - 1)
    scale = np.array([length, width])
    tracking = Tracking(
        ball=_resample(ball, n_out) / scale + 0.5,
        players=_resample(players, n_out) / scale + 0.5,
        team=team, pitch=(length, width),
        periods=[(int(round(a * FPS_OUT / FPS_IN)), min(int(round(b * FPS_OUT / FPS_IN)), n_out))
                 for a, b in halves],
        fps=FPS_OUT, team_names=(meta["home_team"]["short_name"], meta["away_team"]["short_name"]),
        ball_z=_resample(ball_z, n_out).astype(np.float32))
    out_labels = labels[nearest]
    if verbose:
        live = (out_labels == H) | (out_labels == A)
        print("  {} v {}: {} frames at 25 fps, {} players, ball in play {:.0%} of the halves, "
              "unknown {:.1%}{}".format(
                  tracking.team_names[0], tracking.team_names[1], n_out, len(people),
                  live[tracking.in_periods()].mean(),
                  (out_labels[tracking.in_periods()] == X).mean(),
                  "; {} player records not in the line-up".format(unknown_ids) if unknown_ids else ""))
    return tracking, out_labels
