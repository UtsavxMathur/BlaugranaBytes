"""Load a Metrica sample game (public; see training/fetch.py) and derive frame labels from events.

Games 1 and 2: one wide CSV per team (3 header rows), events as CSV.
Game 3:        FIFA/EPTS -- the same format family as the competition ATD -- events as JSON.

These games have no frame labels, so we build PROXY labels from the event data. They follow
CHALLENGE.md as closely as events allow, and they are noisiest at hand-overs and restarts:

  * A team has the ball from the start of any event where one of its players controls it:
    PASS, CARRY, SHOT, RECOVERY, SET PIECE -- and its own BALL LOST / BALL OUT, because a team
    has the ball right up to the moment it loses it.
  * Possession changes only when the OTHER team gets one of those events. A CHALLENGE alone
    changes nothing (a duel without control is not a change of possession).
  * The ball is dead
      - from the END frame of a BALL OUT, and of any event whose subtype contains OUT, GOAL,
        OFFSIDE, HAND BALL or END HALF (shot wide, goal, offside call, ...);
      - from the START frame of FAULT RECEIVED and CARD (the whistle);
    and stays dead until the next control event (normally the SET PIECE restart).
  * Rows before the first and after the last event of a half are X (unknown, never scored).
"""

import json

import numpy as np
import pandas as pd

from possession import epts
from possession.tracking import AWAY, HOME, Tracking

H, A, D, X = 0, 1, 2, 3  # label codes, same as evaluation/possession_metrics.py
PITCH = (105.0, 68.0)

CONTROL = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "BALL OUT"}
DEAD_AT_START = {"FAULT RECEIVED", "CARD"}
DEAD_AT_END_TYPES = {"BALL OUT"}
# Matched against whole '-'-separated subtype tokens, so "GOAL KICK" does not match "GOAL".
DEAD_AT_END_TOKENS = {"OUT", "GOAL", "OFFSIDE", "HAND BALL", "END HALF"}


# -- tracking ------------------------------------------------------------------------------------

def _read_team_csv(path):
    """(frames, players (n, p, 2), ball (n, 2)) from one Metrica wide tracking CSV."""
    names = pd.read_csv(path, skiprows=2, nrows=1, header=None).iloc[0].tolist()
    data = pd.read_csv(path, skiprows=3, header=None).to_numpy(dtype=np.float64)
    frames = data[:, 1].astype(np.int64)
    columns = [i for i, name in enumerate(names) if isinstance(name, str)
               and name.startswith("Player")]
    ball = names.index("Ball")
    players = np.stack([data[:, [c, c + 1]] for c in columns], axis=1)
    return frames, players, data[:, [ball, ball + 1]]


def load_csv_tracking(home_path, away_path):
    home_frames, home_xy, ball = _read_team_csv(home_path)
    away_frames, away_xy, _ = _read_team_csv(away_path)
    if not np.array_equal(home_frames, away_frames):
        raise ValueError("home and away tracking have different frames")
    rows = home_frames - 1
    n_frames = int(rows.max()) + 1
    out_ball = np.full((n_frames, 2), np.nan)
    out_players = np.full((n_frames, home_xy.shape[1] + away_xy.shape[1], 2), np.nan)
    out_ball[rows] = ball
    out_players[rows] = np.concatenate([home_xy, away_xy], axis=1)
    team = np.array([HOME] * home_xy.shape[1] + [AWAY] * away_xy.shape[1], dtype=np.int8)
    return Tracking(ball=out_ball, players=out_players, team=team, pitch=PITCH,
                    periods=[(0, n_frames)], fps=25, team_names=("Home", "Away"))


# -- events --------------------------------------------------------------------------------------

def load_csv_events(path):
    raw = pd.read_csv(path)
    return pd.DataFrame({
        "period": raw["Period"].astype(int),
        "start": raw["Start Frame"].astype(int),
        "end": raw["End Frame"].astype(int),
        "type": raw["Type"].astype(str).str.strip().str.upper(),
        "subtype": raw["Subtype"].fillna("").astype(str).str.strip().str.upper(),
        "team": raw["Team"].astype(str).str.strip().map({"Home": "H", "Away": "A"}),
    })


def load_json_events(path, home_team_id):
    with open(path) as handle:
        data = json.load(handle)["data"]
    rows = []
    for event in data:
        subtypes = event.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        team_id = (event.get("team") or {}).get("id")
        start = int(event["start"]["frame"])
        end = (event.get("end") or {}).get("frame")
        rows.append({
            "period": int(event["period"]), "start": start,
            "end": int(end) if end is not None else start,
            "type": str(event["type"]["name"]).strip().upper(),
            "subtype": "-".join(str(s.get("name", "")).strip().upper() for s in subtypes),
            "team": None if team_id is None else ("H" if team_id == home_team_id else "A"),
        })
    return pd.DataFrame(rows)


def labels_from_events(events, n_frames):
    """(labels, periods): int8 codes per row, and the [(start, stop)] rows of each half.

    Event frames are the file's own 1-based frame numbers, so row = frame - 1.
    """
    events = events.copy()
    events["order"] = np.arange(len(events))
    labels = np.full(n_frames, X, dtype=np.int8)
    code = {"H": H, "A": A}

    bounds = []
    for period, group in events.groupby("period", sort=True):
        first = int(group["start"].min()) - 1
        last = int(np.maximum(group["end"], group["start"]).max()) - 1
        bounds.append([period, max(0, first), min(n_frames, last + 1)])
    for i in range(1, len(bounds)):
        if bounds[i][1] < bounds[i - 1][2]:                 # overlapping halves: cut the earlier
            bounds[i - 1][2] = bounds[i][1]

    periods = []
    for period, run_start, run_stop in bounds:
        group = events[events["period"] == period].sort_values(["start", "order"])
        points = []  # (row, tie-break, new state)
        for k, event in enumerate(group.itertuples(index=False)):
            row_start = int(event.start) - 1
            row_end = int(max(event.end, event.start)) - 1
            tokens = set(event.subtype.split("-")) if event.subtype else set()
            if event.type in DEAD_AT_START:
                points.append((row_start, k, D))
                continue
            if event.type in CONTROL and event.team in code:
                points.append((row_start, k, code[event.team]))
            if event.type in DEAD_AT_END_TYPES or tokens & DEAD_AT_END_TOKENS:
                points.append((row_end, k + 0.5, D))
        points.sort(key=lambda point: (point[0], point[1]))
        state, cursor = D, run_start
        for row, _, new_state in points:
            row = min(max(row, run_start), run_stop)
            if row > cursor:
                labels[cursor:row] = state
                cursor = row
            state = new_state
        labels[cursor:run_stop] = state
        periods.append((run_start, run_stop))
    return labels, periods


# -- one call per game ---------------------------------------------------------------------------

def load(game, paths, verbose=True):
    """(Tracking, labels) for sample game 1, 2 or 3. paths: {file name: path} from fetch.py."""
    def find(fragment):
        return next(path for name, path in paths.items() if fragment in name)

    if game in (1, 2):
        tracking = load_csv_tracking(find("Home_Team"), find("Away_Team"))
        events = load_csv_events(find("RawEventsData"))
    else:
        meta_path = find("metadata")
        tracking = epts.load(meta_path, find("tracking"), pitch=PITCH, verbose=verbose)
        events = load_json_events(find("events"), epts.read_metadata(meta_path).teams[0][0])
    labels, periods = labels_from_events(events, tracking.n_frames)
    return tracking.replace(periods=periods), labels
