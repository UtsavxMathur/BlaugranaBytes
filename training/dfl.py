"""Load a match of the DFL open dataset (public; see training/fetch.py) with its frame labels.

The position file has one <FrameSet> per player per half and one for the ball. Each ball frame
carries the two flags this challenge is about:

    BallStatus      1 = ball in play, 0 = ball dead
    BallPossession  1 = home team has it, 2 = away team (kept even while the ball is dead)

so the label of a frame is D where BallStatus is 0, otherwise H or A from BallPossession -- the
same recipe CHALLENGE.md describes for the competition's ground truth ("D where the corrected
tracking marks the ball dead, otherwise the team the tracking names as owning it").

Coordinates are metres from the centre spot; we convert to the 0..1 convention of Tracking.
The file is one XML element per line, so it is read line by line (about 10x faster than an XML
parser on 400 MB).
"""

import re
import xml.etree.ElementTree as ET

import numpy as np

from possession.tracking import AWAY, HOME, Tracking

H, A, D, X = 0, 1, 2, 3  # label codes, same as evaluation/possession_metrics.py
SECTIONS = ("firstHalf", "secondHalf")
_ATTRS = re.compile(r'(\w+)="([^"]*)"')


def read_info(info_path):
    """(home_team_id, away_team_id, (home_name, away_name), (pitch_length, pitch_width))."""
    root = ET.parse(info_path).getroot()
    general = root.find(".//General")
    environment = root.find(".//Environment")
    return (general.get("HomeTeamId"), general.get("GuestTeamId"),
            (general.get("HomeTeamName"), general.get("GuestTeamName")),
            (float(environment.get("PitchX")), float(environment.get("PitchY"))))


def _read_framesets(positions_path, wanted_teams):
    """Yield (section, team_id, person_id, columns) with columns = dict of float arrays."""
    with open(positions_path, encoding="utf-8") as handle:
        header, rows, index = None, None, None
        for line in handle:
            if line.startswith("<Frame "):
                if header is None:
                    continue
                parts = line.split('"')
                if index is None:  # first frame of the set: find the attributes by name, once
                    # parts = ['<Frame N=', '10000', ' T=', '...', ' X=', '1.5', ...]
                    names = [p.split()[-1].rstrip("=") if p.strip() else "" for p in parts[0::2]]
                    wanted = ["N", "X", "Y"] + (["Z", "BallPossession", "BallStatus"]
                                                if header["TeamId"] == "BALL" else [])
                    index = [2 * names.index(name) + 1 for name in wanted]
                    rows = [[] for _ in index]
                for column, i in zip(rows, index):
                    column.append(parts[i])
            elif line.startswith("<FrameSet"):
                attrs = dict(_ATTRS.findall(line))
                header = attrs if attrs.get("TeamId") in wanted_teams else None
                rows, index = None, None
            elif line.startswith("</FrameSet") and header is not None and rows:
                keys = ["N", "X", "Y", "Z", "BallPossession", "BallStatus"][:len(rows)]
                yield (header["GameSection"], header["TeamId"], header["PersonId"],
                       {key: np.array(column, dtype=np.float64) for key, column in zip(keys, rows)})
                header = None


def load(info_path, positions_path, verbose=True):
    """(Tracking, labels) for one DFL match. labels: int8 codes per row (H/A/D, X if unknown)."""
    home_id, away_id, names, (length, width) = read_info(info_path)

    balls, people = {}, {}  # section -> ball columns ; (team, person) -> {section: columns}
    for section, team_id, person_id, columns in _read_framesets(
            positions_path, {home_id, away_id, "BALL"}):
        if section not in SECTIONS:
            continue  # no extra time in this dataset; ignore anything unexpected
        if team_id == "BALL":
            balls[section] = columns
        else:
            people.setdefault((team_id, person_id), {})[section] = columns

    # Rows: first half, then second half, back to back.
    first_n, offsets, periods, total = {}, {}, [], 0
    for section in SECTIONS:
        frames = balls[section]["N"].astype(np.int64)
        if not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
            raise ValueError("{}: ball frames of {} are not contiguous".format(
                positions_path, section))
        first_n[section], offsets[section] = int(frames[0]), total
        periods.append((total, total + len(frames)))
        total += len(frames)

    slots = sorted(people, key=lambda key: (key[0] != home_id, key[1]))  # home first
    ball = np.full((total, 2), np.nan)
    ball_z = np.full(total, np.nan, dtype=np.float32)
    labels = np.full(total, X, dtype=np.int8)
    players = np.full((total, len(slots), 2), np.nan)
    team = np.array([HOME if team_id == home_id else AWAY for team_id, _ in slots], dtype=np.int8)

    for section, (start, stop) in zip(SECTIONS, periods):
        columns = balls[section]
        ball[start:stop, 0] = columns["X"] / length + 0.5
        ball[start:stop, 1] = columns["Y"] / width + 0.5
        ball_z[start:stop] = columns["Z"]
        alive = columns["BallStatus"] == 1
        part = np.full(stop - start, X, dtype=np.int8)
        part[alive & (columns["BallPossession"] == 1)] = H
        part[alive & (columns["BallPossession"] == 2)] = A
        part[~alive] = D
        labels[start:stop] = part

    for slot, key in enumerate(slots):
        for section, columns in people[key].items():
            start, stop = periods[SECTIONS.index(section)]
            rows = columns["N"].astype(np.int64) - first_n[section] + offsets[section]
            keep = (rows >= start) & (rows < stop)
            players[rows[keep], slot, 0] = columns["X"][keep] / length + 0.5
            players[rows[keep], slot, 1] = columns["Y"][keep] / width + 0.5

    if verbose:
        print("  {} v {}: {} frames, {} player slots, pitch {} x {} m".format(
            names[0], names[1], total, len(slots), length, width))
    tracking = Tracking(ball=ball, players=players, team=team, pitch=(length, width),
                        periods=periods, fps=25, team_names=names, ball_z=ball_z)
    return tracking, labels
