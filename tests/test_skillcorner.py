"""The SkillCorner loader (training/skillcorner.py) on a tiny made-up match in SkillCorner's format."""

import json

import numpy as np

from training import skillcorner

H, A, D, X = 0, 1, 2, 3


def write_match(folder):
    """20 s at 10 fps. Home has the ball, then away; the ball is dead from frame 160; the
    broadcast shows no pitch for frames 50-59 (bridged) and from frame 165 on (unknown)."""
    meta = {"pitch_length": 105, "pitch_width": 68,
            "home_team": {"id": 10, "short_name": "Home"}, "away_team": {"id": 20, "short_name": "Away"},
            "players": [{"id": 1, "team_id": 10}, {"id": 2, "team_id": 20}],
            "match_periods": [{"period": 1, "start_frame": 0, "end_frame": 199}]}
    with open(folder / "m_match.json", "w") as handle:
        json.dump(meta, handle)
    with open(folder / "m_phases_of_play.csv", "w") as handle:
        handle.write("frame_start,frame_end,team_in_possession_id\n0,120,10\n120,160,20\n")
    with open(folder / "m_tracking_extrapolated.jsonl", "w") as handle:
        for f in range(200):
            shown = not (50 <= f < 60 or f >= 165)
            group = "home team" if f < 80 else None if f < 100 else "away team" if f < 150 else None
            record = {"frame": f, "period": 1,
                      "ball_data": {"x": float(f) / 10 if shown else None, "y": 0.0, "z": 0.1,
                                    "is_detected": shown},
                      "possession": {"player_id": None, "group": group},
                      "player_data": [{"player_id": 1, "x": 0.0, "y": 0.0, "is_detected": True},
                                      {"player_id": 2, "x": 10.0, "y": 5.0, "is_detected": True}]
                      if shown else []}
            handle.write(json.dumps(record) + "\n")
    return {name: str(folder / ("m_" + name)) for name in
            ("match.json", "phases_of_play.csv", "tracking_extrapolated.jsonl")}


def test_labels_holes_and_resampling(tmp_path):
    tracking, labels = skillcorner.load(write_match(tmp_path), verbose=False)
    assert tracking.n_frames == 498 and tracking.fps == 25           # 199 * 2.5 + 1
    at = lambda f10: labels[int(round(f10 * 2.5))]                   # noqa: E731
    assert at(0) == H and at(90) == H          # a loose ball stays with the last team
    assert at(100) == A and at(155) == A       # the frame's own team beats the phase's
    assert at(162) == D                        # outside the phases of play
    assert at(180) == X                        # no pitch shown and nothing to bridge to
    assert at(55) == H                         # a short hole is bridged ...
    middle = int(round(55 * 2.5))
    assert not np.isnan(tracking.players[middle, 0, 0])
    assert abs(tracking.ball[middle, 0] * 105 - 52.5 - 5.5) < 0.05   # ... along a straight line
    assert abs(tracking.players[0, 0, 0] - 0.5) < 1e-9 and tracking.team.tolist() == [0, 1]
