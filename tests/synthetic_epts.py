"""Write small synthetic FIFA/EPTS files in the competition's documented format.

Used by the tests so the EPTS reader and Stage 0 can be checked without the gated match data.
The format follows DATA.md and what starter/baseline.py parses: groups "x,y,speed", ball "x,y,z",
"NaN" for a missing value, first <Team> = home.
"""

import numpy as np


def write_epts(folder, players, frames, team_ids=("T1", "T2"),
               team_names=("FC Barcelona", "Athletic Club"), layouts=None, with_spec=True,
               channels=("x", "y", "speed"), ball_channels=("x", "y", "z")):
    """Write meta.xml + raw.txt. Returns (meta_path, raw_path).

    players  [(player_id, team_id)] in <Player> order (= slot order)
    frames   [(frame_id, {player_id: (x, y, s)} , (bx, by, bz) or None)]
    layouts  [(start, end, [player_id, ...])]: which players appear, in which order, per frame
             range. Default: one layout with every player in <Player> order.
    """
    if layouts is None:
        first, last = frames[0][0], frames[-1][0]
        layouts = [(first, last, [pid for pid, _ in players])]

    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<main>", "<Metadata>",
             "<GlobalConfig><FrameRate>25</FrameRate></GlobalConfig>",
             "<Sessions><Session><FieldSize><Width>105</Width><Height>68</Height></FieldSize>"
             "</Session></Sessions>", "<Teams>"]
    for team_id, name in zip(team_ids, team_names):
        lines.append('<Team id="{}"><Name>{}</Name></Team>'.format(team_id, name))
    lines.append("</Teams><Players>")
    for slot, (pid, team_id) in enumerate(players, start=1):
        lines.append('<Player id="{}" teamId="{}"><Name>Track {}</Name>'
                     '<ShirtNumber>{}</ShirtNumber></Player>'.format(pid, team_id, slot, slot))
    lines.append("</Players><Devices><Device><Sensors><Sensor><Channels>")
    for channel in channels:
        lines.append('<Channel id="{}"><Name>{}</Name></Channel>'.format(channel, channel))
    lines.append("</Channels><PlayerChannels>")
    for pid, _ in players:
        for channel in channels:
            lines.append('<PlayerChannel id="{0}_{1}" playerId="{0}" channelId="{1}"/>'.format(
                pid, channel))
    lines.append("</PlayerChannels></Sensor></Sensors></Device></Devices></Metadata>")
    if with_spec:
        lines.append("<DataFormatSpecifications>")
        for start, end, pids in layouts:
            lines.append('<DataFormatSpecification separator=":" startFrame="{}" endFrame="{}">'
                         .format(start, end))
            lines.append('<StringRegister name="frameCount"/><SplitRegister separator=";">')
            for pid in pids:
                refs = "".join('<PlayerChannelRef playerChannelId="{}_{}"/>'.format(pid, c)
                               for c in channels)
                lines.append('<SplitRegister separator=",">{}</SplitRegister>'.format(refs))
            lines.append('</SplitRegister><SplitRegister separator=",">{}</SplitRegister>'.format(
                "".join('<BallChannelRef channelId="{}"/>'.format(c) for c in ball_channels)))
            lines.append("</DataFormatSpecification>")
        lines.append("</DataFormatSpecifications>")
    lines.append("</main>")

    meta_path = folder / "meta.xml"
    meta_path.write_text("\n".join(lines))

    def fmt(value):
        return "NaN" if value is None or (isinstance(value, float) and np.isnan(value)) \
            else "{:.5f}".format(value)

    raw_lines = []
    for frame_id, positions, ball in frames:
        pids = next(p for s, e, p in layouts if s <= frame_id <= e)
        groups = []
        for pid in pids:
            values = positions.get(pid)
            values = values if values is not None else (None,) * len(channels)
            groups.append(",".join(fmt(v) for v in values[:len(channels)]))
        ball_values = ball if ball is not None else (None,) * len(ball_channels)
        raw_lines.append("{}:{}:{}".format(frame_id, ";".join(groups),
                                           ",".join(fmt(v) for v in ball_values)))
    raw_path = folder / "raw.txt"
    raw_path.write_text("\n".join(raw_lines) + "\n")
    return str(meta_path), str(raw_path)


def random_match(n_frames, n_slots, seed=0, p_player_missing=0.3, p_ball_missing=0.4):
    """Random players/frames in write_epts' input format, with a couple of exact ties."""
    rng = np.random.default_rng(seed)
    players = [("P{}".format(i + 1), "T1" if i % 2 == 0 else "T2") for i in range(n_slots)]
    frames = []
    ball_hidden = False
    for frame_id in range(1, n_frames + 1):
        if rng.random() < 0.05:                     # ball missing in runs, not frame by frame
            ball_hidden = rng.random() < p_ball_missing
        ball = None if ball_hidden else (rng.random(), rng.random(), 0.0)
        positions = {}
        for pid, _ in players:
            if rng.random() >= p_player_missing:
                positions[pid] = (rng.random(), rng.random(), rng.random() * 8)
        if frame_id % 97 == 0 and len(positions) >= 2:   # an exact tie between two slots
            first, second = list(positions)[:2]
            positions[second] = positions[first]
        frames.append((frame_id, positions, ball))
    return players, frames
