"""Read a FIFA/EPTS tracking export (metadata XML + raw text) into a Tracking.

The competition ATD and Metrica's public Sample Game 3 are both in this format. The raw file has
one line per frame:

    <frame_id>:<group>;<group>;...;<group>:<ball>          each <group> is "x,y[,speed]"

The metadata's DataFormatSpecification says which track slot each ';'-group belongs to and which
field of a group is x, y or speed. A file can hold several specifications for different frame
ranges (Sample Game 3 changes layout at every substitution), so the layout is looked up per frame.
If there is no specification at all we fall back to what the organisers' starter/baseline.py
assumes: groups in <Player> order, fields x, y, speed.

Teams: the first <Team> in the metadata is the home side (FIFA/EPTS convention, and what the
official baseline assumes). generate_submission.py double-checks that home is Barcelona.

Frame ids count from 1, so row = frame_id - 1 (DATA.md, "Frame alignment").
"""

import bisect
import dataclasses
import xml.etree.ElementTree as ET

import numpy as np

from .tracking import AWAY, HOME, Tracking

FRAME_ID_BASE = 1


@dataclasses.dataclass
class Layout:
    """How to read the lines whose frame_id is in [start, end]."""
    start: float
    end: float
    sep: str             # between the frame / players / ball parts, normally ':'
    group_sep: str       # between player groups, normally ';'
    field_sep: str       # inside a group, normally ','
    i_frame: int         # index of each part in line.split(sep)
    i_players: int
    i_ball: int
    slots: np.ndarray    # track slot of each player group, in line order
    width: int           # fields per player group
    ix: int              # field index of x / y / speed inside a group (speed may be None)
    iy: int
    ispeed: object
    bx: int              # field index of x / y inside the ball group
    by: int


@dataclasses.dataclass
class Metadata:
    teams: list          # [(team_id, name)], file order: home first
    players: list        # [(player_id, team_id)], file order = slot order
    layouts: list        # [Layout] sorted by start
    pitch: object        # (length_m, width_m) from FieldSize, or None
    fps: int
    notes: list          # warnings worth printing


def _kind(channel_id):
    """'x', 'y', 'speed' or None from an EPTS channel id."""
    name = (channel_id or "").strip().lower()
    if name in ("x", "position_x"):
        return "x"
    if name in ("y", "position_y"):
        return "y"
    if name == "s" or "speed" in name:
        return "speed"
    return None


def _layout_from_spec(spec, slot_of_player, channels):
    sep = spec.get("separator", ":")
    i_frame = i_players = i_ball = None
    players_reg = ball_reg = None
    for index, child in enumerate(list(spec)):
        if child.tag == "StringRegister":
            i_frame = index
        elif child.tag == "SplitRegister":
            if child.find(".//PlayerChannelRef") is not None:
                i_players, players_reg = index, child
            elif child.find(".//BallChannelRef") is not None:
                i_ball, ball_reg = index, child
    if i_frame is None or players_reg is None:
        raise ValueError("DataFormatSpecification without a frame register or player groups")

    slots, kinds_per_group, field_sep = [], [], ","
    for group in (g for g in players_reg if g.tag == "SplitRegister"):
        field_sep = group.get("separator", field_sep)
        refs = [r.get("playerChannelId") for r in group if r.tag == "PlayerChannelRef"]
        owners = {channels[ref][0] for ref in refs}
        if len(owners) != 1:
            raise ValueError("a player group mixes channels of {} players".format(len(owners)))
        slots.append(slot_of_player[owners.pop()])
        kinds_per_group.append([_kind(channels[ref][1]) for ref in refs])
    if not slots:
        raise ValueError("player register has no per-player groups")
    kinds = kinds_per_group[0]
    if any(k != kinds for k in kinds_per_group):
        raise NotImplementedError("player groups with different field layouts in one spec")
    if "x" not in kinds or "y" not in kinds:
        raise ValueError("player groups have no x/y channels: {}".format(kinds))

    bx, by = 0, 1
    if ball_reg is not None:
        ball_kinds = [_kind(r.get("channelId")) for r in ball_reg.iter("BallChannelRef")]
        if "x" in ball_kinds and "y" in ball_kinds:
            bx, by = ball_kinds.index("x"), ball_kinds.index("y")

    return Layout(start=float(spec.get("startFrame", "-inf")),
                  end=float(spec.get("endFrame", "inf")), sep=sep,
                  group_sep=players_reg.get("separator", ";"), field_sep=field_sep,
                  i_frame=i_frame, i_players=i_players,
                  i_ball=i_ball if i_ball is not None else -1,
                  slots=np.array(slots, dtype=np.int64), width=len(kinds),
                  ix=kinds.index("x"), iy=kinds.index("y"),
                  ispeed=kinds.index("speed") if "speed" in kinds else None, bx=bx, by=by)


def _fallback_layout(n_players, first_line):
    """No DataFormatSpecification: the convention starter/baseline.py relies on."""
    width = len(first_line.strip().split(":")[1].split(";")[0].split(","))
    return Layout(start=float("-inf"), end=float("inf"), sep=":", group_sep=";", field_sep=",",
                  i_frame=0, i_players=1, i_ball=2, slots=np.arange(n_players), width=width,
                  ix=0, iy=1, ispeed=2 if width >= 3 else None, bx=0, by=1)


def read_metadata(meta_path, raw_path=None):
    root = ET.parse(meta_path).getroot()
    for element in root.iter():                      # drop XML namespaces
        if isinstance(element.tag, str) and "}" in element.tag:
            element.tag = element.tag.split("}", 1)[1]

    teams = [(t.get("id"), (t.findtext("Name") or "").strip()) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 <Team> entries, found {}".format(meta_path, len(teams)))
    players = [(p.get("id"), p.get("teamId")) for p in root.iter("Player")]
    if not players:
        raise ValueError("{}: no <Player> entries".format(meta_path))
    slot_of_player = {player_id: slot for slot, (player_id, _) in enumerate(players)}
    channels = {c.get("id"): (c.get("playerId"), c.get("channelId"))
                for c in root.iter("PlayerChannel")}

    notes = []
    specs = list(root.iter("DataFormatSpecification"))
    if specs:
        layouts = sorted((_layout_from_spec(s, slot_of_player, channels) for s in specs),
                         key=lambda layout: layout.start)
    else:
        if raw_path is None:
            raise ValueError("no DataFormatSpecification and no raw file to infer a layout from")
        with open(raw_path) as handle:
            first_line = next(line for line in handle if line.strip())
        layouts = [_fallback_layout(len(players), first_line)]
        notes.append("metadata has no DataFormatSpecification; assuming <Player> order and "
                     "fields x,y[,speed] like starter/baseline.py")

    pitch = None
    field = root.find(".//FieldSize")
    if field is not None and field.findtext("Width") and field.findtext("Height"):
        pitch = (float(field.findtext("Width")), float(field.findtext("Height")))
    fps = int(float(root.findtext(".//FrameRate") or 25))
    return Metadata(teams=teams, players=players, layouts=layouts, pitch=pitch, fps=fps,
                    notes=notes)


def load(meta_path, raw_path, n_frames=None, pitch=None, periods=None, verbose=True):
    """Parse a FIFA/EPTS pair into a Tracking (row = frame_id - 1).

    n_frames  length of the output arrays; default = highest frame_id in the file.
    pitch     (length_m, width_m); default = FieldSize from the metadata, else 105 x 68.
    periods   [(start, stop)] rows of the halves; default = the whole file as one period.
    """
    meta = read_metadata(meta_path, raw_path)
    layouts = meta.layouts
    starts = [layout.start for layout in layouts]
    sep, i_frame = layouts[0].sep, layouts[0].i_frame
    if any(layout.i_frame != i_frame or layout.sep != sep for layout in layouts):
        raise NotImplementedError("layouts disagree on where the frame id is")

    if n_frames is None:
        with open(raw_path) as handle:
            n_frames = max(int(line.split(sep, i_frame + 1)[i_frame])
                           for line in handle if line.strip()) - FRAME_ID_BASE + 1

    home_id = meta.teams[0][0]
    team = np.array([HOME if team_id == home_id else AWAY for _, team_id in meta.players],
                    dtype=np.int8)
    n_slots = len(meta.players)
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    skipped = {"outside every layout's frame range": 0, "with a frame_id beyond n_frames": 0,
               "with the wrong number of fields": 0}
    seen = np.zeros(n_frames, dtype=bool)
    layout = layouts[0]
    with open(raw_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(sep)
            frame_id = int(parts[i_frame])
            if not layout.start <= frame_id <= layout.end:
                index = bisect.bisect_right(starts, frame_id) - 1
                if index < 0 or not layouts[index].start <= frame_id <= layouts[index].end:
                    skipped["outside every layout's frame range"] += 1
                    continue
                layout = layouts[index]
            row = frame_id - FRAME_ID_BASE
            if not 0 <= row < n_frames:
                skipped["with a frame_id beyond n_frames"] += 1
                continue

            # The whole players part as one flat float array, then reshape. numpy parses "NaN"
            # to nan, so missing players need no special case.
            fields = parts[layout.i_players].replace(layout.group_sep, layout.field_sep)
            values = np.array(fields.split(layout.field_sep), dtype=np.float64)
            if values.size != len(layout.slots) * layout.width:
                skipped["with the wrong number of fields"] += 1
                continue
            values = values.reshape(len(layout.slots), layout.width)
            players[row, layout.slots, 0] = values[:, layout.ix]
            players[row, layout.slots, 1] = values[:, layout.iy]
            if layout.ispeed is not None:
                speed[row, layout.slots] = values[:, layout.ispeed]

            if 0 <= layout.i_ball < len(parts):
                ball_fields = parts[layout.i_ball].split(layout.field_sep)
                if len(ball_fields) > max(layout.bx, layout.by):
                    ball[row, 0] = float(ball_fields[layout.bx])
                    ball[row, 1] = float(ball_fields[layout.by])
            seen[row] = True

    pitch = tuple(pitch or meta.pitch or (105.0, 68.0))
    if verbose:
        print("  {} frames read, {} layout(s), {} track slots, pitch {} x {} m".format(
            int(seen.sum()), len(layouts), n_slots, *pitch))
        print("  teams: H = {}, A = {}".format(meta.teams[0][1], meta.teams[1][1]))
        for note in meta.notes:
            print("  NOTE: {}".format(note))
        skipped["frames with no line at all"] = int(n_frames - seen.sum())
        for what, count in skipped.items():
            if count:
                print("  WARNING: {} line(s)/frame(s) {} (left as NaN)".format(count, what))

    return Tracking(ball=ball, players=players, team=team, pitch=pitch,
                    periods=list(periods) if periods else [(0, n_frames)], fps=meta.fps,
                    team_names=(meta.teams[0][1], meta.teams[1][1]), speed=speed)
