"""Read the Smart Tagging XML: automatic phases of play and set pieces, as video-frame intervals.

DATA.md: <start>/<end> are seconds from the first frame of the tactical video, so
round(seconds * 25) is a video frame. Codes are upper case ("SET PIECES"), tag values title case
("Kick Off (Start)"); everything is lower-cased here so a change of case cannot silently stop a
match. The Team tag names a club ("Barcelona", "Atletic Club") or "N/A" (every throw-in).

The tags are inferred, not ground truth: they are used as evidence, never copied as labels.
"""

import dataclasses
import difflib
import xml.etree.ElementTree as ET

from .tracking import AWAY, HOME

NO_TEAM = -1
SET_PIECES = "set pieces"
KICK_OFF_START = "kick off (start)"
KICK_OFF_GOAL = "kick off (after goal)"


@dataclasses.dataclass
class Tag:
    code: str     # lower case, e.g. "set pieces", "progression"
    start: int    # first video frame of the interval
    stop: int     # one past its last video frame
    team: int     # HOME, AWAY or NO_TEAM
    kind: str     # the "Type" tag in lower case for set pieces ("throw-in", ...), else ""
    half: str     # "1st half" / "2nd half" / ""


def _team_map(values, team_names):
    """{tag value: HOME/AWAY/NO_TEAM}. Club spellings differ between files, so match fuzzily."""
    clubs = sorted(v for v in values if v and v.strip().lower() not in ("n/a", "na", "none", ""))
    mapping = {v: NO_TEAM for v in values}
    if not clubs:
        return mapping

    def likeness(a, b):
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    if len(clubs) == 1:
        home_like = likeness(clubs[0], team_names[0]) >= likeness(clubs[0], team_names[1])
        mapping[clubs[0]] = HOME if home_like else AWAY
        return mapping
    if len(clubs) > 2:
        raise ValueError("Smart Tagging names more than two teams: {}".format(clubs))
    straight = likeness(clubs[0], team_names[0]) + likeness(clubs[1], team_names[1])
    crossed = likeness(clubs[0], team_names[1]) + likeness(clubs[1], team_names[0])
    first = HOME if straight >= crossed else AWAY
    mapping[clubs[0]], mapping[clubs[1]] = first, (AWAY if first == HOME else HOME)
    return mapping


def load(path, team_names, fps=25):
    """([Tag] sorted by start, {team tag value: HOME/AWAY/NO_TEAM})."""
    rows = []
    for instance in ET.parse(path).iter("instance"):
        tags = {label.findtext("group"): (label.findtext("text") or "")
                for label in instance.findall("label")}
        rows.append((int(round(float(instance.findtext("start")) * fps)),
                     int(round(float(instance.findtext("end")) * fps)),
                     (instance.findtext("code") or "").strip().lower(), tags))
    mapping = _team_map({tags.get("Team") for _, _, _, tags in rows}, team_names)
    out = [Tag(code=code, start=start, stop=max(stop, start + 1), team=mapping[tags.get("Team")],
               kind=(tags.get("Type") or "").strip().lower(),
               half=(tags.get("Half") or "").strip().lower())
           for start, stop, code, tags in rows]
    out.sort(key=lambda tag: (tag.start, tag.stop, tag.code))
    return out, mapping


def kickoffs(tags):
    """{period index: first frame of that half's 'Kick Off (Start)' tag}, for halves that have one."""
    out = {}
    for tag in tags:
        if tag.code == SET_PIECES and tag.kind == KICK_OFF_START:
            period = 1 if tag.half.startswith("2") else 0
            out[period] = min(tag.start, out.get(period, tag.start))
    return out
