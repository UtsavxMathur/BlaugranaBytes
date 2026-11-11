"""One in-memory shape for tracking data, whichever file it came from.

Every loader (competition ATD, Metrica samples, DFL open data) returns a Tracking, and the model
only ever sees a Tracking. So exactly the same code runs on the competition match and on the
public games we train and validate on.

Row i of every array is video frame i.
"""

import dataclasses
import hashlib
import os

import numpy as np

HOME, AWAY = 0, 1

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parsed arrays are cached here so slow text parsing runs once. A cache made from data/ is a
# frame-level extract of the match data: it must never be committed or shared. .cache/ is
# gitignored for that reason.
CACHE_DIR = os.environ.get("POSSESSION_CACHE_DIR") or os.path.join(ROOT, ".cache", "arrays")
CACHE_VERSION = 1  # bump when a parser changes, so stale caches are not reused


@dataclasses.dataclass
class Tracking:
    """Positions for one recording.

    ball      (n, 2)      x, y as fractions 0..1 of the pitch. NaN = ball not seen.
    players   (n, s, 2)   same, one column per track slot. NaN = slot empty on that frame.
    team      (s,)        HOME (0) or AWAY (1) per slot.
    pitch     (length_m, width_m): x * length and y * width give metres.
    periods   [(start, stop)] half-open row ranges of the halves (the frames that are graded).
    speed     (n, s) m/s as delivered by the file, or None. Informational; features recompute it.
    ball_z    (n,) ball height in metres, or None. Only some public games have it; it is used
              by the simulator in training/, never by the model.

    A slot is a column of the tracking file. In the competition ATD a slot is a tracklet, not a
    person (46 slots for 22 players, identities switch), so nothing here assumes slot == player.
    """
    ball: np.ndarray
    players: np.ndarray
    team: np.ndarray
    pitch: tuple
    periods: list
    fps: int = 25
    team_names: tuple = ("home", "away")
    speed: np.ndarray = None
    ball_z: np.ndarray = None

    @property
    def n_frames(self):
        return self.ball.shape[0]

    @property
    def n_slots(self):
        return self.players.shape[1]

    def ball_visible(self):
        """Boolean (n,): True where the ball has a position."""
        return ~np.isnan(self.ball).any(axis=1)

    def in_periods(self):
        """Boolean (n,): True inside the halves."""
        mask = np.zeros(self.n_frames, dtype=bool)
        for start, stop in self.periods:
            mask[start:stop] = True
        return mask

    def replace(self, **changes):
        """A copy sharing every array that is not replaced."""
        return dataclasses.replace(self, **changes)

    # -- cache -----------------------------------------------------------------------------------

    def save(self, path, compact=False):
        """compact=True stores positions as float32 (half the disk; fine for public games)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        kind = np.float32 if compact else np.float64
        arrays = dict(ball=self.ball.astype(kind), players=self.players.astype(kind),
                      team=self.team, pitch=np.array(self.pitch, dtype=np.float64),
                      periods=np.array(self.periods, dtype=np.int64).reshape(-1, 2),
                      fps=np.array(self.fps), team_names=np.array(self.team_names))
        if self.speed is not None:
            arrays["speed"] = self.speed.astype(np.float32)
        if self.ball_z is not None:
            arrays["ball_z"] = self.ball_z.astype(np.float32)
        tmp = path + ".tmp.npz"  # np.savez appends .npz unless the name already ends with it
        np.savez(tmp, **arrays)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(ball=z["ball"].astype(np.float64), players=z["players"].astype(np.float64),
                       team=z["team"], pitch=tuple(float(v) for v in z["pitch"]),
                       periods=[(int(a), int(b)) for a, b in z["periods"]], fps=int(z["fps"]),
                       team_names=tuple(str(v) for v in z["team_names"]),
                       speed=z["speed"] if "speed" in z else None,
                       ball_z=z["ball_z"] if "ball_z" in z else None)


def show_path(path):
    """Repo-relative path for printing when inside the repo, absolute otherwise."""
    path = os.path.abspath(path)
    return os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path


def cache_path(sources, tag="", cache_dir=None):
    """Cache file name for a set of source files. Changes when any source file changes."""
    digest = hashlib.sha1()
    for path in sources:
        stat = os.stat(path)
        digest.update("{}|{}|{}\n".format(os.path.abspath(path), stat.st_size,
                                          stat.st_mtime_ns).encode())
    digest.update("v{}|{}".format(CACHE_VERSION, tag).encode())
    return os.path.join(cache_dir or CACHE_DIR, digest.hexdigest()[:20] + ".npz")
