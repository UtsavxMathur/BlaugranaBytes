"""The public games we train and validate on, behind one function: load(name) -> Game.

    dfl-J03WMX ... dfl-J03WR9    7 Bundesliga matches with TRUE frame labels (training/dfl.py)
    metrica-1 ... metrica-3      3 sample games with PROXY labels from events (training/metrica.py)

Parsing a game takes a while, so the result is cached under .cache/arrays/.
"""

import dataclasses
import os

import numpy as np

from possession.tracking import Tracking, cache_path
from training import dfl, fetch, metrica

DFL_GAMES = ["dfl-" + match for match in sorted(fetch.DFL)]
METRICA_GAMES = ["metrica-{}".format(game) for game in sorted(fetch.METRICA)]
ALL_GAMES = DFL_GAMES + METRICA_GAMES


@dataclasses.dataclass
class Game:
    name: str
    tracking: Tracking      # as delivered: clean, with (nearly) every ball position
    labels: np.ndarray      # int8 (n,): 0 = H, 1 = A, 2 = D, 3 = X (unknown / outside the halves)

    @property
    def source(self):
        return self.name.split("-")[0]


def _sources(name, root):
    kind, key = name.split("-", 1)
    if kind == "dfl":
        paths = [fetch.dfl_path(key, "info", root), fetch.dfl_path(key, "positions", root)]
    else:
        paths = [fetch.metrica_path(int(key), file, root) for file in fetch.METRICA[int(key)]]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise IOError("{} not found. Run: python training/fetch.py".format(missing[0]))
    return paths


def load(name, root=fetch.PUBLIC_DIR, use_cache=True, verbose=True):
    paths = _sources(name, root)
    cache = cache_path(paths, tag="game-" + name)
    labels_cache = cache.replace(".npz", ".labels.npy")
    if use_cache and os.path.isfile(cache) and os.path.isfile(labels_cache):
        return Game(name, Tracking.load(cache), np.load(labels_cache))

    if verbose:
        print("parsing {} ...".format(name))
    kind, key = name.split("-", 1)
    if kind == "dfl":
        tracking, labels = dfl.load(paths[0], paths[1], verbose=verbose)
    else:
        files = dict(zip(fetch.METRICA[int(key)], paths))
        tracking, labels = metrica.load(int(key), files, verbose=verbose)
    if use_cache:
        tracking.save(cache, compact=True)
        np.save(labels_cache, labels)
        tracking = Tracking.load(cache)   # so a cached and a fresh load give identical arrays
    return Game(name, tracking, labels)
