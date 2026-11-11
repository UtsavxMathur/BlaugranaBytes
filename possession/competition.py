"""Load the competition match from the data bundle (data/, or $MTAH_DATA_DIR).

File names, frame counts, pitch size and the frames of the two halves all come from the
organisers' games.py, so nothing is hard-coded twice.

The parsed arrays are cached under .cache/arrays/. That cache is a frame-level extract of the
match data: it is gitignored and must never be committed or shared.
"""

import os
import sys

from . import epts, smart_tagging
from .tracking import ROOT, Tracking, cache_path, show_path

sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

EXPECTED_HOME = "barcelona"  # DATA.md: home (H) is FC Barcelona
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def periods():
    """[(start, stop)] half-open video-frame ranges of the two graded halves."""
    return [(first, last + 1) for _, first, last in half_video_blocks(GAME)]


def n_frames():
    return source_frames("tactical", GAME)


def load_tracking(data_dir=None, use_cache=True, verbose=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    cache = cache_path([meta_path, raw_path], tag="competition-tactical")
    if use_cache and os.path.isfile(cache):
        if verbose:
            print("  using cached arrays {}".format(show_path(cache)))
        tracking = Tracking.load(cache)
    else:
        if verbose:
            print("reading the tactical ATD (takes a few seconds) ...")
        tracking = epts.load(meta_path, raw_path, n_frames=n_frames(),
                             pitch=GAMES[GAME]["pitch"], periods=periods(), verbose=verbose)
        if use_cache:
            tracking.save(cache)
    if verbose and EXPECTED_HOME not in tracking.team_names[0].lower():
        print("  WARNING: home team is {!r}, expected Barcelona. H and A may be swapped -- "
              "check before submitting.".format(tracking.team_names[0]))
    return tracking


def load_homography(data_dir=None):
    """(matrices (n, 3, 3), valid (n,)) for the tactical video: pitch 0..1 -> pixels (DATA.md)."""
    import numpy as np
    import pandas as pd
    path = require(["homography"], GAME, data_dir)[0]
    table = pd.read_csv(path)
    columns = ["h{}{}".format(row, col) for row in range(3) for col in range(3)]
    matrices = np.full((n_frames(), 3, 3), np.nan)
    valid = np.zeros(n_frames(), dtype=bool)
    rows = table["frame"].to_numpy(dtype=np.int64)
    keep = (rows >= 0) & (rows < n_frames())
    matrices[rows[keep]] = table[columns].to_numpy(dtype=np.float64).reshape(-1, 3, 3)[keep]
    valid[rows[keep]] = table["valid"].to_numpy().astype(bool)[keep]
    valid &= ~np.isnan(matrices).any(axis=(1, 2))
    return matrices, valid


def load_tags(tracking, data_dir=None, verbose=True):
    """([Tag], team mapping) from the Smart Tagging file."""
    path = require(["smart_tagging"], GAME, data_dir)[0]
    tags, mapping = smart_tagging.load(path, tracking.team_names, tracking.fps)
    if verbose:
        names = {0: "H", 1: "A", -1: "no team"}
        print("  Smart Tagging: {} intervals; team tags: {}".format(
            len(tags), ", ".join("{!r} -> {}".format(k, names[v]) for k, v in sorted(
                mapping.items(), key=lambda item: str(item[0])))))
    return tags, mapping
