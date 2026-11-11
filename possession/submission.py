"""Write, check and fingerprint the frame,label submission file, and score label arrays.

Everything about validity and scoring comes from the official evaluation/possession_metrics.py,
imported rather than copied, so a file that passes here passes the organisers' scorer and a score
printed here is the official definition.
"""

import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402  (official scorer; codes: H=0 A=1 D=2 X=3)

TEAM_NAME = "Blaugrana Bytes"  # as registered; the file must be <team_name>.csv (SUBMISSION.md)
DEFAULT_OUT = os.path.join(ROOT, "outputs", TEAM_NAME + ".csv")
LETTERS = np.array(["H", "A", "D", "X"])  # LETTERS[code] -> label


def to_letters(codes):
    return LETTERS[np.asarray(codes)]


def finalize(labels, periods):
    """Copy of the labels with every frame outside the halves set to X (never scored)."""
    out = np.full(len(labels), "X", dtype="<U1")
    for start, stop in periods:
        out[start:stop] = np.asarray(labels)[start:stop]
    return out


def write(path, labels):
    """Write the CSV byte-for-byte deterministically: UTF-8, '\\n' line endings on every OS."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def check(path, n_rows):
    """None if the official format checks pass, else the official one-line reason."""
    try:
        table = pm.read_label_csv(path)
    except Exception as error:  # unreadable, or wrong columns
        return "could not read: {}".format(error)
    return pm.validate_submission(table, n_rows)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(codes, periods):
    """Counts the scorer would see inside the halves: possessions, mean length, share of D."""
    codes = np.asarray(codes)
    spells = pm.possession_spells(codes, list(periods))
    lengths = np.array([stop - start for start, stop, _ in spells], dtype=float)
    inside = np.concatenate([codes[start:stop] for start, stop in periods])
    return {"n_possessions": len(spells),
            "mean_s": float(lengths.mean() / pm.FPS) if len(lengths) else 0.0,
            "median_s": float(np.median(lengths) / pm.FPS) if len(lengths) else 0.0,
            "share_D": float(np.mean(inside == pm.D)) if len(inside) else 0.0}


def score(gt_codes, pred_codes, periods):
    """Official spellF1 / macroF1 over several periods, scored as one match.

    The periods are joined with one X frame between them. X is never graded, and the scorer
    treats each run of graded frames separately, exactly like the two halves of the real match.
    """
    gap = np.array([pm.X], dtype=np.int8)
    gt = np.concatenate([part for s, e in periods for part in (gt_codes[s:e], gap)])
    pred = np.concatenate([part for s, e in periods for part in (pred_codes[s:e], gap)])
    return pm.evaluate(gt, pred)
