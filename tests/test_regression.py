"""Pin the whole labelling pipeline's output on a made-up match.

Features, the trees in models/possession.npz, the decoder, the Smart Tagging second pass and the
broadcast evidence all run exactly as generate_submission.py runs them. If this checksum
changes, the submission would change too: that must be on purpose, and then PINNED is updated
together with the documents' checksum.
"""

import hashlib
import types

import numpy as np

import generate_submission
from possession import broadcast, model as model_module
from possession.smart_tagging import NO_TEAM, Tag
from possession.tracking import Tracking
from test_broadcast import PITCH, made_up_play

PINNED = "a8f92ddd72bb515ca30786760fdf9c755704b7ade67e900597bee58c36d0756c"
N = 15000


def pipeline_output():
    """(label codes, broadcast evidence costs) for the made-up match."""
    ball, players = made_up_play(N, seed=7)
    rng = np.random.default_rng(7)
    seen = np.ones(N, dtype=bool)
    for start in rng.integers(0, N, 25):
        seen[start:start + int(rng.uniform(5, 20) * 25)] = False
    tracking = Tracking(ball=np.where(seen[:, None], ball, np.nan), players=players,
                        team=np.repeat([0, 1], 11).astype(np.int8), pitch=PITCH,
                        periods=[(0, N // 2), (N // 2, N)])
    tags = [Tag("set pieces", start, start + 875, NO_TEAM, "throw-in", "1st half")
            for start in (1000, 4000, 9000, 12000)]
    state = np.full(N, broadcast.LIVE, dtype=np.int8)
    for start in (2000, 6000, 10500):
        state[start:start + 400] = broadcast.OTHER
    letters, prob = generate_submission.predict("possession", tracking, model_module.DEFAULT_PATH,
                                                tags, types.SimpleNamespace(state=state))
    codes = np.array([{"H": 0, "A": 1, "D": 2, "X": 3}[v] for v in letters], dtype=np.int8)
    extra, _ = broadcast.dead_evidence(state, prob[:, 2], tracking.periods)
    return codes, extra


def test_pipeline_output_is_pinned():
    codes, extra = pipeline_output()
    digest = hashlib.sha256(codes.tobytes() + extra.astype(np.int64).tobytes()).hexdigest()
    assert digest == PINNED, "pipeline output changed: {}".format(digest)
