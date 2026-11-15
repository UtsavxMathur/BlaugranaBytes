# Blaugrana Bytes: how to run our solution

This is the practical guide: how to install, make the submission, check it, and retrain the
model. If you want to know what we built and why, read the technical document,
[Blaugrana Bytes.md](Blaugrana%20Bytes.md).

The organisers' own files (`CHALLENGE.md`, `DATA.md`, `evaluation/`, `starter/`, ...) are
untouched. We only added a pointer at the top of `README.md` and a few lines to `.gitignore`.

## What happens when you run it

```
data/tactical/ (the match)       models/possession.npz (trees) + models/sequence_wide.npz,
        |                                          models/sequence2.npz (two networks)
        v                                          v
  read the tracking  -->  ~110 numbers per frame  -->  "dead?" and "home?" for every frame,
                                                       the models' answers averaged (trees 1/2)
                                                                 |
  read Smart Tagging  -->  goal to kick-off        -->  decoder: frames into possessions
                                                                 |
                                                                 v
                                             outputs/Blaugrana Bytes.csv  +  its SHA-256
```

## Setup (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt     # pinned versions, Python 3.11
```

Put the data bundle in `data/`, or point `MTAH_DATA_DIR` at it. The submission reads only
`data/tactical/`; `data/broadcast/` is needed only for the optional `--broadcast` step. We do not
use the videos.

## Make the submission

```bash
.venv/bin/python generate_submission.py
```

This writes `outputs/Blaugrana Bytes.csv`, runs the official format check on it and prints its
SHA-256. It takes about a minute and a half. Only numpy and pandas are needed: the trained trees
are stored in `models/possession.npz` and the two sequence networks in `models/sequence_wide.npz`
and `models/sequence2.npz` with whole numbers for weights, and we run them ourselves in numpy (the
networks in integer arithmetic), so a fresh clone gives the same file, byte for byte. Our
checksum is in the technical document.

This is our strict version: nothing in it is trained, tuned or set from the provided match (see
the technical document). Our earlier versions, whose training simulator was set from summary
statistics of the match's tracking, are in the git history (commit `6630f72` and before).
`--no-sequence` leaves both networks out (the trees only).

`--broadcast` adds an optional step that lines the broadcast recording up with the tactical one
and uses its replays and close-ups as evidence of dead ball. It is not part of our submission:
its weights are measured on the match itself, and the organisers' clarification of 15 November
asks for tuned values to come from external data (see the technical document).

## Check it

```bash
# Our reader against the organisers' baseline: must print 100.00% and MATCH
.venv/bin/python generate_submission.py --model baseline \
    --compare evaluation/results/baseline_smooth50.csv --check-public

# Diagnostics from earlier development, not used by this version: our model on the public
# 5 minutes, summary statistics of the real tracking, and the broadcast line-up
.venv/bin/python generate_submission.py --check-public
.venv/bin/python scripts/profile_atd.py
.venv/bin/python scripts/broadcast_report.py

# The tests
.venv/bin/python -m pytest tests -q
```

Everything these commands print is a count, a percentage or a score. None of them prints frames
or positions.

## Retrain the model

```bash
.venv/bin/pip install -r requirements-train.txt   # scikit-learn and PyTorch
.venv/bin/python training/fetch.py --only metrica   # public games into .cache/public/ (190 MB)
.venv/bin/python training/fetch.py --only dfl       # (2.6 GB)
# 1. trees on the labels as the files give them: their "dead?" part is kept (~30 min)
.venv/bin/python training/train.py --work-dir .cache/strict/train \
    --out .cache/strict/models/possession_v1.npz
# 2. trees on controlled-possession labels (training/ownership.py): the "home?" part (~30 min)
.venv/bin/python training/train.py --labels controlled --work-dir .cache/strict/train_controlled \
    --out .cache/strict/models/possession_controlled.npz
# 3. put the two together and choose their decoder prices on held-out games (writes models/possession.npz)
.venv/bin/python training/combine.py --dead-model .cache/strict/models/possession_v1.npz \
    --dead-work-dir .cache/strict/train --team-model .cache/strict/models/possession_controlled.npz \
    --team-work-dir .cache/strict/train_controlled --work-dir .cache/strict/train_hybrid
# 4. the wide network (128 channels, 6 copies, 3000 steps, cosine learning rate, dropout 0.1):
#    held-out predictions in whole numbers (~50 min)
.venv/bin/python training/sequence.py --steps cv int --copies 6 --train-steps 3000 --cosine \
    --dropout 0.1 --width 128 --work-dir .cache/strict/train_seqB --out .cache/strict/models/sequence_wide.pt
# 5. the second network (also trained with event heads, which are dropped): held-out predictions,
#    the average of trees and both networks, its decoder prices, the final network (~20 min)
.venv/bin/python training/sequence.py --change --with .cache/strict/train_seqB_int \
    --trees .cache/strict/train_hybrid --name strict2 --compare none \
    --work-dir .cache/strict/train_seq_change --out .cache/strict/models/sequence2.pt --export models/sequence2.npz
# 6. the final wide network, with the same prices
.venv/bin/python training/sequence.py --steps final --copies 6 --train-steps 3000 --cosine \
    --dropout 0.1 --width 128 --work-dir .cache/strict/train_seqB --out .cache/strict/models/sequence_wide.pt \
    --export models/sequence_wide.npz --prices .cache/strict/train_seq_change_blend/prices.json
```

Each training run tests the model on games it has not seen, picks the decoder settings on those,
and then trains the final model on all ten games. A retrained model can differ slightly from the
one in the repository, because scikit-learn and PyTorch add up some sums on several threads or on
the graphics processor. Only training needs PyTorch.

More scripts for changing the model:

```bash
.venv/bin/python training/ownership.py    # what the controlled-possession rule does to each game
.venv/bin/python training/benchmark.py --labels controlled --save NAME --compare OTHER
                                          # held-out scores in detail, two versions side by side
```

## Where things are

```
generate_submission.py     the one command that makes the submission
models/possession.npz      the trained trees
models/sequence_wide.npz   the two sequence networks, in whole numbers
models/sequence2.npz       (averaged; the decoder prices are in the second)
possession/                everything used on the match
training/                  everything that touches the public training games
scripts/profile_atd.py     summary statistics of the real tracking (a diagnostic, not used)
scripts/broadcast_report.py  how the broadcast lines up, and what it adds
tests/                     the checks
```

| file | what it does |
|---|---|
| `possession/epts.py`, `tracking.py`, `competition.py` | read the tracking into arrays; load the match |
| `possession/features.py` | the numbers we compute for every frame |
| `possession/trees.py` | runs the trained trees in numpy |
| `possession/decode.py` | turns frame probabilities into possessions |
| `possession/model.py` | puts features, trees, network and decoder together |
| `possession/seqnet.py` | runs the sequence network in integer arithmetic |
| `training/sequence.py` | trains a sequence network (PyTorch), tests it on held-out games, blends it with the trees (and with `--with`, with the other network) |
| `possession/smart_tagging.py`, `tags.py` | read the tags; use them for goal-to-kick-off stoppages |
| `possession/broadcast.py` | the optional `--broadcast` step: line the broadcast up with the tactical recording; turn its replays and close-ups into evidence of dead ball |
| `possession/baseline.py` | the organisers' baseline on our arrays, to check the reader |
| `possession/profile.py` | summary statistics of a tracking (a diagnostic; not used by the submission) |
| `possession/submission.py` | writes and checks the CSV |
| `training/fetch.py`, `dfl.py`, `metrica.py`, `dataset.py` | download and load the public games |
| `training/simulate.py` | damage public tracking like an automatic tracker, over wide ranges fixed in advance |
| `training/train.py`, `analyse.py` | train, test on held-out games, look at the errors |
| `training/ownership.py` | the controlled-possession rule that relabels the public games |
| `training/combine.py` | one model's "dead?" part with another's "home?" part, prices chosen on held-out games |
| `training/benchmark.py`, `possession/errors.py` | held-out scores per IoU threshold, why possessions are missed, paired comparisons |
| `tests/make_fake_bundle.py` | builds a stand-in for the data bundle from a public game |

## Results on games the model never saw

A game is never labelled by a model that was trained on it. Scores are from the official
`evaluation/possession_metrics.py`, against the controlled-possession labels. We test under three
fixed simulator settings across the ranges the training draws from: low, mid and high damage
(`training/simulate.py`). "Trees alone" is `--no-sequence`.

| held-out games | simulator setting | our spellF1 | trees alone | baseline | our macroF1 | baseline macroF1 |
|---|---|---|---|---|---|---|
| DFL (7) | low damage | 0.771 | 0.751 | 0.336 | 0.930 | 0.726 |
| DFL (7) | mid | 0.748 | 0.717 | 0.249 | 0.926 | 0.611 |
| DFL (7) | high damage | 0.724 | 0.681 | 0.179 | 0.911 | 0.501 |
| Metrica (3) | low damage | 0.823 | 0.801 | 0.369 | 0.913 | 0.761 |
| Metrica (3) | mid | 0.787 | 0.760 | 0.268 | 0.899 | 0.616 |
| Metrica (3) | high damage | 0.755 | 0.696 | 0.187 | 0.885 | 0.527 |

The choice of networks was made on these held-out games, with each fold's prices chosen on the
other folds: trees with the wide and the second network 0.760, with the narrow network instead
0.754, with the wide network alone 0.751, trees alone 0.727. (Our earlier versions, with a
simulator set from summary statistics of the match, are compared in the git history.)

On the real match the baseline scores spellF1 0.2433 and macroF1 0.6724 (the organisers' figures),
which sits inside the range of our three settings.

The optional broadcast step cannot be in this table: the public games have no broadcast feed.

One experiment is worth keeping on record, because it shaped the model. The metric counts each
possession once, however long it lasts, so we tried giving short possessions and short stoppages
more weight in training (small models, same held-out games, an earlier version of the simulator):

| training weights | Metrica spellF1 | DFL spellF1 |
|---|---|---|
| every frame counts the same | 0.600 | 0.505 |
| segments under 6 s count more (up to 8 times) | 0.611 | 0.538 |
| segments under 12 s count more (up to 8 times), which we kept | 0.639 | 0.544 |
| segments under 24 s count more (up to 16 times) | 0.643 | 0.550 |
| the one we kept, plus "a moment before and after" features | 0.613 | 0.534 |
