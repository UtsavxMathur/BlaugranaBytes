# Blaugrana Bytes: how to run our solution

This is the practical guide: how to install, make the submission, check it, and retrain the
model. If you want to know what we built and why, read the technical document,
[Blaugrana Bytes.md](Blaugrana%20Bytes.md).

The organisers' own files (`CHALLENGE.md`, `DATA.md`, `evaluation/`, `starter/`, ...) are
untouched. We only added a pointer at the top of `README.md` and a few lines to `.gitignore`.

## What happens when you run it

```
data/tactical/ (the match)       models/possession.npz (trees) + models/sequence.npz (network)
        |                                          |
        v                                          v
  read the tracking  -->  ~110 numbers per frame  -->  "dead?" and "home?" for every frame,
                                                       the two models' answers averaged
                                                                 |
  read Smart Tagging  -->  kick-offs, goals, restarts  -->  decoder: frames into possessions
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
SHA-256. It takes about 40 seconds. Only numpy and pandas are needed: the trained trees are
stored in `models/possession.npz` and the sequence network in `models/sequence.npz` with whole
numbers for weights, and we run both ourselves in numpy (the network in integer arithmetic), so a
fresh clone gives the same file, byte for byte. Our checksum is in the technical document.

`--no-sequence` leaves the network out and writes our second submission; adding
`--model-file models/possession_v1.npz` writes our first, with the model trained on the labels as
the files give them.

`--broadcast` adds an optional step that lines the broadcast recording up with the tactical one
and uses its replays and close-ups as evidence of dead ball. It is not part of our submission:
its weights are measured on the match itself, and the organisers' clarification of 15 November
asks for tuned values to come from external data (see the technical document).

## Check it

```bash
# Our reader against the organisers' baseline: must print 100.00% and MATCH
.venv/bin/python generate_submission.py --model baseline \
    --compare evaluation/results/baseline_smooth50.csv --check-public

# Our model on the public 5 minutes. A sanity check, not a way to choose settings.
.venv/bin/python generate_submission.py --check-public

# Summary statistics of the real tracking (saved to outputs/atd_profile.txt)
.venv/bin/python scripts/profile_atd.py

# How the broadcast lines up with the tactical recording, and what it shows when
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
# 1. labels as the files give them: our first model, whose "dead?" part we keep (~30 min)
.venv/bin/python training/train.py --out models/possession_v1.npz
# 2. controlled-possession labels (training/ownership.py): the "home?" part (~30 min)
.venv/bin/python training/train.py --labels controlled \
    --work-dir .cache/train_controlled --out .cache/models/possession_controlled.npz
# 3. put the two together and choose the decoder prices on held-out games
.venv/bin/python training/combine.py      # writes models/possession.npz
# 4. the sequence network, blended with those trees; prices chosen on held-out games (~20 min)
.venv/bin/python training/sequence.py --export models/sequence.npz
```

Each training run tests the model on games it has not seen, picks the decoder settings on those,
and then trains the final model on all ten games. A retrained model can differ slightly from the
one in the repository, because scikit-learn and PyTorch add up some sums on several threads or on
the graphics processor. Only training needs PyTorch.

More scripts for changing the simulator or the model:

```bash
.venv/bin/python training/calibrate.py    # simulated tracking next to the real profile
.venv/bin/python training/ownership.py    # what the controlled-possession rule does to each game
.venv/bin/python training/benchmark.py --labels controlled --save NAME --compare OTHER
                                          # held-out scores in detail, two versions side by side
```

## Where things are

```
generate_submission.py     the one command that makes the submission
models/possession.npz      the trained trees (models/possession_v1.npz: our first ones)
models/sequence.npz        the sequence network, in whole numbers
possession/                everything used on the match
training/                  everything that touches the public training games
scripts/profile_atd.py     summary statistics of the real tracking
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
| `training/sequence.py` | trains the sequence network (PyTorch), tests it on held-out games, blends it with the trees |
| `possession/smart_tagging.py`, `tags.py` | read the tags; use them for goals and restarts |
| `possession/broadcast.py` | the optional `--broadcast` step: line the broadcast up with the tactical recording; turn its replays and close-ups into evidence of dead ball |
| `possession/baseline.py` | the organisers' baseline on our arrays, to check the reader |
| `possession/profile.py` | the summary statistics |
| `possession/submission.py` | writes and checks the CSV |
| `training/fetch.py`, `dfl.py`, `metrica.py`, `dataset.py` | download and load the public games |
| `training/simulate.py`, `calibrate.py` | make public tracking look like the real tracker's output |
| `training/train.py`, `analyse.py` | train, test on held-out games, look at the errors |
| `training/ownership.py` | the controlled-possession rule that relabels the public games |
| `training/combine.py` | one model's "dead?" part with another's "home?" part, prices chosen on held-out games |
| `training/benchmark.py`, `possession/errors.py` | held-out scores per IoU threshold, why possessions are missed, paired comparisons |
| `tests/make_fake_bundle.py` | builds a stand-in for the data bundle from a public game |

## Results on games the model never saw

A game is never labelled by a model that was trained on it. Scores are from the official
`evaluation/possession_metrics.py`, against the controlled-possession labels. We test under three
simulator settings: our best estimate of the real tracker, one with more ball gaps during play,
and one where the ball is tracked more often during stoppages. "Trees alone" is our second
submission (`--no-sequence`), "first model" the one in `models/possession_v1.npz`.

| held-out games | simulator setting | our spellF1 | trees alone | first model | baseline | our macroF1 | baseline macroF1 |
|---|---|---|---|---|---|---|---|
| DFL (7) | typical | 0.760 | 0.731 | 0.609 | 0.295 | 0.924 | 0.700 |
| DFL (7) | more gaps during play | 0.721 | 0.700 | 0.594 | 0.256 | 0.913 | 0.671 |
| DFL (7) | ball tracked more when dead | 0.751 | 0.732 | 0.604 | 0.281 | 0.920 | 0.623 |
| Metrica (3) | typical | 0.772 | 0.763 | 0.713 | 0.321 | 0.901 | 0.717 |
| Metrica (3) | more gaps during play | 0.776 | 0.752 | 0.716 | 0.290 | 0.890 | 0.681 |
| Metrica (3) | ball tracked more when dead | 0.787 | 0.784 | 0.724 | 0.337 | 0.896 | 0.665 |

Paired by game, the trees trained on the controlled-possession labels gained +0.12 on the DFL
games and +0.05 on the Metrica games over the first model, better in all ten (twice the standard
error across games: 0.03). Adding the sequence network gained another +0.019, better in nine of
ten (twice the standard error: 0.009).

On the real match the baseline scores spellF1 0.2433 and macroF1 0.6724 (the organisers' figures),
so our simulated games are about as hard for it as the real one.

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
