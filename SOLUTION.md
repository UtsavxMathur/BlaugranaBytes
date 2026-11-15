# Blaugrana Bytes: how to run our solution

This is the practical guide: how to install, make the submission, check it, and retrain the
model. If you want to know what we built and why, read the technical document,
[Blaugrana Bytes.md](Blaugrana%20Bytes.md).

The organisers' own files (`CHALLENGE.md`, `DATA.md`, `evaluation/`, `starter/`, ...) are
untouched. We only added a pointer at the top of `README.md` and a few lines to `.gitignore`.

## What happens when you run it

```
data/tactical/ (the match)            models/possession.npz (trained on public games)
        |                                          |
        v                                          v
  read the tracking  -->  ~110 numbers per frame  -->  "dead?" and "home?" for every frame
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
SHA-256. It takes about half a minute. Only numpy and pandas are needed: the trained trees are
stored in `models/possession.npz` and we run them ourselves in numpy, so a fresh clone gives the
same file, byte for byte. Our checksum is in the technical document.

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
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python training/fetch.py        # downloads 2.8 GB of public games into .cache/public/
.venv/bin/python training/train.py        # about 30 minutes, writes models/possession.npz
```

Training tests the model on games it has not seen, picks the decoder settings on those, and then
trains the final model on all ten games. A retrained model can differ slightly from the one in
the repository, because scikit-learn builds trees on several threads.

Two more scripts are useful when changing the simulator or the model:

```bash
.venv/bin/python training/calibrate.py    # simulated tracking next to the real profile
.venv/bin/python training/analyse.py      # which possessions the model finds and misses
```

## Where things are

```
generate_submission.py     the one command that makes the submission
models/possession.npz      the trained model
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
| `possession/model.py` | puts features, trees and decoder together |
| `possession/smart_tagging.py`, `tags.py` | read the tags; use them for goals and restarts |
| `possession/broadcast.py` | the optional `--broadcast` step: line the broadcast up with the tactical recording; turn its replays and close-ups into evidence of dead ball |
| `possession/baseline.py` | the organisers' baseline on our arrays, to check the reader |
| `possession/profile.py` | the summary statistics |
| `possession/submission.py` | writes and checks the CSV |
| `training/fetch.py`, `dfl.py`, `metrica.py`, `dataset.py` | download and load the public games |
| `training/simulate.py`, `calibrate.py` | make public tracking look like the real tracker's output |
| `training/train.py`, `analyse.py` | train, test on held-out games, look at the errors |
| `tests/make_fake_bundle.py` | builds a stand-in for the data bundle from a public game |

## Results on games the model never saw

A game is never labelled by a model that was trained on it. Scores are from the official
`evaluation/possession_metrics.py`. We test under three simulator settings: our best estimate of
the real tracker, one with more ball gaps during play, and one where the ball is tracked more
often during stoppages.

| held-out games | simulator setting | our spellF1 | baseline spellF1 | our macroF1 | baseline macroF1 |
|---|---|---|---|---|---|
| DFL (7) | typical | 0.570 | 0.284 | 0.895 | 0.692 |
| DFL (7) | more gaps during play | 0.537 | 0.252 | 0.882 | 0.664 |
| DFL (7) | ball tracked more when dead | 0.566 | 0.267 | 0.893 | 0.616 |
| Metrica (3) | typical | 0.635 | 0.314 | 0.882 | 0.711 |
| Metrica (3) | more gaps during play | 0.626 | 0.286 | 0.873 | 0.676 |
| Metrica (3) | ball tracked more when dead | 0.643 | 0.320 | 0.881 | 0.659 |

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
