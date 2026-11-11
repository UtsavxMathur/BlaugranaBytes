# Team Blaugrana Bytes: our solution

Working notes for the team: how to install, run and reproduce. The technical document for the
submission is `Blaugrana Bytes.md`. The deadline is Nov 17, 2026. The organisers' documents
(`CHALLENGE.md`, `DATA.md`, ...) are unchanged; `README.md` only gained a pointer at the top.

## The approach

1. **Train on other matches.** The rules forbid training or tuning on the public 5 minutes and
   forbid hand labels on this match, so the model learns from public games:
   - DFL open data: 7 Bundesliga matches whose tracking carries, on every frame, the team in
     possession and whether the ball is in play (the same kind of label we are scored on).
   - Metrica sample data: 3 games in the competition's own file formats, with labels we derive
     from their event data.
2. **Render the clean public tracking the way the competition's tracker delivers it** before
   training on it (`training/simulate.py`). The profile of the real tracking
   (`scripts/profile_atd.py`, aggregates only) shows the tracker bridges every ball gap under 5 s
   with a straight line, smooths the track, loses the ball at stoppages and finds it at the
   restart, and sometimes loses it for 5 s or more during play. The simulator reproduces that
   chain, and `training/calibrate.py` checks its output against the real profile.
3. **Classify every frame** (`possession/features.py`, `possession/model.py`): about 110 numbers
   per frame (who is nearest the ball and for how long, ball touches, how long the ball has been
   unseen, how fast the players move, team shapes) go into two gradient-boosted tree ensembles:
   "is the ball dead?" and "if not, does home have it?". Each is asked twice, once with the teams
   exchanged, so the model cannot prefer one side.
4. **Decode into possessions** (`possession/decode.py`): a Viterbi pass that pays a price for
   every change of label, so flicker does not become invented possessions.
5. **Use Smart Tagging as gated evidence** (`possession/tags.py`): the stretch between a goal
   and its kick-off is set to dead when the model already mostly agrees, and a set-piece tag with
   no stoppage found nearby gets a short one just before the usual restart moment.

## Setup (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt     # pinned versions, Python 3.11
```

Put the data bundle in `data/` (or set `MTAH_DATA_DIR`). The videos are not used.

## Commands

```bash
# The submission file: "outputs/Blaugrana Bytes.csv", official format check, SHA-256
.venv/bin/python generate_submission.py

# Sanity checks on the real match (all print aggregates only)
.venv/bin/python generate_submission.py --check-public --drift
.venv/bin/python generate_submission.py --model baseline \
    --compare evaluation/results/baseline_smooth50.csv --check-public   # must print 100.00%
.venv/bin/python scripts/profile_atd.py                # saved to outputs/atd_profile.txt

# Retrain from the public games (needs requirements-train.txt; about 15 minutes)
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python training/fetch.py                     # 2.8 GB into .cache/public/
.venv/bin/python training/train.py                     # writes models/possession.npz

.venv/bin/python -m pytest tests -q
```

Making the submission needs only numpy and pandas: the trained trees live in
`models/possession.npz` and are evaluated in plain numpy, with integer arithmetic in the decoder,
so a fresh clone produces the same file byte for byte.

## Layout

| path | what |
|---|---|
| `generate_submission.py` | entry point: data bundle -> `outputs/Blaugrana Bytes.csv` |
| `models/possession.npz` | the trained model (trees, feature names, decoder prices) |
| `possession/tracking.py` | `Tracking`: one array format for every data source |
| `possession/epts.py` | FIFA/EPTS reader (competition ATD, Metrica Sample Game 3) |
| `possession/competition.py` | loads the competition match (tracking, Smart Tagging, homography) |
| `possession/features.py` | tracking -> numbers per frame |
| `possession/trees.py` | tree ensembles: export from scikit-learn, exact inference in numpy |
| `possession/decode.py` | frame probabilities -> possessions |
| `possession/model.py` | features + trees + decoder; feature-drift report |
| `possession/baseline.py` | organisers' baseline on arrays: the known floor, and a check of the reader |
| `possession/smart_tagging.py`, `possession/tags.py` | Smart Tagging reader; gated evidence from it |
| `possession/profile.py` | aggregate statistics of a tracking |
| `possession/submission.py` | write / validate / fingerprint the CSV, official scoring helpers |
| `training/fetch.py`, `dfl.py`, `metrica.py`, `dataset.py` | public games: download, load, labels |
| `training/simulate.py`, `training/calibrate.py` | renders clean tracking like the competition's tracker; compares with the real profile |
| `training/train.py` | cross-validation by game, decoder prices, final model |
| `scripts/profile_atd.py` | prints the profile of the real tracking (you run it) |
| `tests/` | pytest suite; `tests/make_fake_bundle.py` builds a stand-in bundle from a public game |

## Results on held-out public games

Cross-validated by game (a game is never in the training set of the model that labels it), scored
with the official `evaluation/possession_metrics.py`. Three simulator settings: the calibrated
one, one with more ball gaps during play, one with the ball tracked more often during stoppages.

| held-out games | simulator setting | our spellF1 | baseline spellF1 | our macroF1 | baseline macroF1 |
|---|---|---|---|---|---|
| DFL (7) | typical | 0.570 | 0.284 | 0.895 | 0.692 |
| DFL (7) | more gaps during play | 0.537 | 0.252 | 0.882 | 0.664 |
| DFL (7) | ball tracked more when dead | 0.566 | 0.267 | 0.893 | 0.616 |
| Metrica (3) | typical | 0.635 | 0.314 | 0.882 | 0.711 |
| Metrica (3) | more gaps during play | 0.626 | 0.286 | 0.873 | 0.676 |
| Metrica (3) | ball tracked more when dead | 0.643 | 0.320 | 0.881 | 0.659 |

For scale: the same baseline scores spellF1 0.2433 and macroF1 0.6724 on the competition match
(organisers' figures), so the simulated games are about as hard for it as the real one.

Training choices measured on these held-out games (small models, same folds):

| change | Metrica spellF1 | DFL spellF1 |
|---|---|---|
| frames weighted equally | 0.600 | 0.505 |
| short segments up-weighted (under 6 s, up to 8x) | 0.611 | 0.538 |
| short segments up-weighted (under 12 s, up to 8x) -- adopted | 0.639 | 0.544 |
| ... under 24 s, up to 16x | 0.643 | 0.550 |
| adopted weighting + "moment before/after" features | 0.613 | 0.534 |

The metric counts each possession once, so a classifier trained on frames must be told that the
frames of a short possession matter as much, in total, as those of a long one.

## Findings so far

- Our reader and baseline reproduce the organisers' exactly on the real match: 100.00% agreement
  with `baseline_smooth50.csv`, and the public-window score matches the published 0.3256.
- The real ball track has almost no gaps under 5 s (17 in the match) and 132 gaps of 5 s or
  more; it is heavily smoothed. So a "visible" ball is often an interpolated one, and a short
  stoppage can leave no gap at all.
- About a quarter of the long gaps happen while the players are still running: play went on
  unseen. "Ball missing" is therefore not "ball dead", and the reverse fails too (the ball is
  tracked 80% of the time in the two minutes before the second half kicks off).
- Smart Tagging timing: set-piece tags are 35 s windows that start about 6 s before the restart;
  a kick-off is taken about 3.4 s after its tag starts; a GOALS tag runs until just before the
  following kick-off and the ball is seen only 12% of the time inside it.
- Label conventions differ between sources. Possessions per minute of live play: public truth
  4.2 (17 possessions, 5 minutes), Metrica event-derived labels 5.8-6.4, DFL 7.1-11.0. All sources
  agree on stoppages: about 90-113 per match, median 15-24 s.
- A check of label meaning on the public window showed our first models called a stoppage dead
  only while the ball was missing (34% of true dead frames found, at 92% precision). The real
  tracker often follows a moving ball during stoppages. The simulator now writes that in
  (`_dead_story` in `training/simulate.py`: the ball rolls to rest, is passed via nearby players
  to the taker, and is carried), so the dead-ball decision leans on what the players do.
- On held-out games the model finds almost every stoppage of 3 s or more; most remaining errors
  are possessions shorter than 4 s (45% of all possessions) that it merges into their neighbours.

## Open risks

- The simulator matches the real profile on gap lengths, visible-run lengths, ball smoothness and
  player speeds, but it puts the ball near the touchlines about twice as often as the real
  tracking does.
- The public 5 minutes are used only to confirm the pipeline is not broken, never to choose
  settings (RULES.md), so there is no direct measurement on the competition match.

## Status

- [x] Core: reader, baseline, CSV writer, entry point, tests
- [x] Public data: 7 DFL matches + 3 Metrica games load with labels
- [x] Profile of the real tracking; simulator rebuilt to match it
- [x] Features, classifier, decoder; model in `models/possession.npz`
- [x] Smart Tagging evidence (gated)
- [x] Short possessions: training weights by segment length (+0.04 held-out)
- [x] Technical document: `Blaugrana Bytes.md`
- [ ] Freeze, reproducibility check on a fresh clone, public repo, submit
