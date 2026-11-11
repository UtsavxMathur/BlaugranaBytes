# Team Blaugrana Bytes: technical document

More than a Hack 2027, Initial Challenge: one label (H / A / D) for each of the 199,319 frames of
FC Barcelona vs Athletic Club.

## 1. Approach in one paragraph

We learn what possession and dead ball look like from **other matches** that carry frame labels,
after rendering their clean tracking **the way the competition's tracker delivers it**. A
gradient-boosted classifier scores every frame from about 110 collective features of the ball and
the players; a Viterbi decoder turns the frame probabilities into whole possessions; the Smart
Tagging file adds evidence about restarts where the tracking agrees with it. Nothing was labelled
by hand, and no label of the provided match was used to train or tune.

```
tactical ATD ─► features (ball, players, 110 per frame) ─► 2 tree ensembles ─► Viterbi decoder ─► labels
                                                            "dead?" "home?"         ▲
Smart Tagging ─► kick-offs, goal-to-kick-off stretches, restart strikes ────────────┘
```

**Why this and not something else.** The rules forbid training or tuning on the public 5 minutes
and hand-labelling the match, so supervision has to come from other games. Hand-written rules on
tracking alone were our starting point (the organisers' baseline, spellF1 0.2433), but the hard
cases (is a short touch a possession? is the ball dead while the tracker still follows one?) need
many examples to get right. We did not build our own video ball detector: with one week, the
provided tracking plus a faithful model of its failures was the better use of time.

## 2. Assets

| asset | used? | how |
|---|---|---|
| Tactical ATD (ball and 46 track slots) | yes | every feature. Slots are treated as anonymous: features are "nearest", "how many", "team average". |
| Smart Tagging | yes, as gated evidence | (a) the stretch between a `GOALS` tag and the next `Kick Off (After Goal)` tag is set to dead if the model already calls most of it dead; (b) for throw-in, corner and goal-kick tags we find the moment the ball is struck (slow or unseen, then fast) and set the stretch up to it to dead, joined to the stoppage the model found. A tag is used only if tags of its kind agree on the strike time (they do within about 3 s; free kicks do not and are ignored). |
| `games.py` constants | yes | frame ranges of the halves; the second-half kick-off anchor (frames before it are dead, as FAQ.md states). |
| Smart Tagging phase teams (Build Up, Progression, ...) | dropped | the tagged team agrees with the nearest-player team on only 55-76% of frames; not strong enough to add. |
| Homography | looked at, dropped | the tactical camera pans with play, but its centre is a median 12 m from the ball along the pitch: too coarse to place a missing ball. |
| ATD speed channel | dropped | present on 92% of tracked players and equal to speed from positions within 0.11 m/s; we recompute speed from positions so the same code runs on the training games. |
| Tactical / broadcast video, broadcast ATD | not used | no time for our own detector; the broadcast feed is not synchronised. |

## 3. Training data

| dataset | games | frames | labels |
|---|---|---|---|
| DFL open data (Bassek et al., *Scientific Data* 2025, doi 10.6084/m9.figshare.28196177, CC-BY 4.0) | 7 Bundesliga / 2. Bundesliga matches, 2022/23 | about 1.0 million at 25 fps | `BallStatus` (in play / dead) and `BallPossession` (home / away) on every frame, produced by the league's official data provider. A frame is D if the ball is dead, else H or A. |
| Metrica Sports sample data (github.com/metrica-sports/sample-data) | 3 anonymised games | about 430,000 at 25 fps | none in the files. We derive them from the public event data with fixed rules (`training/metrica.py`): a team has the ball from its pass, carry, shot, recovery or set piece until the other team gets one; the ball is dead from a ball-out, foul or goal until the next restart. No person labelled frames. |

No labelled data comes from the provided game.

**What we took from the provided game without labels.** Aggregate statistics of its tracking
(`scripts/profile_atd.py`: gap-length histograms, ball speed when lost, player counts, timing of
tags against ball gaps), used to make the simulator below behave like the real tracker. They are
listed in `training/calibrate.py`.

**The public 5 minutes.** Used to confirm the reader (our re-implementation of the baseline
agrees with `baseline_smooth50.csv` on 100.00% of graded frames and scores the published 0.3256
there) and, as CHALLENGE.md suggests, to understand what the truth means by "dead": a per-class
check showed our early models called a stoppage dead only while the ball was missing (34% of the
true dead frames), because the real tracker often follows a moving ball during stoppages. We
fixed that in the simulator, from what a tracker plausibly does, and judged every change on
held-out public games. No setting was fitted to the public window. For transparency, successive
versions scored spellF1 0.51, 0.58, 0.56, 0.66 and 0.64 on it; the submitted one is the last.

## 4. The simulator: making public tracking look like the provided tracking

The public games have a true ball position on every frame. The provided tracking does not, and
its profile shows exactly how it differs (`training/simulate.py` reproduces each point):

- **No short gaps.** 132 ball gaps of 5 s or more and almost none shorter: the tracker bridges
  shorter gaps with a straight line. A "visible" ball is often interpolated.
- **Heavy smoothing.** Velocity changes are about a third of the true ones.
- **The ball is lost at stoppages and found at restarts**, but about one gap in four happens
  while the players are still running (play went on unseen for 5 s or more).
- **The ball is often tracked while dead.** It keeps its motion when play stops, is passed to the
  player who restarts, and is carried; the simulator writes this with the players' real movements.
- **Airborne balls are drawn off the pitch** (flat-pitch projection), up to 20 m out.
- Players are tracked steadily (10.2 of 11 per team), with rare losses.

Every training game is rendered three times under randomly drawn settings around the calibrated
ones, so the model does not depend on any single guess.

## 5. Model

- **Features** (`possession/features.py`): distance from the ball to the three nearest players of
  each team; who has been nearer over the past and next 0.4-16 s; ball touches (velocity changes
  near a player); ball speed, position, distance to the lines; how long the ball has been unseen
  and what the gap looks like; the players' median speed and share of walkers; team centroids,
  spread and defensive lines. The pitch is turned so home always attacks the same way.
- **Two ensembles** of 300 gradient-boosted trees: "is the ball dead?" and "if not, does home
  have it?". Each is evaluated twice, once with the teams exchanged, and the two answers are
  combined, so swapping the teams swaps H and A exactly.
- **Training weights by segment length.** spellF1 counts each possession once. Frames of a
  segment shorter than 12 s are up-weighted (up to 8x) so that a short possession matters as much
  as a long one. On held-out games this alone added about 0.04 spellF1.
- **Decoder** (`possession/decode.py`): Viterbi over H / A / D with a price for each change of
  label, chosen on held-out games. Integer arithmetic.
- The trees are trained with scikit-learn and then **evaluated in plain numpy**
  (`possession/trees.py`), so producing the submission needs only numpy and pandas and gives the
  same bytes on any machine.

## 6. Possession rules we implemented

- **Who has it** is what the training labels say: the team whose player last controlled the
  ball. Loose balls, duels and balls in flight stay with the previous team because that is how
  the labels are written; the decoder's price for changing team removes changes the evidence
  does not sustain.
- **Short touches.** A change of team is kept only if the classifier's evidence outweighs the
  price of two changes. With the segment weights the model keeps about 78% as many possessions
  as the Metrica labels have; the shortest ones (under 1 s) are still mostly merged.
- **Dead ball** runs from the moment play stops to the moment the ball is struck again. The
  model decides both from the ball and, mainly, from the players (they slow down; the ball is
  carried to a spot). Frames of the second half before its kick-off are dead (FAQ.md). Between a
  goal and its kick-off the ball is dead.
- **No abstentions.** We never write X on a graded frame: on spellF1 it only repeats the previous
  label, and on macroF1 it is a miss.

## 7. Validation

Cross-validation by game (a game is never labelled by a model trained on it), scored with the
official `evaluation/possession_metrics.py`. The organisers' baseline is run on the same rendered
tracking; on the real match it scores 0.2433.

| held-out games | simulator setting | our spellF1 | baseline spellF1 | our macroF1 | baseline macroF1 |
|---|---|---|---|---|---|
| DFL (7) | typical | 0.570 | 0.284 | 0.895 | 0.692 |
| DFL (7) | more gaps during play | 0.537 | 0.252 | 0.882 | 0.664 |
| DFL (7) | ball tracked more when dead | 0.566 | 0.267 | 0.893 | 0.616 |
| Metrica (3) | typical | 0.635 | 0.314 | 0.882 | 0.711 |
| Metrica (3) | more gaps during play | 0.626 | 0.286 | 0.873 | 0.676 |
| Metrica (3) | ball tracked more when dead | 0.643 | 0.320 | 0.881 | 0.659 |

These numbers come from simulated tracking and, for Metrica, from labels we derived, so they are
an estimate, not a promise.

## 8. Limitations

- Possessions shorter than about 2 s are mostly missed; they are a quarter of all possessions in
  the Metrica labels.
- A stoppage of 5-6 s whose only trace is a ball gap looks like play that went on unseen, and is
  sometimes missed.
- Stoppages tend to be recognised two to three seconds late when the ball stays in view, and a
  ball passed around during a stoppage can still be read as a short possession.
- The simulator puts the ball near the touchlines about twice as often as the real tracker.
- The possession convention of the training labels (DFL flips on single touches more readily
  than Metrica) is not exactly the competition's.
- With more time: the broadcast feed (replays mark stoppages), a detector for the ball in the
  tactical video, and a second model that scores whole candidate possessions.

## 9. Tools

The code was written with an AI coding assistant (Claude Code, a hosted LLM). The competition's
terms forbid sending match data to such a service, and none was: the assistant never read
`data/`, and every script that touches the match was run by a team member on their own machine.
What the assistant saw of the match is only what those scripts print, which is summary
statistics over the whole match (the profile in `scripts/profile_atd.py`, and the counts,
percentages and scores printed by `generate_submission.py`). No frame, position or tag was shared.

## 10. Reproduction

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt          # numpy, pandas; Python 3.11
# unpack the data bundle into data/ (or set MTAH_DATA_DIR)
.venv/bin/python generate_submission.py                 # writes "outputs/Blaugrana Bytes.csv"
```

The command prints the SHA-256 of the file; ours is
`a320484ed426d4d71c70ce02406d19b65965604b55c524b26a830f49f2c29f5d`. The model in
`models/possession.npz` is included. To retrain it from the public games (about 30 minutes;
scikit-learn's histogram trees are multi-threaded, so a retrained model can differ in the last
digits):

```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python training/fetch.py                      # 2.8 GB of public data
.venv/bin/python training/train.py
```

`python -m pytest tests -q` runs the checks, including that our reader and baseline reproduce
`starter/baseline.py` exactly and that the numpy tree evaluation equals scikit-learn's.
