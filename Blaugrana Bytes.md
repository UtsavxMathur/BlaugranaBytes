# Blaugrana Bytes: how we label possession

This is our entry for the Initial Challenge of More than a Hack 2027. For each of the 199,319
frames of FC Barcelona vs Athletic Club we say whether Barcelona has the ball (H), Athletic has it
(A), or the ball is dead (D). Below we explain what we built and why, what we tried and dropped,
and where we know it still goes wrong.

## For reviewers

One command regenerates our file: `python generate_submission.py` (setup in "Running it"; about a
minute). It writes `outputs/Blaugrana Bytes.csv`, sha256
`260ef1fc052f844b7926118467a395d41a9e5004acc978c97b93e239203c6f4d`, and a fresh clone of the
public repository reproduces it byte for byte.

How the method stays within the rules (the organisers' clarification of 15 November, and their
reminder that anything not explicitly allowed is not permitted):

| what the method does | the explicit permission |
|---|---|
| reads the provided tactical tracking and Smart Tagging as input | "Using as input ... the automatic tracking data, the smart tagging" |
| marks the frames before the second-half kick-off as dead, from the organisers' `games.py` | FAQ: frames before the kick-off are dead |
| trains every model on public games only (DFL, Metrica) | "If you need to train ... you MUST use only external data"; RULES.md: "External data about other games" |
| chooses every setting (decoder prices, designs) on held-out public games | "tune them on external data" |
| draws the training simulator's settings from ranges written down in advance | nothing is tuned on the provided data |
| designed its features after inspecting the data | "Inspecting the provided data ... to better understand"; RULES.md: "inspecting the data to ... design features" |
| checked its output on the match with counts only | "Running automatic checks using only the provided data" |

Not used for this file: the videos, the broadcast recording, hand labels of any kind, any value
measured on the match, and the ground truth of the public 5 minutes.

Data and acknowledgements: DFL open data (Bassek et al., *Scientific Data* 2025,
doi 10.6084/m9.figshare.28196177), CC-BY 4.0. Metrica Sports sample data
(github.com/metrica-sports/sample-data), used as its README asks: responsibly, with the source
acknowledged here. SkillCorner open data (MIT) was tried and is not used in the submitted models.

## The short version

We could not learn from this match, so we learned from other matches. We took public games that
come with a label on every frame, rewrote those labels so that possession changes only when a
team gains control of the ball (the challenge's definition), damaged their tracking the way an
automatic tracker can (over a wide range of trackers fixed in advance, not fitted to the tracking
we were given), and trained two kinds of model on them: gradient-boosted trees that judge each
frame, and two sequence networks that read about 25 seconds either side. Their answers are
averaged, the trees counting for half. A second step turns those frame-by-frame answers into
whole possessions, and the Smart Tagging file marks the stoppages between a goal and the
kick-off. We also built an optional step that uses the broadcast recording; it is not used for
the file we submit.

This is our strict version: nothing in it is trained, tuned or set from the provided match, its
video or its ground truth (section "The rules clarification of 15 November").

```
   TRAINING (public games only)                    LABELLING THE MATCH

   7 DFL + 3 Metrica games,                        tactical tracking of the match
   a label on every frame                                       |
              |                                                 v
              v                                     +-----------------------+
   +------------------------+                       | ~110 numbers per frame|
   | simulator: damage the  |                       +-----------------------+
   | tracking like many     |                                   |
   | automatic trackers     |                                   v
   +------------------------+                       +-----------------------+
              |                    trained models   | trees + two sequence  |
              v                  =================> | networks, averaged:   |
   +------------------------+                       | "dead?"   "home?"     |
   | ~110 numbers per frame |                       +-----------------------+
   +------------------------+                                   |
              |                                                 v
              v                                     +-----------------------+   Smart Tagging:
   +------------------------+                       | decoder: frames into  |<- goal to kick-off
   | train the trees and    |                       | whole possessions     |   (if the model agrees)
   | the networks, test on  |                       +-----------------------+
   | games left out         |                                   |
   +------------------------+                                   v
                                                    one label per frame: H, A or D
```

Nothing was labelled by hand, and nothing from the provided match (its labels, its video or
statistics of its tracking) was used to train, tune or set anything.

## How we got here

We first rebuilt the organisers' baseline on our own file reader. It agrees with
`baseline_smooth50.csv` on 100.00% of graded frames, which told us our reading of the files, the
frame numbering and the home/away mapping were right. The rules then closed the obvious shortcuts
(no hand labels, no training or tuning on this match), so the examples had to come from other
games (section "Training data"). What we learned on the way, in order:

1. **Public tracking is clean; ours is not.** A first model trained on the public games as they
   are did much worse on the real match. A script that prints only summary numbers about the real
   tracking (counts and medians, no frames) showed how it fails, for example almost no ball gaps
   under 5 seconds because the tracker bridges them. So we render the training games the way an
   automatic tracker delivers them (section "The simulator"). An earlier version set the
   simulator from those numbers; the version we submit does not (see below).
2. **A missing ball is not a dead ball.** Checking what our labels meant on the public 5 minutes
   showed we found only a third of the dead-ball frames: the model had learned "ball missing
   means ball dead". Real trackers often follow the ball through a stoppage while it rolls away,
   is thrown back and is carried to the spot. The simulator does that now, so the model looks
   mostly at what the players are doing.
3. **The metric counts possessions, not frames.** A frame classifier sees a 40-second possession
   forty times more often than a 1-second one, so we weight the training frames so that short
   possessions count as much as long ones.
4. **The labels must follow the challenge's definition.** Both sources change possession on
   single touches, about twice as often per minute as the definition allows. Relabelling them
   with one rule taken from the definition (section "Training data") gained about 0.10 spellF1 on
   held-out games.
5. **Timing needs a model that sees time.** Changes of team were the least precise edges, so we
   added sequence networks that read about 25 seconds either side (section "The sequence
   network"). Averaged with the trees, the first gained about 0.02 spellF1 on held-out games, a
   second 0.008 and a wider first one about 0.007.

We also built a step that uses the broadcast recording, and held it back because its weights
come from the match itself (section "The broadcast recording").

Ideas that failed the held-out test, and that we left out:

- **Twenty more public matches** (SkillCorner, broadcast tracking at 10 fps): -0.021 spellF1.
  Their off-screen players are estimated, and the model learned a cue our tracking does not have.
- **A side task** for the sequence network (where is the true ball while the tracker has lost
  it): within the noise, and frames without a ball got no more accurate.
- **A second model judging whole candidate possessions**, aimed at the short spells of the other
  team we invent: +0.004, no better than the decoder's own evidence.
- **Network outputs marking changes, stoppages and restarts**, used as decoder prices: +0.003
  with each fold's setting chosen on the other folds. Our second network was trained with these
  outputs; we use only its "dead?" and "home?" answers.
- **More of what already worked**: twenty-four simulated trackers per game instead of twelve,
  and a different share for the trees in the average: both exactly +0.000.

What we did not do is build our own ball detector from the video. The broadcast tracking already
gives a second camera's view of the ball, and it added little, because both cameras lose the
ball at the same moments. A detector would also have had to be set up without this match, and
the public games have no video to test it on.

**The version we submit is a strict rebuild.** Two steps of our earlier versions took a number
from the match itself, without labels: the simulator's settings were chosen from summary statistics of the
provided tracking, and the restart timing used a typical delay per kind of set piece measured on
the match while the code ran. The organisers' clarification asks for every tuned value to come
from external data, so we removed both. The simulator now draws every setting from a wide range
written down in advance (section "The simulator"), the restart timing is gone, and every model,
every decoder price and the choice between our network designs were made again on held-out
public games under the new simulator. Last, giving every model twelve simulated trackers per
game instead of three added about 0.01 (section "The sequence network"). The gains quoted in the
lessons above were measured with the earlier simulator; the tables in "How well it works" are for
this version.

## What we used from the bundle, and what we left out

| asset | used? | what we did with it |
|---|---|---|
| Tactical tracking (ball and 46 track slots) | yes | Everything is computed from it. We never treat a slot as a person; we only ask things like "who is nearest", "how many are close", "how fast is the team moving". |
| Smart Tagging | yes, carefully | Tags can be wrong, so we only use one when the model agrees with it. Between a goal and the next kick-off the ball is dead, and we enforce that when the model already thinks so for at least half of the stretch. |
| `games.py` | yes | The frame ranges of the two halves, and the second-half kick-off. Frames before that kick-off are dead, as the FAQ says. |
| Team tags on phases (Build Up, Progression, ...) | no | We checked them against the nearest player and they agree only 55-76% of the time. That is too weak to help. |
| Set-piece tags (throw-ins, corners, goal kicks, free kicks) | no | An earlier version timed restarts from them, with a typical delay per kind measured on this match. That is a value taken from the match, so this version leaves restarts to the model. |
| Broadcast recording (tracking, homography, ball) | no | Used only by the optional `--broadcast` step, which is not part of our file (section "The broadcast recording"). Its ball adds little anyway: where the tactical ball is missing, the broadcast has it on only 9% of frames. |
| Tactical homography | no | The tactical camera does follow play, but the centre of the picture is typically 12 m from the ball. That is too rough to tell us where a missing ball is. |
| Speed column of the tracking | no | It matches the speed we compute from positions to within 0.11 m/s, and computing it ourselves lets the same code run on the training games. |
| Videos | no | We never decode pixels. The two trackings carry what we need from the cameras. |

## Training data

| dataset | games | frames | where the labels come from |
|---|---|---|---|
| DFL open data (Bassek et al., *Scientific Data* 2025, doi 10.6084/m9.figshare.28196177, CC-BY 4.0) | 7 matches of the Bundesliga and 2. Bundesliga, 2022/23 | about 1.0 million at 25 fps | The files carry two flags on every frame, set by the league's official data provider: whether the ball is in play, and which team has it. We label a frame D when the ball is not in play, otherwise H or A. |
| Metrica Sports sample data (github.com/metrica-sports/sample-data) | 3 anonymised games | about 430,000 at 25 fps | These have no frame labels, only event data. We turn the events into labels with fixed rules (`training/metrica.py`): a team has the ball from its pass, carry, shot, recovery or set piece until the other team gets one, and the ball is dead from a ball-out, foul or goal until the next restart. Nobody labelled frames by hand. |

None of this comes from the provided match.

Then we put both sources on one convention, the challenge's: possession changes only when a
player of the other team gains *controlled* possession (`training/ownership.py`). Every
possession that follows the other team's directly is checked against the clean tracking of the
public game. It counts only if a player of the new team keeps the ball for a second (as the
nearest player, within 1.5 m, with the ball below 2 m), or if two of its players pass between
them. Otherwise it was a touch or a deflection, and its frames go back to the team that had the
ball. The first possession after a stoppage always counts, and dead-ball frames never change.
These constants come from the definition and from what a touch looks like at 25 frames per
second, not from the provided match.

| | possessions per minute of play | possessions under 2 s |
|---|---|---|
| DFL, labels as given | 8.3 | 25-37% per game |
| DFL, controlled possession | 4.3 | 2-9% |
| Metrica, labels as given | 6.1 | 24-29% |
| Metrica, controlled possession | 3.8 | 2-9% |

For comparison, the public 5 minutes have 4.2 possessions per minute of play. We did not fit
anything to that number; it only tells us that this reading of the definition is close to the
truth's.

An earlier version took one thing from the provided match: summary statistics of its tracking
(counts, percentages and medians over the whole match), used to set up the simulator. This
version does not: its simulator settings are drawn from ranges fixed in advance (section "The
simulator").

We want to be clear about the public 5 minutes. We used them to confirm that our reader is
correct, and, as CHALLENGE.md suggests, to understand what the truth means by "dead". That is how
we found the dead-ball problem described above. We fixed it by changing what the simulator does,
and we judged every change on held-out public games. We did not fit any setting to the public
window, and we did not pick between versions by their score on it. For the record, our first
five versions scored spellF1 0.51, 0.58, 0.56, 0.66 and 0.64 there. The version trained on the
controlled-possession labels scores 0.65, the one that adds the sequence network 0.64, and the
one with a second network 0.64 as well. We chose all three on held-out games, where they gained
about 0.10, 0.02 and 0.008, and ran them on the public window only to check that nothing had
broken. For the last change (a wider first network, +0.007 on held-out games) we did not look at
the public window at all: its check on the match used only counts over the whole match (the
number of possessions and stoppages, and how many frames changed), so that no ground truth played
any part in the decision. The same holds for this strict version: it was not run on the public
window, and its check on the match used only counts over the whole match.

## The rules clarification of 15 November

The organisers' clarification says that anything trained or tuned must use external data only,
never the ground truth or the provided video; the table in "For reviewers" shows how each part of
the method fits. An earlier version took two values from the match's own automatic data (never
its ground truth or its video): the simulator's settings, from summary statistics of the
tactical tracking, and a typical delay between a set-piece tag and the strike, worked out while
the code ran. We asked the organisers about both, and removed them for this version so that no
reading of the rules is needed.

## The simulator

The public games have a true ball position on every frame. Automatic tracking does not, and our
simulator (`training/simulate.py`) damages the public tracking the way an automatic tracker can:

- it loses the ball in live play: in the air, among players, when the ball lies or is held still;
- it may bridge short gaps with a straight line, so a ball that looks tracked can be interpolated;
- it smooths the ball track and adds detection noise;
- it often keeps following the ball for a while after play stops, tracks passes on the way to
  the player who restarts, and picks the ball up again before the restart;
- a ball in the air is drawn where the camera's line of sight meets the ground;
- it loses players the camera does not show, now and then a single player, now and then half
  of them at once.

The picture below shows why a missing ball and a dead ball are not the same thing. A tracker that
keeps seeing the ball for a while after play stops, and picks it up again before the restart,
leaves a gap shorter than the stoppage. A short stoppage may leave no gap at all.

```
                        play stops                            ball is struck again
                            |                                         |
truth        H H H H H H H H|D D D D D D D D D D D D D D D D D D D D D|A A A A A A A
ball seen    ###############|########.........................########|#############
we receive   ---------------|--------           (gap)         --------|-------------
                             \______/                         \______/
                          still tracked                     tracked again
                          after play stops                  before the restart

ball seen    ####......####             we receive  --------------       short gap: bridged
ball seen    ####................####   we receive  ----            ---- long gap: left empty
```

How much of each a tracker does, we do not take from the match. Every training copy of every
game draws each setting from a wide range written down in advance, from a near-perfect tracker
to a poor one, so the model has to cope with all of them:

| what the tracker does | range drawn for each training copy |
|---|---|
| bridges ball gaps shorter than | 0 to 6 s |
| smooths the ball track / adds detection noise | 0 to 0.5 s / 0 to 0.5 m |
| loses the ball in live play | 0 to 8 times a minute, typically for 0.5 to 3 s |
| loses a ball lying or held still | 0 to 100% of the time |
| loses a ball in the air (above 1 to 4 m) | 0 to 100% of the time; always above 4 to 15 m |
| keeps tracking after play stops | 0 to 5 s on the pitch, 0 to 3 s after it went out |
| tracks the ball before the restart | with a chance of 0 to 100%, for 0 to 6 s |
| tracks a pass during a stoppage / the whole stoppage | 0 to 100% / 0 to 50% of stoppages |
| camera height / distance behind the touchline | 10 to 40 m / 10 to 60 m |
| loses players far from the action (30 to 70 m away) | 0 to 100% of the time |
| loses single players / half the players at once | 0 to 10% / 0 to 10% of the time |

Every kind of restart (goal kick, corner, other) gets the same ranges: we assume nothing about
how this tracker treats them. Held-out tests use three fixed settings across these ranges: low,
mid and high damage.

## The model

For every frame we compute about 110 numbers (`possession/features.py`). They describe how far
the ball is from the three nearest players of each team, which team has been nearer over the last
and next 0.4 to 16 seconds, when each team last touched the ball, how fast the ball is and where
it is on the pitch, how long it has been unseen, how fast the players are moving and how many are
walking, and the shape of each team. We turn the pitch so that the home team always attacks in
the same direction.

Two models of 300 gradient-boosted trees each answer two questions: is the ball dead, and if not,
does the home team have it? We ask each question twice, the second time with the teams swapped,
and combine the answers. This guarantees that the model has no built-in preference for home or
away.

The "home?" model is trained on the controlled-possession labels. The "dead?" model is trained on
the labels as the files give them: the new convention changes who owns the ball, never whether it
is dead, so that model must not change with it (`training/combine.py` puts the two together).

The frames of short possessions and short stoppages get more weight in training (up to 8 times,
for anything shorter than 12 seconds). On held-out games this added about 0.04 spellF1.

The classifier's answers flicker from frame to frame, and every flicker would count as a
possession. A decoder (`possession/decode.py`, the Viterbi algorithm) finds the sequence of
labels that best fits the probabilities while paying a price for every change of label. We chose
the prices on held-out games, for the average we submit (the trees and both networks of the next
section): a change of team costs 3,000, entering a stoppage 4,000 and leaving one 4,000, where a
single frame can cost at most 600. A change of team may never cost more than a stoppage plus a
restart: otherwise the decoder would rather slip a one-frame stoppage into every change of team.

```
classifier, frame by frame    H H H H A H H H H H D H D D D D D D H D D A A A A A
after the decoder             H H H H H H H H H H D D D D D D D D D D D A A A A A
                                      ^             ^             ^
                              each of these flickers would have counted as a possession
```

We train the trees with scikit-learn, then export them and run them ourselves in plain numpy
(`possession/trees.py`). Making the submission therefore needs only numpy and pandas, and the
output is the same file on any machine.

## The sequence network

The trees judge each frame on its own, from numbers that summarise windows around it. The sequence
network (`training/sequence.py`) reads the whole sequence of those same 110 numbers, at 5 frames
per second, with a temporal convolutional network: seven layers of 1-D convolutions whose taps
spread 1, 2, 4 ... 64 steps apart, so each answer looks about 25 seconds back and 25 seconds
ahead. It answers the same two questions, "dead?" and "home?", and is trained with PyTorch on
the same simulated public games, the same controlled-possession labels and the same weights as
the trees. Like the trees, it is asked twice, once with the teams swapped, so it cannot prefer
either side.

The two models are combined by averaging their log-odds, question by question. On held-out games
that average beat the trees alone in nine of the ten games, by +0.019 spellF1 (twice the
standard error across games is 0.009). It mostly helps with timing: on the DFL games, changes of
team placed within a second of the truth rose from 49-50% to 56-57%.

We then trained a second network in the same way, with three extra outputs that mark where a
change of team, a stoppage and a restart happen. Those outputs did not help the decoder (see "How
we got here") and are dropped, but the network's own "dead?" and "home?" answers are a second
opinion. The submission averages the two networks, and averages that with the trees, so the trees
count for half and each network for a quarter. On held-out games this gained another +0.008
spellF1, better in nine of the ten games (twice the standard error: 0.003). The gain held when
the decoder's prices were left as they were (+0.0077), and when each fold's prices were chosen
on the other folds only (+0.0086).

Last, we tried three changes to how the first network is built and trained (with our earlier
simulator), deciding before we ran them that each had to gain at least 0.005 by more than twice its standard error. Six
simulated copies of each game instead of three, twice the training steps with a learning rate
that falls to zero, and a little dropout gained only +0.005, in six of ten games, so we did not
take it. The same recipe with a network twice as wide (128 channels) gained +0.0073 in the
average, +0.0081 when each fold's prices were chosen on the other folds, and its prices came out
the same as before; on its own this network beats the narrow one by +0.012, in nine of ten games.
The submission uses it in place of the first network. Training the second network the same way
did not add anything (-0.0003), so it stays as it was.

All the comparisons in this section were made with our earlier simulator. For the strict version
we made the choice again with the strict one, on held-out games, each blend with its own prices
and each fold's prices chosen on the other folds, under a rule written down before the run (keep
the trees with the wide and the second network unless another blend beats it by 0.005 and by
twice its standard error). The trees with the wide and the second network score 0.760; with the
narrow network instead of the wide one, 0.754; with the wide network alone, 0.751; and the same
three-model average with trees trained only on the controlled labels, 0.761, which is within the
noise. We kept the first. The trees alone score 0.727.

Last, we gave every model more variety. The strict simulator draws a different tracker for each
training copy of a game, but the trees and the second network had seen only three copies per
game, and the wide network six. With twelve copies per game for all of them, at the same amount
of computation (the trees keep every 16th frame of each copy instead of every 4th), the average
gained +0.0096 on held-out games with each fold's prices chosen on the other folds, better in nine
of the ten (twice the standard error: 0.006), under a rule written down before the run. The
submitted models are trained this way.

PyTorch is only used for training. A floating-point network can add its sums in a different order
on another machine and change a last digit, so for the submission the weights are stored as whole
numbers (`models/sequence_wide.npz`, `models/sequence2.npz`) and run in integer arithmetic by
`possession/seqnet.py`: every product and sum is exact. For the first network that version gives
the same labels as the floating-point one on 99.9997% of 4.3 million held-out frames; with both
networks the held-out scores are the same to four decimals. A test checks it against PyTorch.

## The broadcast recording (built, not used)

We also built an optional step, `--broadcast`, that lines the broadcast recording up with the
tactical one from their tracking (the broadcast has its own timeline, with cuts and replays) and
treats replays and close-ups as evidence of a dead ball. Its weights would be measured on the match
itself, so it is not part of our file and none of the results in this document use it. The code
is in `possession/broadcast.py`.

## Our possession rules

We did not write the rules by hand; the model learned them from the labels. In practice they come
out like this.

The ball belongs to the team whose player last had it under control. A touch, a deflection or a
duel without control does not change that, and a loose ball or a ball in flight stays with the
team that had it: that is how we rewrote the training labels, so it is what the model learned.
The decoder also refuses a change of team that the evidence does not hold up for long enough.

Under this rule very short possessions are rare (2-9% of the relabelled ones last under 2
seconds), and the model finds only about a third of them. On held-out games it
predicts about 7% fewer possessions than the truth has.

The ball is dead from the moment play stops until it is struck again. The model judges this
mainly from the players: they slow down, and someone carries the ball to a spot. Two stretches
are dead by definition: the second half before its kick-off, and the time between a goal and the
kick-off that follows. The seconds before a set piece is taken are left to the model.

We never write X on a graded frame. On spellF1 it would only repeat our previous label, and on
macroF1 it counts as a mistake.

## How well it works

We tested by leaving games out: a game is never labelled by a model that was trained on it. The
scores come from the official `evaluation/possession_metrics.py`, against the controlled-possession
labels, under the three simulator settings of section "The simulator" (low, mid and high damage).
The trees alone and the organisers' baseline run on the same simulated tracking for comparison;
on the real match the baseline scores 0.2433.

| held-out games | simulator setting | our spellF1 | trees alone | baseline | our macroF1 | baseline macroF1 |
|---|---|---|---|---|---|---|
| DFL (7) | low damage | 0.774 | 0.758 | 0.336 | 0.931 | 0.726 |
| DFL (7) | mid | 0.763 | 0.732 | 0.249 | 0.928 | 0.611 |
| DFL (7) | high damage | 0.744 | 0.688 | 0.179 | 0.915 | 0.501 |
| Metrica (3) | low damage | 0.827 | 0.810 | 0.369 | 0.915 | 0.761 |
| Metrica (3) | mid | 0.792 | 0.768 | 0.268 | 0.904 | 0.616 |
| Metrica (3) | high damage | 0.761 | 0.705 | 0.187 | 0.888 | 0.527 |

```
spellF1 on games the model never saw (mid simulator setting)

Metrica   ours         ########################################  0.79
          trees alone  ######################################   0.77
          baseline     #############                            0.27
DFL       ours         ######################################   0.76
          trees alone  #####################################    0.73
          baseline     ############                             0.25
                       |         |         |         |
                       0        0.2       0.4       0.6
```

So on games it has never seen, the model scores two to four times what the baseline does, and
the networks add 0.02 to 0.06 to the trees alone. These numbers are not comparable with those of
our earlier versions, because the simulated test data changed. Please read them as an estimate:
the tracking in these tests is simulated, and the labels are our reading of the definition.

## Where it still fails

- Possessions shorter than about 2 seconds are mostly missed, and a short spell of the other
  team inside a long possession tends to be merged away: on the public window the version
  before the wide network made 11 possessions where the truth has 17.
- A stoppage of 5 or 6 seconds that shows up only as a ball gap looks just like play that went on
  unseen, and we sometimes miss it.
- When the ball stays in view, we tend to notice a stoppage two or three seconds late, and a
  stoppage during which the ball is passed around can be read as play going on. On the public
  window the version before the wide network found 63% of the dead-ball frames; most of the
  rest is one 14.5-second stoppage with the ball in view throughout.
- This version's simulator spans a wide range of trackers instead of being matched to the one we
  were given. That makes the model robust, but it may be less sharp on this match than a
  matched one would be, and we cannot measure the difference without the ground truth.
- Our rule for controlled possession is our reading of the definition. The corrected tracking
  behind the truth may draw the line elsewhere, for example after more than a second of control.

With more time we would try to find the start of a stoppage from the moment the ball crosses a
line. We decided against detecting the ball in the video ourselves: the public games have no
video, so it could not be tested on them, and any way of setting it up would have depended on
this match. A second model that judges whole candidate
possessions, which we tried, did not help (see "How we got here"): the details that decide
whether a team gained control are smoothed away in the tracking we receive.

## Tools we used

We wrote the code with the help of an AI coding assistant (Claude Code, a hosted LLM). The
competition's terms forbid sending match data to such a service, and we did not. The assistant
never read the `data/` folder, and every script that touches the match was run by a team member
on their own machine. All the assistant saw of the match is what those scripts print: summary
statistics over the whole match from `scripts/profile_atd.py` and `scripts/broadcast_report.py`,
and the counts, percentages and scores printed by `generate_submission.py`. No frame, position or
tag was shared. For this strict version it saw only counts printed by `generate_submission.py`
(numbers of possessions and stoppages, shares of frames), and nothing it saw set any value.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt          # numpy and pandas, Python 3.11
# unpack the data bundle into data/ (or set MTAH_DATA_DIR); data/tactical/ is all it reads
.venv/bin/python generate_submission.py                 # writes "outputs/Blaugrana Bytes.csv"
```

The command prints the SHA-256 of the file it wrote. Ours is
`260ef1fc052f844b7926118467a395d41a9e5004acc978c97b93e239203c6f4d`, and we got the same value from a fresh clone of the repository. It takes about
a minute. Our earlier versions, whose training simulator was set from summary statistics of the
match's tracking, are in the git history (commit `6630f72` and before). `--broadcast` (which
also needs `data/broadcast/`) adds the optional broadcast step, which is not part of our file.

The trained models are in the repository (`models/possession.npz`, `models/sequence_wide.npz`,
`models/sequence2.npz`). To build them again from the public games:

```bash
.venv/bin/pip install -r requirements-train.txt          # scikit-learn and PyTorch
.venv/bin/python training/fetch.py --only metrica         # 190 MB of public data
.venv/bin/python training/fetch.py --only dfl             # 2.6 GB
# the trees: "dead?" from the labels as given, "home?" from controlled possession, combined
# (12 simulated copies of each game, every 16th frame of each)
.venv/bin/python training/train.py --copies 12 --thin 16 --work-dir .cache/strict/train \
    --out .cache/strict/models/possession_v1.npz
.venv/bin/python training/train.py --labels controlled --copies 12 --thin 16 \
    --work-dir .cache/strict/train_controlled --out .cache/strict/models/possession_controlled.npz
.venv/bin/python training/combine.py --dead-model .cache/strict/models/possession_v1.npz \
    --dead-work-dir .cache/strict/train --team-model .cache/strict/models/possession_controlled.npz \
    --team-work-dir .cache/strict/train_controlled --work-dir .cache/strict/train_hybrid
# the wide network: held-out predictions, in whole numbers
.venv/bin/python training/sequence.py --steps cv int --copies 12 --train-steps 3000 --cosine \
    --dropout 0.1 --width 128 --work-dir .cache/strict/train_seqB --out .cache/strict/models/sequence_wide.pt
# the second network: held-out predictions, the average of all three and its prices, the final network
.venv/bin/python training/sequence.py --change --copies 12 --with .cache/strict/train_seqB_int \
    --trees .cache/strict/train_hybrid --name strict2 --compare none \
    --work-dir .cache/strict/train_seq_change --out .cache/strict/models/sequence2.pt --export models/sequence2.npz
# the final wide network, with the same prices
.venv/bin/python training/sequence.py --steps final --copies 12 --train-steps 3000 --cosine \
    --dropout 0.1 --width 128 --work-dir .cache/strict/train_seqB --out .cache/strict/models/sequence_wide.pt \
    --export models/sequence_wide.npz --prices .cache/strict/train_seq_change_blend/prices.json
```

Each tree run takes about 30 minutes, the wide network about 50 and the second network about 20.
A retrained model can differ slightly from ours, because scikit-learn and PyTorch add up some
sums on several threads or on the graphics processor. The submitted file comes from the models in
the repository.

`.venv/bin/python -m pytest tests -q` runs our checks. Among other things they confirm that our
reader and baseline give exactly what `starter/baseline.py` gives, that running the trees in
numpy gives exactly what scikit-learn gives, that the whole-number network matches PyTorch, that
swapping the teams swaps home and away exactly, that the relabelling treats touches, holds and
passes as described above, that the whole pipeline still gives the same output on a made-up
match, and that the broadcast line-up recovers a known delay, cut and mirrored axes on made-up
recordings.
