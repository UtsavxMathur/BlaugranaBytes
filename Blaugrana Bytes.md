# Blaugrana Bytes: how we label possession

This is our entry for the Initial Challenge of More than a Hack 2027. For each of the 199,319
frames of FC Barcelona vs Athletic Club we say whether Barcelona has the ball (H), Athletic has it
(A), or the ball is dead (D). Below we explain what we built and why, what we tried and dropped,
and where we know it still goes wrong.

## The short version

We could not learn from this match, so we learned from other matches. We took public games that
come with a label on every frame, made their tracking look like the tracking we were given, and
trained a classifier on them. A second step turns the classifier's frame-by-frame answers into
whole possessions, and the Smart Tagging file helps us pin down restarts. We also built an
optional step that uses the broadcast recording; for the reason given in "The rules
clarification of 15 November", it is not used for the file we submit.

```
   TRAINING (public games only)                    LABELLING THE MATCH

   7 DFL + 3 Metrica games,                        tactical tracking of the match
   a label on every frame                                       |
              |                                                 v
              v                                     +-----------------------+
   +------------------------+                       | ~110 numbers per frame|
   | simulator: render the  |                       +-----------------------+
   | tracking the way the   |                                   |
   | real tracker would     |                                   v
   +------------------------+                       +-----------------------+
              |                    trained model    | two tree models:      |
              v                  =================> | "dead?"   "home?"     |
   +------------------------+                       +-----------------------+
   | ~110 numbers per frame |                                   |
   +------------------------+                                   v
              |                                     +-----------------------+   Smart Tagging:
              v                                     | decoder: frames into  |<- kick-offs, goals,
   +------------------------+                       | whole possessions     |   set-piece restarts
   | train two tree models, |                       +-----------------------+
   | test on games left out |                                   |
   +------------------------+                                   v
                                                    one label per frame: H, A or D
```

Nothing was labelled by hand, and no label of the provided match was used to train or tune
anything.

## How we got here

We started with the organisers' baseline: whoever is nearest the ball has it, then smooth. Before
anything else we rewrote it on top of our own file reader and checked that it gives the same
answer as theirs: it agrees with `baseline_smooth50.csv` on 100.00% of graded frames and scores
the published 0.3256 on the public window. That told us our reading of the files, the frame
numbering and the home/away mapping were right.

The rules then closed the obvious shortcuts. We may not label this match by hand, and we may not
train or tune on the public 5 minutes. So the examples had to come from somewhere else, and we
found two public sources (section "Training data").

Our first model, trained on those games as they are, looked good on them and much worse on the
real match. The reason turned out to be simple: public tracking is clean, and the tracking we
were given is not. We wrote a script that prints only summary numbers about the real tracking
(how long the ball gaps are, how fast the ball is when it gets lost, how many players are
tracked) and it showed things we had not guessed. For example, there are almost no ball gaps
shorter than 5 seconds. The tracker has already drawn a straight line through them. So we stopped
training on the public games as they are and started rendering them the way this tracker would
have delivered them.

The second lesson came from checking what our labels meant on the public 5 minutes. Barcelona
and Athletic were fine, but we were finding only a third of the dead-ball frames. Our model had
learned "ball missing means ball dead". In the real tracking the ball is often still followed
during a stoppage, while it rolls away, gets thrown back and is carried to the spot. We taught
the simulator to do that, using the players' real movements, and the model now looks mostly at
what the players are doing.

The third lesson was about the metric. spellF1 counts each possession once, but a classifier
trained on frames sees a 40-second possession forty times more often than a 1-second one. We
weight the training frames so that short possessions count as much as long ones.

After we had frozen a first version we built one more step. The broadcast feed comes with its
own tracking, and its homography is not valid on replays, close-ups and crowd shots, which mostly
happen while the ball is dead. The feed is not synchronised with the tactical one, so we lined
the two up from their tracking (section "The broadcast recording"). We then held the step back:
its weights are measured on the provided match itself, and the organisers' clarification of 15
November asks for every tuned value to come from external data.

What we did not do is build our own ball detector from the video. The broadcast tracking already
gives a second camera's view of the ball, and it added little, because both cameras lose the
ball at the same moments. We judged that the provided tracking plus an honest model of how it
fails was the better bet.

## What we used from the bundle, and what we left out

| asset | used? | what we did with it |
|---|---|---|
| Tactical tracking (ball and 46 track slots) | yes | Everything is computed from it. We never treat a slot as a person; we only ask things like "who is nearest", "how many are close", "how fast is the team moving". |
| Smart Tagging | yes, carefully | Tags can be wrong, so we only use one when the tracking or the model agrees with it. Between a goal and the next kick-off the ball is dead, and we enforce that when the model already thinks so for most of the stretch (it did for all four). For throw-ins, corners and goal kicks we look for the moment the ball is struck (it goes from slow or unseen to fast) and mark the seconds before it as dead. |
| `games.py` | yes | The frame ranges of the two halves, and the second-half kick-off. Frames before that kick-off are dead, as the FAQ says. |
| Team tags on phases (Build Up, Progression, ...) | no | We checked them against the nearest player and they agree only 55-76% of the time. That is too weak to help. |
| Free-kick tags | no | There are only five, and they do not agree on when the kick comes relative to the tag. |
| Broadcast tracking | not in the submitted file | Used only by the optional `--broadcast` step: to line the broadcast up with the tactical recording, and to tell live pictures from replays (during live play its players stand where the tactical players stand). |
| Broadcast homography | not in the submitted file | Used only by the optional step: its `valid` flag, since no pitch in the picture means a close-up, a crowd shot or a replay. |
| Broadcast ball | no | Where the tactical ball is missing, the broadcast has it on only 9% of frames. Both cameras lose the ball at the same moments. |
| Tactical homography | no | The tactical camera does follow play, but the centre of the picture is typically 12 m from the ball. That is too rough to tell us where a missing ball is. |
| Speed column of the tracking | no | It matches the speed we compute from positions to within 0.11 m/s, and computing it ourselves lets the same code run on the training games. |
| Videos | no | We never decode pixels. The two trackings carry what we need from the cameras. |

This is how a set-piece tag helps. The tags start a fixed few seconds before the restart, and in
the tracking the restart itself is easy to see: the ball goes from slow or unseen to fast. Left
alone, the model often thinks play has restarted as soon as a player walks to the spot with the
ball. With the tag, the stoppage runs up to the moment the ball is struck. How many seconds after
its tag a kind of set piece is usually struck is worked out by the code on the match itself, from
the tags and the ball speed alone (one median per kind), and a kind whose tags disagree is left
out.

```
set-piece tag          |<------------------- 35 s -------------------->|
ball speed       fast   slow or unseen ........ FAST
                                                ^ struck 5 to 7 s after the tag starts
model alone      A A A D D D D D D D A A A A A A A A A A      play seems to restart early
with the tag     A A A D D D D D D D D D D D D A A A A A      dead up to the strike
```

## Training data

| dataset | games | frames | where the labels come from |
|---|---|---|---|
| DFL open data (Bassek et al., *Scientific Data* 2025, doi 10.6084/m9.figshare.28196177, CC-BY 4.0) | 7 matches of the Bundesliga and 2. Bundesliga, 2022/23 | about 1.0 million at 25 fps | The files carry two flags on every frame, set by the league's official data provider: whether the ball is in play, and which team has it. We label a frame D when the ball is not in play, otherwise H or A. |
| Metrica Sports sample data (github.com/metrica-sports/sample-data) | 3 anonymised games | about 430,000 at 25 fps | These have no frame labels, only event data. We turn the events into labels with fixed rules (`training/metrica.py`): a team has the ball from its pass, carry, shot, recovery or set piece until the other team gets one, and the ball is dead from a ball-out, foul or goal until the next restart. Nobody labelled frames by hand. |

None of this comes from the provided match.

We did take one thing from the provided match: summary statistics of its tracking, printed by
`scripts/profile_atd.py` and listed in `training/calibrate.py`. They are counts, percentages and
medians over the whole match, and we used them to set up the simulator described below.

We want to be clear about the public 5 minutes. We used them to confirm that our reader is
correct, and, as CHALLENGE.md suggests, to understand what the truth means by "dead". That is how
we found the dead-ball problem described above. We fixed it by changing what the simulator does,
and we judged every change on held-out public games. We did not fit any setting to the public
window, and we did not pick between versions by their score on it. For the record, our five
versions scored spellF1 0.51, 0.58, 0.56, 0.66 and 0.64 there, and we submit the last one.

The optional broadcast step could not be judged on held-out games, because the public games have
no broadcast feed. Its weights are measured on the match itself without labels, and we looked at
what it did on the public window as a check: dead-ball recall rose from 0.68 to 0.86 with
precision unchanged at 0.96, macroF1 rose from 0.85 to 0.90, and spellF1 stayed at 0.64. We do
not submit it (see the next section).

## The rules clarification of 15 November

The organisers' clarification says that anything trained or tuned must use external data only,
never the ground truth or the provided video. Our model is trained only on the public games, and
the decoder settings are chosen on held-out public games. We have asked the organisers about
three things that use the provided game's own automatic data, never its ground truth or video:

1. the simulator's settings, which we chose from summary statistics of the provided tactical
   tracking (section "Training data");
2. values the code works out on the match while it runs: the typical delay between a set-piece
   tag and the strike (used in our file), and the broadcast step's weights (not used);
3. looking at the public 5 minutes as a check before keeping a step.

Until they answer, we submit the version without the broadcast step: the one in which nothing is
estimated from our own model's predictions on the match.

## The simulator

The public games have a true ball position on every frame. The provided tracking does not, and
the differences are specific. Our simulator (`training/simulate.py`) reproduces each of them:

- There are 132 ball gaps of 5 seconds or more and almost none shorter. The tracker bridges
  shorter gaps with a straight line, so a ball that looks tracked is often interpolated.
- The ball track is smoothed a lot. A pass shows up as a change of velocity spread over about a
  second.
- The ball is usually lost at a stoppage and found again at the restart. But about one gap in
  four happens while the players are still running, which means play went on unseen.
- The ball is often tracked while it is dead. It keeps moving when play stops, is passed to the
  player who will restart, and is carried by that player.
- A ball in the air is drawn where the camera's line of sight hits the ground, which can be up
  to 20 m off the pitch.
- Players are tracked steadily (10.2 of 11 per team on average) and are occasionally lost.

The picture below shows why a missing ball and a dead ball are not the same thing. The tracker
keeps seeing the ball for a while after play stops and picks it up again before the restart, so
the gap we receive is shorter than the stoppage. A short stoppage may leave no gap at all.

```
                        play stops                            ball is struck again
                            |                                         |
truth        H H H H H H H H|D D D D D D D D D D D D D D D D D D D D D|A A A A A A A
ball seen    ###############|########.........................########|#############
we receive   ---------------|--------           (gap)         --------|-------------
                             \______/                         \______/
                          still tracked                     tracked again
                          after play stops                  before the restart

ball seen    ####......####             we receive  --------------       under 5 s: a straight line
ball seen    ####................####   we receive  ----            ---- 5 s or more: left empty
```

We do not know the exact settings of the real tracker, so every training game is rendered three
times with settings drawn at random around our best estimate. The model then cannot lean on any
single guess of ours.

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

The frames of short possessions and short stoppages get more weight in training (up to 8 times,
for anything shorter than 12 seconds). On held-out games this added about 0.04 spellF1.

The classifier's answers flicker from frame to frame, and every flicker would count as a
possession. A decoder (`possession/decode.py`, the Viterbi algorithm) finds the sequence of
labels that best fits the probabilities while paying a price for every change of label. We chose
the prices on held-out games.

```
classifier, frame by frame    H H H H A H H H H H D H D D D D D D H D D A A A A A
after the decoder             H H H H H H H H H H D D D D D D D D D D D A A A A A
                                      ^             ^             ^
                              each of these flickers would have counted as a possession
```

We train the trees with scikit-learn, then export them and run them ourselves in plain numpy
(`possession/trees.py`). Making the submission therefore needs only numpy and pandas, and the
output is the same file on any machine.

## The broadcast recording (optional, not in the submitted file)

This step runs only with `--broadcast`. We describe it because we built and tested it, and
because it is the best evidence of a stoppage we found.

The broadcast feed has 164,488 frames on its own timeline, with cuts, replays and close-ups, and
no table that maps it to the tactical frames. But its tracking puts players on the same pitch as
the tactical tracking, so whenever the broadcast shows live play its players must stand where the
tactical players stand at that moment. We use that to line the two up
(`possession/broadcast.py`):

1. Coarse: for every 5 minutes of broadcast, we try every shift in steps of one second and count
   the seconds on which both ball tracks are within 3 m of each other. One shift wins clearly.
2. Fine: for every 10 seconds of broadcast, we try shifts frame by frame near the coarse one and
   keep the one at which the broadcast players are closest to the tactical players.
3. A window that disagrees with its neighbours is dropped, and within each stretch the shift is
   the running median of the rest.

On this match the broadcast keeps a fixed delay within each half: 4,014 frames (2 min 40.6 s) in
the first and 26,919 frames (17 min 56.8 s) in the second, where the halftime break was cut. On
live pictures the two trackings put the players a median of 0.28 m apart. That gives every
tactical frame a broadcast state: live pictures (61.6% of graded frames), a replay (2.2%: a pitch
is shown but its players are somewhere else), no pitch at all (31.6%), unclear (2.8%) or no
broadcast frame (1.7%).

A director cuts away from live play a second or more after the ball goes dead, and often comes
back only after the restart, while a replay finishes:

```
truth            H H H H H H|D D D D D D D D D D D D D D D D D D|A A A A A A A
broadcast        live pictures   | replay, close-up, no pitch ......|...| live pictures
                                 ^ cuts away after play stops         ^ often back only
                                                                        after the restart
evidence of D    ---------------- [ strong ................ | weaker ]  -------------
```

So a stretch of non-live pictures is evidence of a dead ball, weaker towards its end. How strong
it is we measure on the match itself, without labels. For each stage of a stretch, counted back
from the moment live pictures return, we compare how often the stage occurs on frames our model
thinks are dead and on frames it thinks are live, using its probabilities as soft counts. More
than 10 s before live pictures return the ratio is 17; in the 10 s before it falls to 5.6, 4.4
and 3.1, because in 32 of 76 of our stoppages the replay was still running when we had play
restart. In the decoder this ratio multiplies the odds of D (it is added as a cost to H and A),
so a frame changes only where the model was unsure. On the full match it changed 1.05% of graded
frames, all from live to dead: 343 possessions and 106 stoppages instead of 344 and 103.

What it cannot do: it does not fix stoppages we notice late, because the director cuts away even
later; a replay shown while play goes on (a goalkeeper holding the ball after a save, say) pushes
the wrong way; and the line-up has only been tested on this match and on made-up recordings.

## Our possession rules

We did not write the rules by hand; the model learned them from the labels. In practice they come
out like this.

The ball belongs to the team whose player last controlled it. A loose ball, a duel or a ball in
flight stays with the previous team, because that is how the training labels are written. The
decoder also refuses a change of team that the evidence does not hold up for long enough.

A short touch becomes a possession only if the classifier is confident enough to pay for two
changes of label, into the new team and back. With the training weights the model keeps about 80%
as many possessions as our Metrica labels contain. The very shortest ones, under a second, are
still mostly merged into their neighbours.

The ball is dead from the moment play stops until it is struck again. The model judges this
mainly from the players: they slow down, and someone carries the ball to a spot. Three stretches
are dead by definition: the second half before its kick-off, the time between a goal and the
kick-off that follows, and the seconds before a set piece is taken.

We never write X on a graded frame. On spellF1 it would only repeat our previous label, and on
macroF1 it counts as a mistake.

## How well it works

We tested by leaving games out: a game is never labelled by a model that was trained on it. The
scores come from the official `evaluation/possession_metrics.py`. The organisers' baseline runs on
the same simulated tracking for comparison; on the real match it scores 0.2433.

| held-out games | simulator setting | our spellF1 | baseline spellF1 | our macroF1 | baseline macroF1 |
|---|---|---|---|---|---|
| DFL (7) | typical | 0.570 | 0.284 | 0.895 | 0.692 |
| DFL (7) | more gaps during play | 0.537 | 0.252 | 0.882 | 0.664 |
| DFL (7) | ball tracked more when dead | 0.566 | 0.267 | 0.893 | 0.616 |
| Metrica (3) | typical | 0.635 | 0.314 | 0.882 | 0.711 |
| Metrica (3) | more gaps during play | 0.626 | 0.286 | 0.873 | 0.676 |
| Metrica (3) | ball tracked more when dead | 0.643 | 0.320 | 0.881 | 0.659 |

```
spellF1 on games the model never saw (typical simulator setting)

Metrica   ours      ################################  0.64
          baseline  ################                  0.31
DFL       ours      #############################     0.57
          baseline  ##############                    0.28
                    |         |         |         |
                    0        0.2       0.4       0.6
```

So on games it has never seen, the model scores about twice what the baseline does. Please read
this as an estimate. The tracking in these tests is simulated, and the Metrica labels are ones we
derived ourselves.

## Where it still fails

- Possessions shorter than about 2 seconds are mostly missed. In the Metrica labels they are a
  quarter of all possessions.
- A stoppage of 5 or 6 seconds that shows up only as a ball gap looks just like play that went on
  unseen, and we sometimes miss it.
- When the ball stays in view, we tend to notice a stoppage two or three seconds late. A ball
  passed around during a stoppage can still be read as a short possession. On the public window
  we find 68% of the dead-ball frames.
- Our simulator puts the ball near the touchlines about twice as often as the real tracker does.
- The two training sources do not agree with each other, or exactly with the competition, on
  what counts as a change of possession. The DFL flags change on single touches more readily
  than the Metrica events do.

With more time we would try detecting the ball in the tactical video ourselves, find the start of
a stoppage from the moment the ball crosses a line, and add a second model that judges whole
candidate possessions and not single frames.

## Tools we used

We wrote the code with the help of an AI coding assistant (Claude Code, a hosted LLM). The
competition's terms forbid sending match data to such a service, and we did not. The assistant
never read the `data/` folder, and every script that touches the match was run by a team member
on their own machine. All the assistant saw of the match is what those scripts print: summary
statistics over the whole match from `scripts/profile_atd.py` and `scripts/broadcast_report.py`,
and the counts, percentages and scores printed by `generate_submission.py`. No frame, position or
tag was shared.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt          # numpy and pandas, Python 3.11
# unpack the data bundle into data/ (or set MTAH_DATA_DIR); data/tactical/ is all it reads
.venv/bin/python generate_submission.py                 # writes "outputs/Blaugrana Bytes.csv"
```

The command prints the SHA-256 of the file it wrote. Ours is
`a320484ed426d4d71c70ce02406d19b65965604b55c524b26a830f49f2c29f5d`, and we got the same value
from a fresh clone of the repository. With `--broadcast` (which also needs `data/broadcast/`) it
adds the optional broadcast step and writes
`b08a4d78f28889f4719d9b3c5560887eb14b42aae0418ef43891b7838e08cde2`.

The trained model is in the repository (`models/possession.npz`). To train it again from the
public games:

```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python training/fetch.py                      # downloads 2.8 GB of public data
.venv/bin/python training/train.py                      # about 30 minutes
```

A retrained model can differ slightly from ours, because scikit-learn builds its trees on
several threads. The submitted file comes from the model in the repository.

`.venv/bin/python -m pytest tests -q` runs our checks. Among other things they confirm that our
reader and baseline give exactly what `starter/baseline.py` gives, that running the trees in
numpy gives exactly what scikit-learn gives, and that the broadcast line-up recovers a known
delay, cut and mirrored axes on made-up recordings.
