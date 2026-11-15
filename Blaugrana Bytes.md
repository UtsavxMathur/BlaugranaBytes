# Blaugrana Bytes: how we label possession

This is our entry for the Initial Challenge of More than a Hack 2027. For each of the 199,319
frames of FC Barcelona vs Athletic Club we say whether Barcelona has the ball (H), Athletic has it
(A), or the ball is dead (D). Below we explain what we built and why, what we tried and dropped,
and where we know it still goes wrong.

## The short version

We could not learn from this match, so we learned from other matches. We took public games that
come with a label on every frame, rewrote those labels so that possession changes only when a
team gains control of the ball (the challenge's definition), made their tracking look like the
tracking we were given, and trained two kinds of model on them: gradient-boosted trees that judge
each frame, and two sequence networks that read about 25 seconds either side. Their answers are
averaged, the trees counting for half. A second step turns those frame-by-frame answers into
whole possessions, and the Smart
Tagging file helps us pin down restarts. We also built an
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
              |                    trained models   | trees + two sequence  |
              v                  =================> | networks, averaged:   |
   +------------------------+                       | "dead?"   "home?"     |
   | ~110 numbers per frame |                       +-----------------------+
   +------------------------+                                   |
              |                                                 v
              v                                     +-----------------------+   Smart Tagging:
   +------------------------+                       | decoder: frames into  |<- kick-offs, goals,
   | train the trees and    |                       | whole possessions     |   set-piece restarts
   | the network, test on   |                       +-----------------------+
   | games left out         |                                   |
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

The fourth lesson was about the labels themselves. Our two sources each follow their own
convention: DFL's possession flag changes team on a single touch, and our Metrica labels change
at the other team's events. Both give about twice as many possessions per minute of play as the
challenge's definition allows, where a touch, a deflection or a duel without control is not a
change of possession. Our model had learned to split possessions the same way. We rewrote the
training labels with one rule taken from that definition (section "Training data"), and on games
it had not seen the model trained on them scored about 0.10 higher.

The fifth lesson was about timing. A breakdown of the remaining errors showed that the edges of
possessions cost the most at the strict IoU thresholds, and that changes of team were the least
precise edges. Trees judge one frame at a time, so we added a network that reads the sequence
(section "The sequence network"). On its own it was only a little better; averaged with the
trees it gained about 0.02 spellF1 on held-out games, in nine of the ten. A second network of the
same kind, averaged in as well, added another 0.008, again in nine of the ten. Making the first
network twice as wide, and training it longer on more simulated copies, added about 0.007 more.

After we had frozen a first version we built one more step. The broadcast feed comes with its
own tracking, and its homography is not valid on replays, close-ups and crowd shots, which mostly
happen while the ball is dead. The feed is not synchronised with the tactical one, so we lined
the two up from their tracking (section "The broadcast recording"). We then held the step back:
its weights are measured on the provided match itself, and the organisers' clarification of 15
November asks for every tuned value to come from external data.

Four more ideas failed the held-out test, and we left them out. Twenty more public matches
(SkillCorner's A-League open data, broadcast tracking at 10 frames per second) made the model
worse on the held-out DFL and Metrica games, by 0.021 spellF1. Their off-screen players are
estimated, and the model learned a team-shape cue from them that our tracking does not have. We
also taught the sequence network to predict, as a side task, where the true ball is while the
tracker has lost it (the simulator knows). The score moved by less than the noise, and the frames
without a ball got no more accurate. The other two aimed at the possessions we invent: short
spells of the other team, 2 to 8 seconds long, where under the challenge's rule nobody gained
control. A second model that judged each candidate possession as a whole (its length, how sure
the frames were, who was near the ball) separated invented from real ones no better than the
decoder's own evidence did (+0.004). And network outputs that mark where possession changes,
play stops and play restarts, used as prices in the decoder, looked like +0.007 when tuned on
the same games they were scored on, but only +0.003 (within the noise) when each fold's setting
was chosen on the other folds. The second network in our submission is the one trained with
those outputs; we use only its "dead?" and "home?" answers.

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

We did take one thing from the provided match: summary statistics of its tracking, printed by
`scripts/profile_atd.py` and listed in `training/calibrate.py`. They are counts, percentages and
medians over the whole match, and we used them to set up the simulator described below.

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
any part in the decision.

The optional broadcast step could not be judged on held-out games, because the public games have
no broadcast feed. Its weights are measured on the match itself without labels, and we looked at
what it did on the public window as a check, with our first model: dead-ball recall rose from 0.68
to 0.86 with precision unchanged at 0.96, macroF1 rose from 0.85 to 0.90, and spellF1 stayed at
0.64. We do not submit it (see the next section).

## The rules clarification of 15 November

The organisers' clarification says that anything trained or tuned must use external data only,
never the ground truth or the provided video. Our model is trained only on the public games, the
rule we use to relabel them comes from the definition, and the decoder settings are chosen on
held-out public games. We have asked the organisers about
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
single frame can cost at most 200.

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

Last, we tried three changes to how the first network is built and trained, deciding before we
ran them that each had to gain at least 0.005 by more than twice its standard error. Six
simulated copies of each game instead of three, twice the training steps with a learning rate
that falls to zero, and a little dropout gained only +0.005, in six of ten games, so we did not
take it. The same recipe with a network twice as wide (128 channels) gained +0.0073 in the
average, +0.0081 when each fold's prices were chosen on the other folds, and its prices came out
the same as before; on its own this network beats the narrow one by +0.012, in nine of ten games.
The submission uses it in place of the first network. Training the second network the same way
did not add anything (-0.0003), so it stays as it was.

PyTorch is only used for training. A floating-point network can add its sums in a different order
on another machine and change a last digit, so for the submission the weights are stored as whole
numbers (`models/sequence_wide.npz`, `models/sequence2.npz`) and run in integer arithmetic by
`possession/seqnet.py`: every product and sum is exact. For the first network that version gives
the same labels as the floating-point one on 99.9997% of 4.3 million held-out frames; with both
networks the held-out scores are the same to four decimals. A test checks it against PyTorch.

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
live pictures the two trackings put the players a median of 0.28 m apart. (The numbers in this
section were measured with our first submission's model.) That gives every
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

The ball belongs to the team whose player last had it under control. A touch, a deflection or a
duel without control does not change that, and a loose ball or a ball in flight stays with the
team that had it: that is how we rewrote the training labels, so it is what the model learned.
The decoder also refuses a change of team that the evidence does not hold up for long enough.

Under this rule very short possessions are rare (2-9% of the relabelled ones last under 2
seconds), and the model finds only about a third of them. On held-out games it
predicts about 6% fewer possessions than the truth has.

The ball is dead from the moment play stops until it is struck again. The model judges this
mainly from the players: they slow down, and someone carries the ball to a spot. Three stretches
are dead by definition: the second half before its kick-off, the time between a goal and the
kick-off that follows, and the seconds before a set piece is taken.

We never write X on a graded frame. On spellF1 it would only repeat our previous label, and on
macroF1 it counts as a mistake.

## How well it works

We tested by leaving games out: a game is never labelled by a model that was trained on it. The
scores come from the official `evaluation/possession_metrics.py`, against the controlled-possession
labels. The trees alone, our first submission's model and the organisers' baseline run on the
same simulated tracking for comparison; on the real match the baseline scores 0.2433.

| held-out games | simulator setting | our spellF1 | trees alone | first model | baseline | our macroF1 | baseline macroF1 |
|---|---|---|---|---|---|---|---|
| DFL (7) | typical | 0.766 | 0.731 | 0.609 | 0.295 | 0.929 | 0.700 |
| DFL (7) | more gaps during play | 0.746 | 0.700 | 0.594 | 0.256 | 0.921 | 0.671 |
| DFL (7) | ball tracked more when dead | 0.769 | 0.732 | 0.604 | 0.281 | 0.931 | 0.623 |
| Metrica (3) | typical | 0.786 | 0.763 | 0.713 | 0.321 | 0.908 | 0.717 |
| Metrica (3) | more gaps during play | 0.783 | 0.752 | 0.716 | 0.290 | 0.898 | 0.681 |
| Metrica (3) | ball tracked more when dead | 0.803 | 0.784 | 0.724 | 0.337 | 0.903 | 0.665 |

```
spellF1 on games the model never saw (typical simulator setting)

Metrica   ours         #######################################  0.78
          first model  ###################################      0.71
          baseline     ################                         0.32
DFL       ours         ######################################   0.77
          first model  ##############################           0.61
          baseline     ###############                          0.30
                       |         |         |         |
                       0        0.2       0.4       0.6
```

So on games it has never seen, the model scores more than twice what the baseline does, and it
beat our first model in every one of the ten games. Please read this as an estimate. The tracking
in these tests is simulated, and the labels are our reading of the definition.

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
- Our simulator puts the ball near the touchlines about twice as often as the real tracker does.
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
tag was shared.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt          # numpy and pandas, Python 3.11
# unpack the data bundle into data/ (or set MTAH_DATA_DIR); data/tactical/ is all it reads
.venv/bin/python generate_submission.py                 # writes "outputs/Blaugrana Bytes.csv"
```

The command prints the SHA-256 of the file it wrote. Ours is
`2afa3d8e43cb4c176a92b3612e7e2b9c035dfbc7e3ad8f2b8167a1cc650102af`, and we got the same value
from a fresh clone of the repository. It takes about a minute and a half. Our earlier files can
be made from the same repository: `--sequence models/sequence.npz models/sequence2.npz` (the
narrow first network) writes the version before the wide network
(`5b6400a13e1bf218354d8f874089dc32ed5f875a53e92939d8eacaa7f8246205`), `--sequence
models/sequence.npz` (one network) our third
submission (`dfa6bf3e193cb200849debb83d36cbfe4ac6e9569f1952517d267dce0877d785`), `--no-sequence`
(trees only) our second (`0674c775c4b6758a8dd2c89a38f7b3c20dd584fe6772b2992ee1a8324f3afa24`),
and `--no-sequence --model-file models/possession_v1.npz` our first
(`a320484ed426d4d71c70ce02406d19b65965604b55c524b26a830f49f2c29f5d`). `--broadcast` (which also
needs `data/broadcast/`) adds the optional broadcast step.

The trained models are in the repository (`models/possession.npz`, `models/sequence_wide.npz`,
`models/sequence.npz`,
`models/sequence2.npz`). To build them again from the public games:

```bash
.venv/bin/pip install -r requirements-train.txt          # scikit-learn and PyTorch
.venv/bin/python training/fetch.py --only metrica         # 190 MB of public data
.venv/bin/python training/fetch.py --only dfl             # 2.6 GB
.venv/bin/python training/train.py --out models/possession_v1.npz        # labels as given
.venv/bin/python training/train.py --labels controlled \
    --work-dir .cache/train_controlled --out .cache/models/possession_controlled.npz
.venv/bin/python training/combine.py                    # writes models/possession.npz
.venv/bin/python training/sequence.py --export models/sequence.npz
.venv/bin/python training/sequence.py --change --with .cache/train_seq_int --name sequence2 \
    --work-dir .cache/train_seq_change --out .cache/models/sequence2.pt --export models/sequence2.npz
.venv/bin/python training/sequence.py --copies 6 --train-steps 3000 --cosine --dropout 0.1 \
    --width 128 --with .cache/train_seq_change_int --name seqB --work-dir .cache/train_seqB \
    --out .cache/models/sequence_wide.pt --export models/sequence_wide.npz
```

Each tree training run takes about 30 minutes, each narrow network about 20 and the wide one
about 50. A retrained model can
differ slightly from ours, because scikit-learn and PyTorch add up some sums on several threads
or on the graphics processor. The submitted file comes from the models in the repository.

`.venv/bin/python -m pytest tests -q` runs our checks. Among other things they confirm that our
reader and baseline give exactly what `starter/baseline.py` gives, that running the trees in
numpy gives exactly what scikit-learn gives, that the whole-number network matches PyTorch, that
swapping the teams swaps home and away exactly, that the relabelling treats touches, holds and
passes as described above, that the whole pipeline still gives the same output on a made-up
match, and that the broadcast line-up recovers a known delay, cut and mirrored axes on made-up
recordings.
