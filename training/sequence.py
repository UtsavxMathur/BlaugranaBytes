"""A sequence model for possession: it reads the match forwards and backwards.

    python training/sequence.py --export models/sequence.npz      # the first network
    python training/sequence.py --change --with .cache/train_seq_int --name sequence2 \
        --work-dir .cache/train_seq_change --out .cache/models/sequence2.pt \
        --export models/sequence2.npz                              # the second network
    python training/sequence.py --quick         # small and fast, to check that everything runs

Each run: cross-validation, decoder prices, the held-out predictions of the whole-number
networks, the blend with the trees (and, with --with, with the other network), its prices, and
the final network trained on all games, exported in whole numbers with the blend's prices.

The second network is trained the same way plus three event heads (--change: where a change of
team, a stoppage and a restart happen). We tried those heads as prices in the decoder, and on
held-out games they did not help; they are dropped on export. What the second network adds is a
second opinion on "dead?" and "home?": averaged with the first and the trees it gained +0.008
spellF1 on held-out games (9 of 10).

The tree models of train.py judge one frame at a time, from numbers that summarise windows
around it. This model reads the whole sequence of those per-frame numbers with a temporal
convolutional network: stacked 1-D convolutions whose taps spread further apart layer by layer
(dilations 1, 2, 4 ... 64), so each output sees about 25 seconds before and after. It learns from
the same simulated public games, the same controlled-possession labels and the same weights as
the trees ("dead?" weighted by the labels as given), and writes held-out predictions in the same
format, so training/benchmark.py compares the two game by game.

It runs at 5 frames per second (every fifth frame); its scores are spread back to 25 fps for the
decoder. Home and away are treated exactly alike, as with the trees: each game is also read with
the teams swapped, and the two readings are combined.

Needs PyTorch (requirements-train.txt). Not used by generate_submission.py, which runs the
exported whole-number networks with possession/seqnet.py.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import baseline, features, geometry, model as model_module  # noqa: E402
from possession.submission import pm  # noqa: E402
from training import dataset, simulate, train  # noqa: E402

H, A, D, X = 0, 1, 2, 3
STRIDE = 5               # 25 fps -> 5 fps
WIDTH = 64               # channels inside the network
DILATIONS = (1, 2, 4, 8, 16, 32, 64)
CONTEXT = 1 + 2 * sum(DILATIONS)       # steps one output depends on: 255 = 51 s at 5 fps
WINDOW = 1024            # steps per training window, plus CONTEXT // 2 of context either side
BATCH = 16
STEPS = 1500             # gradient steps per model
QUICK_STEPS = 60
LEARNING_RATE = 1e-3
# A side task (--aux): also predict where the true ball is, which the simulator knows even when
# it hides the ball. Distances from the true ball to the nearest home and away player (log-scaled
# up to AUX_CLIP_M), and whether it is in the air (DFL has the ball's height). Live play only.
AUX = False
AUX_WEIGHT = 0.5
AUX_CLIP_M = 30.0
AIR_M = 1.5
# Event heads (--change, phase 5): where a change of team, a stoppage and a restart happen, for
# the decoder. Target: the step of each such event in the training labels, +-EVENT_TOL steps.
CHANGE = False
EVENTS = ("change", "stop", "start")
EVENT_TOL = 1
EVENT_POS_WEIGHT = 5.0
WORK_DIR = os.path.join(ROOT, ".cache", "train_seq")
OUT = os.path.join(ROOT, ".cache", "models", "sequence.pt")


# -- data ------------------------------------------------------------------------------------------

def _copy(game, copy):
    """The damaged tracking, labels and source labels of one training copy, as train.py makes it."""
    rng = np.random.default_rng(train.seed_of(game.name, "regime", copy))
    regime = simulate.Regime.random(rng)
    tracking, labels = game.tracking, game.labels
    source = game.source_labels if game.source_labels is not None else labels
    if copy % 2 == 1:
        tracking, labels, source = (features.swap_teams(tracking), train.swap_labels(labels),
                                    train.swap_labels(source))
    damaged = simulate.degrade(tracking, labels, regime, train.seed_of(game.name, "damage", copy))
    return damaged, labels, source, tracking


def ball_targets(tracking):
    """(n, 3) float32 about the TRUE ball of a clean tracking: scaled distance to the nearest home
    player and to the nearest away player, and 1 / 0 for a ball in the air. NaN where unknown."""
    ball, players = geometry.to_metres(tracking)
    near = geometry.team_distances(ball, players, tracking.team, k=1)[:, :, 0].astype(np.float64)
    near[~np.isfinite(near)] = np.nan
    scaled = np.log1p(np.minimum(near, AUX_CLIP_M)) / np.log1p(AUX_CLIP_M)
    air = np.full(len(near), np.nan)
    if tracking.ball_z is not None:
        z = np.asarray(tracking.ball_z, dtype=np.float64)
        air = np.where(np.isnan(z), np.nan, (z > AIR_M).astype(np.float64))
    return np.column_stack([scaled, air]).astype(np.float32)


def event_targets(labels, a, b):
    """(steps, 3) float32 at STRIDE for the half [a, b): 1 within EVENT_TOL steps of a change of
    team, a stoppage or a restart (the first frame of the new label), else 0; NaN where the labels
    are unknown."""
    part = labels[a:b]
    n_steps = len(range(0, b - a, STRIDE))
    out = np.zeros((n_steps, 3), dtype=np.float32)
    prev, cur = part[:-1], part[1:]
    live_prev, live_cur = (prev == H) | (prev == A), (cur == H) | (cur == A)
    kinds = (live_prev & live_cur & (prev != cur), live_prev & (cur == D), (prev == D) & live_cur)
    for k, mask in enumerate(kinds):
        for f in np.flatnonzero(mask) + 1:
            s = int(round(f / float(STRIDE)))
            out[max(0, s - EVENT_TOL):min(n_steps, s + EVENT_TOL + 1), k] = 1.0
    out[part[::STRIDE] == X_CODE] = np.nan
    return out


def sequences(game, copy, spell_weight):
    """One (features, labels, team weights, dead weights, ball targets, event targets) tuple per
    half, at STRIDE."""
    damaged, labels, source, clean = _copy(game, copy)
    X, _ = features.compute(damaged)
    weights = train.spell_weights(labels, damaged.fps, spell_weight, train.WEIGHT_CAP)
    dead_weights = train.spell_weights(source, damaged.fps, spell_weight, train.WEIGHT_CAP)
    targets = ball_targets(clean)
    # Live play only: in a stoppage the simulator draws its own ball path (DFL parks the ball
    # where play stopped, Metrica has none), so the clean tracking's ball is not the true one.
    targets[(labels != H) & (labels != A)] = np.nan
    return [(X[a:b:STRIDE], labels[a:b:STRIDE], weights[a:b:STRIDE], dead_weights[a:b:STRIDE],
             targets[a:b:STRIDE], event_targets(labels, a, b)) for a, b in damaged.periods]


class Scaler:
    """Standardise each feature with the training mean and spread; a missing value becomes 0
    and a second channel says it was missing."""

    def __init__(self, rows):
        import warnings
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # a feature that is always NaN
            self.mean = np.nan_to_num(np.nanmean(rows, axis=0))
            self.std = np.nan_to_num(np.nanstd(rows, axis=0), nan=1.0)
        self.std[self.std < 1e-6] = 1.0

    def __call__(self, X):
        missing = np.isnan(X)
        z = np.clip((np.where(missing, self.mean, X) - self.mean) / self.std, -10, 10)
        return np.concatenate([z, missing], axis=1).astype(np.float32)

    def state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


# -- the network -----------------------------------------------------------------------------------

def network(n_in, n_out=2):
    import torch
    from torch import nn

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(n_in, WIDTH, 1)
            self.taps = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, 3, dilation=d, padding=d)
                                      for d in DILATIONS)
            self.mix = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, 1) for _ in DILATIONS)
            self.out = nn.Conv1d(WIDTH, n_out, 1)      # dead?, home? (log-odds) [, ball side task]

        def forward(self, x):                          # (batch, channels, steps)
            h = torch.relu(self.inp(x))
            for tap, mix in zip(self.taps, self.mix):
                h = h + mix(torch.relu(tap(h)))
            return self.out(torch.relu(h))

    return TCN()


def n_outputs():
    """dead?, home?, then the ball side task (AUX), then the event heads (CHANGE)."""
    return 2 + (3 if AUX else 0) + (3 if CHANGE else 0)


def event_rows():
    first = 2 + (3 if AUX else 0)
    return slice(first, first + 3)


def _device():
    import torch
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def fit(parts, steps, seed=0):
    """Train one network on [(features, labels, weights, dead weights[, ball targets[, event
    targets]])] sequences. With AUX the ball targets are learned as a side task; with CHANGE the
    event heads are trained too."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaler = Scaler(np.concatenate([p[0] for p in parts]))
    half = CONTEXT // 2
    data = []
    for part in parts:
        X, labels, w, wd = part[:4]
        aux = part[4] if len(part) > 4 else np.full((len(labels), 3), np.nan, dtype=np.float32)
        events = part[5] if len(part) > 5 else np.full((len(labels), 3), np.nan, dtype=np.float32)
        pad = lambda a, value: np.concatenate(  # noqa: E731
            [np.full((half,) + a.shape[1:], value, a.dtype), a, np.full((half,) + a.shape[1:], value, a.dtype)])
        data.append((pad(scaler(X), 0), pad(labels, X_CODE), pad(w, 0).astype(np.float32),
                     pad(wd, 0).astype(np.float32), pad(aux.astype(np.float32), np.nan),
                     pad(events.astype(np.float32), np.nan)))
    lengths = np.array([len(d[1]) - 2 * half for d in data])
    pick = lengths / lengths.sum()

    device = _device()
    net = network(data[0][0].shape[1], n_outputs()).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    span = WINDOW + 2 * half
    for step in range(steps):
        xs, ys, ws, wds, auxs, evs = [], [], [], [], [], []
        for k in rng.choice(len(data), size=BATCH, p=pick):
            x, y, w, wd, aux, ev = data[k]
            start = int(rng.integers(0, max(1, len(y) - span)))
            for out_list, a, fill in ((xs, x, 0), (ys, y, X_CODE), (ws, w, 0), (wds, wd, 0),
                                      (auxs, aux, np.nan), (evs, ev, np.nan)):
                part = a[start:start + span]
                if len(part) < span:            # a half shorter than the window: unknown after it
                    part = np.concatenate([part, np.full((span - len(part),) + a.shape[1:],
                                                         fill, a.dtype)])
                out_list.append(part)
        x = torch.from_numpy(np.stack(xs)).transpose(1, 2).to(device)
        y = torch.from_numpy(np.stack(ys).astype(np.int64))[:, half:-half].to(device)
        w = torch.from_numpy(np.stack(ws))[:, half:-half].to(device)
        wd = torch.from_numpy(np.stack(wds))[:, half:-half].to(device)
        out = net(x)[:, :, half:-half]
        known = (y != X_CODE).float()
        live = ((y == H) | (y == A)).float()
        dead_loss = (loss_fn(out[:, 0], (y == D).float()) * wd * known).sum() / (wd * known).sum().clamp(min=1)
        home_loss = (loss_fn(out[:, 1], (y == H).float()) * w * live).sum() / (w * live).sum().clamp(min=1)
        loss = dead_loss + home_loss
        aux_loss = torch.zeros((), device=device)
        if AUX:
            target = torch.from_numpy(np.stack(auxs))[:, half:-half].transpose(1, 2).to(device)
            seen = ~torch.isnan(target)
            clean = torch.nan_to_num(target)
            near = torch.nn.functional.smooth_l1_loss(out[:, 2:4], clean[:, 0:2], reduction="none")
            air = loss_fn(out[:, 4], clean[:, 2])
            aux_loss = ((near * seen[:, 0:2]).sum() / seen[:, 0:2].sum().clamp(min=1)
                        + (air * seen[:, 2]).sum() / seen[:, 2].sum().clamp(min=1))
            loss = loss + AUX_WEIGHT * aux_loss
        event_loss = torch.zeros((), device=device)
        if CHANGE:
            target = torch.from_numpy(np.stack(evs))[:, half:-half].transpose(1, 2).to(device)
            seen = (~torch.isnan(target)).float()
            clean = torch.nan_to_num(target)
            weight = (1.0 + (EVENT_POS_WEIGHT - 1.0) * clean) * seen
            event_loss = ((loss_fn(out[:, event_rows()], clean) * weight).sum()
                          / weight.sum().clamp(min=1))
            loss = loss + event_loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 250 == 0 or step == steps - 1:
            print("    step {:>5}  loss dead {:.4f}  home {:.4f}{}{}".format(
                step, dead_loss.item(), home_loss.item(),
                "  ball {:.4f}".format(aux_loss.item()) if AUX else "",
                "  events {:.4f}".format(event_loss.item()) if CHANGE else ""), flush=True)
    return net.cpu().eval(), scaler


X_CODE = X


def scores(net, scaler, X):
    """(steps, 2) log-odds of dead and home for one half at STRIDE."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(scaler(X)).T[None]
        return net(x)[0, :2].T.numpy().astype(np.float64)


def predict_probabilities(net, scaler, tracking, plain=None, swapped=None):
    """(n, 3) probabilities of H, A, D at 25 fps for one tracking (NaN outside the halves).
    plain / swapped: its features, and those with the teams swapped, if already computed."""
    if plain is None:
        plain, _ = features.compute(tracking)
    if swapped is None:
        swapped, _ = features.compute(features.swap_teams(tracking))
    out = np.full((tracking.n_frames, 3), np.nan)
    for a, b in tracking.periods:
        s1, s2 = scores(net, scaler, plain[a:b:STRIDE]), scores(net, scaler, swapped[a:b:STRIDE])
        steps = np.arange(a, b, STRIDE) - a
        frames = np.arange(b - a)
        raw = [np.interp(frames, steps, column) for column in
               (s1[:, 0], s2[:, 0], s1[:, 1], s2[:, 1])]
        out[a:b] = model_module.combine(*raw)
    return out


def predict_events(net, scaler, tracking, plain, swapped):
    """(n, 3) probabilities, at 25 fps, of a change of team, a stoppage and a restart within
    EVENT_TOL steps of each frame, from the event heads. A change is a change whichever team is
    home, so the plain and teams-swapped readings are averaged."""
    import torch
    rows = event_rows()
    out = np.full((tracking.n_frames, 3), np.nan)
    for a, b in tracking.periods:
        with torch.no_grad():
            s1 = net(torch.from_numpy(scaler(plain[a:b:STRIDE])).T[None])[0, rows].T.numpy()
            s2 = net(torch.from_numpy(scaler(swapped[a:b:STRIDE])).T[None])[0, rows].T.numpy()
        z = 0.5 * (s1.astype(np.float64) + s2)
        steps = np.arange(a, b, STRIDE) - a
        frames = np.arange(b - a)
        for k in range(3):
            out[a:b, k] = 1.0 / (1.0 + np.exp(-np.interp(frames, steps, z[:, k])))
    return out


# -- steps -----------------------------------------------------------------------------------------

def build(games, quick, spell_weight):
    out = {}
    for name in games:
        game = dataset.load(name, verbose=False, labels=train.LABELS)
        started = time.time()
        out[name] = [part for copy in range(1 if quick else train.COPIES)
                     for part in sequences(game, copy, spell_weight)]
        print("  sequences {:<12} {:>7} steps ({:.0f} s)".format(
            name, sum(len(p[1]) for p in out[name]), time.time() - started), flush=True)
    return out


def cross_validate(parts, quick):
    import torch
    os.makedirs(WORK_DIR, exist_ok=True)
    for fold, held_out in enumerate(train.FOLDS):
        started = time.time()
        net, scaler = fit([p for name in parts if name not in held_out for p in parts[name]],
                          QUICK_STEPS if quick else STEPS, seed=fold)
        torch.save({"state": net.state_dict(), "scaler": scaler.state()},
                   os.path.join(WORK_DIR, "fold_{}.pt".format(fold)))
        for name in held_out:
            game = dataset.load(name, verbose=False, labels=train.LABELS)
            for regime_name in simulate.VALIDATION:
                damaged = simulate.degrade(game.tracking, game.labels,
                                           simulate.VALIDATION[regime_name],
                                           train.seed_of(game.name, regime_name))
                plain, _ = features.compute(damaged)
                swapped, _ = features.compute(features.swap_teams(damaged))
                prob = predict_probabilities(net, scaler, damaged, plain, swapped)
                extra = {}
                if CHANGE:
                    extra["events"] = predict_events(net, scaler, damaged, plain,
                                                     swapped).astype(np.float32)
                np.savez(os.path.join(WORK_DIR, "oof_{}_{}.npz".format(name, regime_name)),
                         prob=prob.astype(np.float32),
                         baseline=pm.label_codes(baseline.predict(damaged)), **extra)
        print("  fold {} (held out {}) done in {:.0f} s".format(
            fold, ", ".join(held_out), time.time() - started), flush=True)


def integer_network(path):
    """possession.seqnet.Network from a saved PyTorch network (fold_k.pt or the final .pt)."""
    import torch
    from possession import seqnet
    saved = torch.load(path, map_location="cpu")
    return seqnet.Network(seqnet.from_torch(saved["state"], saved["scaler"]["mean"],
                                            saved["scaler"]["std"], STRIDE, DILATIONS))


def integer_held_out(out_dir):
    """Held-out predictions of the whole-number networks of every fold, in the oof format."""
    os.makedirs(out_dir, exist_ok=True)
    for fold, held_out in enumerate(train.FOLDS):
        net = integer_network(os.path.join(WORK_DIR, "fold_{}.pt".format(fold)))
        for name in held_out:
            game = dataset.load(name, verbose=False, labels=train.LABELS)
            for regime_name in simulate.VALIDATION:
                damaged = simulate.degrade(game.tracking, game.labels,
                                           simulate.VALIDATION[regime_name],
                                           train.seed_of(game.name, regime_name))
                plain, _ = features.compute(damaged)
                swapped, _ = features.compute(features.swap_teams(damaged))
                raw = net.raw(plain, swapped, damaged.periods, damaged.n_frames)
                prob = model_module.combine(*raw)
                prob[~damaged.in_periods()] = np.nan
                np.savez(os.path.join(out_dir, "oof_{}_{}.npz".format(name, regime_name)),
                         prob=prob.astype(np.float32),
                         baseline=pm.label_codes(baseline.predict(damaged)))
        print("  fold {}: whole-number network done".format(fold), flush=True)


def _log_odds(prob):
    p_dead = np.clip(prob[:, 2], 1e-6, 1 - 1e-6)
    live = prob[:, 0] + prob[:, 1]
    with np.errstate(all="ignore"):
        p_home = np.where(live > 0, prob[:, 0] / live, 0.5)     # surely dead: no team information
    p_home = np.clip(p_home, 1e-6, 1 - 1e-6)
    return np.log(p_dead / (1 - p_dead)), np.log(p_home / (1 - p_home))


def blend_held_out(tree_dir, net_dir, out_dir, games, others=()):
    """Average the log-odds of held-out predictions, head by head: the trees' with this
    network's, or (others: more networks' held-out predictions) the trees' for one half and the
    networks' for the other half, in equal shares (added up in the order trees, others, this
    network), as possession/model.py does for the submission."""
    os.makedirs(out_dir, exist_ok=True)
    for name in games:
        for regime_name in simulate.VALIDATION:
            file = "oof_{}_{}.npz".format(name, regime_name)
            with np.load(os.path.join(tree_dir, file)) as z:
                tree, base = z["prob"].astype(np.float64), z["baseline"]
            with np.load(os.path.join(net_dir, file)) as z:
                net = z["prob"].astype(np.float64)
            outside = np.isnan(tree[:, 0])
            (d1, h1), (d2, h2) = _log_odds(np.nan_to_num(tree)), _log_odds(np.nan_to_num(net))
            if not others:
                p_dead = 1 / (1 + np.exp(-0.5 * (d1 + d2)))
                p_home = 1 / (1 + np.exp(-0.5 * (h1 + h2)))
            else:
                n_nets = len(others) + 1
                total = float(2 * n_nets)
                dead, home = n_nets / total * d1, n_nets / total * h1
                for folder in others:
                    with np.load(os.path.join(folder, file)) as z:
                        d, h = _log_odds(np.nan_to_num(z["prob"].astype(np.float64)))
                    dead, home = dead + 1 / total * d, home + 1 / total * h
                dead, home = dead + 1 / total * d2, home + 1 / total * h2
                p_dead, p_home = 1 / (1 + np.exp(-dead)), 1 / (1 + np.exp(-home))
            prob = np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], axis=1)
            prob[outside] = np.nan
            np.savez(os.path.join(out_dir, file), prob=prob.astype(np.float32), baseline=base)


def tune_and_bench(work_dir, sheet_name, compare_with):
    """Decoder prices chosen on the held-out predictions in work_dir, then the benchmark."""
    from training import benchmark
    games = [name for fold in train.FOLDS for name in fold]
    train.WORK_DIR = work_dir
    grid, sources, regimes, score, oof, decoded = train.step_tune(games)
    best = int(score.mean(axis=(1, 2)).argmax())
    prices = dict(zip(("cap", "flip", "stop", "start"), (int(v) for v in grid[best])))
    with open(os.path.join(work_dir, "prices.json"), "w") as handle:
        json.dump(prices, handle)
    sheet = benchmark.run(work_dir, prices)
    sheet.update(name=sheet_name, prices=prices, labels=train.LABELS,
                 work_dir=os.path.relpath(work_dir, ROOT))
    benchmark.report(sheet)
    with open(os.path.join(benchmark.SHEETS, sheet_name + ".json"), "w") as handle:
        json.dump(sheet, handle, indent=1, sort_keys=True, default=lambda value: value.item())
    reference = os.path.join(benchmark.SHEETS, compare_with + ".json")
    if os.path.isfile(reference):
        with open(reference) as handle:
            benchmark.compare(sheet, json.load(handle))
    return prices


def main():
    global WORK_DIR, OUT, AUX, CHANGE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", nargs="+",
                        default=["cv", "tune", "bench", "int", "blend", "final"],
                        choices=["cv", "tune", "bench", "int", "blend", "final"])
    parser.add_argument("--compare", default="hybrid", help="score sheet to compare with")
    parser.add_argument("--name", default="sequence", help="name of the score sheets written")
    parser.add_argument("--aux", action="store_true",
                        help="also learn where the true ball is, as a side task")
    parser.add_argument("--change", action="store_true",
                        help="also train event heads: change of team, stoppage, restart")
    parser.add_argument("--trees", default=os.path.join(ROOT, ".cache", "train_hybrid"),
                        help="held-out predictions of the tree model to blend with")
    parser.add_argument("--with", dest="others", nargs="+", default=[], metavar="DIR",
                        help="whole-number held-out predictions of other networks to average in "
                             "(the submission's second network: --with .cache/train_seq_int)")
    parser.add_argument("--export", default=os.path.join(ROOT, ".cache", "models", "sequence.npz"),
                        help="where the final network goes, in whole numbers, with blend prices")
    parser.add_argument("--prices", default=None,
                        help="decoder prices (JSON) stored with the exported network (default: "
                             "the blend's, <work dir>_blend/prices.json)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--work-dir", default=WORK_DIR)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    WORK_DIR, OUT, AUX, CHANGE = args.work_dir, args.out, args.aux, args.change
    train.LABELS, train.FOLDS, train.WORK_DIR = "controlled", train.BASE_FOLDS, WORK_DIR
    games = [name for fold in train.FOLDS for name in fold]

    parts = None
    if "cv" in args.steps or "final" in args.steps:
        print("building sequences ...")
        parts = build(games, args.quick, train.SPELL_WEIGHT_S)
    if "cv" in args.steps:
        print("cross-validation (device {}) ...".format(_device()))
        cross_validate(parts, args.quick)
    if "tune" in args.steps:
        print("choosing decoder prices on held-out predictions ...")
        grid, sources, regimes, score, oof, decoded = train.step_tune(games)
        best = int(score.mean(axis=(1, 2)).argmax())
        train.report(grid, sources, regimes, score, oof, decoded, best)
        prices = dict(zip(("cap", "flip", "stop", "start"), (int(v) for v in grid[best])))
        with open(os.path.join(WORK_DIR, "prices.json"), "w") as handle:
            json.dump(prices, handle)
    if "bench" in args.steps:
        from training import benchmark
        with open(os.path.join(WORK_DIR, "prices.json")) as handle:
            prices = json.load(handle)
        sheet = benchmark.run(WORK_DIR, prices)
        sheet.update(name=args.name, prices=prices, labels=train.LABELS,
                     work_dir=os.path.relpath(WORK_DIR, ROOT))
        benchmark.report(sheet)
        os.makedirs(benchmark.SHEETS, exist_ok=True)
        with open(os.path.join(benchmark.SHEETS, args.name + ".json"), "w") as handle:
            json.dump(sheet, handle, indent=1, sort_keys=True, default=lambda value: value.item())
        reference = os.path.join(benchmark.SHEETS, args.compare + ".json")
        if os.path.isfile(reference):
            with open(reference) as handle:
                benchmark.compare(sheet, json.load(handle))
    if "int" in args.steps:
        print("held-out predictions of the whole-number networks ...")
        integer_held_out(WORK_DIR + "_int")
    if "blend" in args.steps:
        print("blending with the trees and choosing prices on held-out games ...")
        blend_held_out(args.trees, WORK_DIR + "_int", WORK_DIR + "_blend", games, args.others)
        tune_and_bench(WORK_DIR + "_blend", args.name + "_blend", args.compare)
    if "final" in args.steps:
        import torch
        from possession import seqnet
        print("training the final network on all games ...")
        net, scaler = fit([p for name in parts for p in parts[name]],
                          QUICK_STEPS if args.quick else STEPS, seed=99)
        with open(os.path.join(WORK_DIR, "prices.json")) as handle:
            prices = json.load(handle)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        torch.save({"state": net.state_dict(), "scaler": scaler.state(), "prices": prices,
                    "config": {"stride": STRIDE, "width": WIDTH, "dilations": list(DILATIONS),
                               "aux": AUX, "change": CHANGE}},
                   OUT)
        print("wrote {}".format(os.path.relpath(OUT, ROOT)))
        blend_prices = args.prices or os.path.join(WORK_DIR + "_blend", "prices.json")
        if os.path.isfile(blend_prices):
            with open(blend_prices) as handle:
                prices = json.load(handle)
            arrays = seqnet.from_torch(net.state_dict(), scaler.mean, scaler.std, STRIDE, DILATIONS)
            arrays["prices"] = np.array(json.dumps(prices, sort_keys=True))
            arrays["names"] = np.array(json.dumps(
                model_module.Model.load(model_module.DEFAULT_PATH).names))   # the trees' 112
            np.savez_compressed(args.export, **arrays)
            print("wrote {} (whole numbers, blend prices {})".format(
                os.path.relpath(args.export, ROOT), prices))
    return 0


if __name__ == "__main__":
    sys.exit(main())
