"""Gradient-boosted trees: taken out of scikit-learn once, then run in plain numpy.

Why not just call scikit-learn when making the submission? RULES.md wants the repository to
regenerate the *same file*. A tree only compares a feature with a threshold and adds up leaf
values in a fixed order, so running the trees ourselves gives bit-identical scores on any
machine, with numpy as the only dependency. scikit-learn is needed for training only.

export() is checked against the estimator's own decision_function at training time.
"""

import numpy as np

FIELDS = ("feature", "threshold", "left", "right", "value", "leaf", "missing_left", "starts",
          "baseline")


def export(estimator):
    """A fitted binary HistGradientBoostingClassifier as a dict of flat arrays."""
    parts = {name: [] for name in FIELDS[:7]}
    starts, offset = [], 0
    for stage in estimator._predictors:
        if len(stage) != 1:
            raise ValueError("expected a binary classifier (one tree per boosting step)")
        nodes = stage[0].nodes
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits are not supported")
        parts["feature"].append(nodes["feature_idx"].astype(np.int32))
        parts["threshold"].append(nodes["num_threshold"].astype(np.float64))
        parts["left"].append(nodes["left"].astype(np.int32) + offset)
        parts["right"].append(nodes["right"].astype(np.int32) + offset)
        parts["value"].append(nodes["value"].astype(np.float64))
        parts["leaf"].append(nodes["is_leaf"].astype(bool))
        parts["missing_left"].append(nodes["missing_go_to_left"].astype(bool))
        starts.append(offset)
        offset += len(nodes)
    model = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    model["starts"] = np.array(starts, dtype=np.int32)
    model["baseline"] = np.array(float(np.ravel(estimator._baseline_prediction)[0]))
    return model


def raw_score(model, X):
    """Log-odds of the positive class for every row of X (NaN = missing value)."""
    feature, threshold = model["feature"], model["threshold"]
    left, right, value = model["left"], model["right"], model["value"]
    leaf, missing_left = model["leaf"], model["missing_left"]
    n = len(X)
    score = np.full(n, float(model["baseline"]), dtype=np.float64)
    for start in model["starts"]:
        node = np.full(n, start, dtype=np.int64)
        active = np.flatnonzero(~leaf[node])
        while len(active):
            current = node[active]
            x = X[active, feature[current]]
            go_left = np.where(np.isnan(x), missing_left[current], x <= threshold[current])
            node[active] = np.where(go_left, left[current], right[current])
            active = active[~leaf[node[active]]]
        score += value[node]      # one tree at a time, in order: the sum is reproducible
    return score
