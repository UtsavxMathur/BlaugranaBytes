"""The error breakdown (possession/errors.py) on tiny label sequences worked out by hand."""

import numpy as np

from possession import errors

H, A, D = 0, 1, 2


def seq(*parts):
    """seq((H, 100), (D, 50)) -> codes array."""
    return np.concatenate([np.full(n, code, dtype=np.int8) for code, n in parts])


def test_a_perfect_prediction_is_all_good_with_zero_edge_errors():
    truth = seq((H, 100), (D, 50), (A, 100))
    tax = errors.taxonomy(truth, truth.copy(), [(0, 250)])
    assert (tax["true"], tax["good"], tax["missed"], tax["invented"]) == (2, 2, 0, 0)
    edge = errors.edges(truth, truth.copy(), [(0, 250)])
    assert edge["start after a restart"] == [0.0, 0.0]     # the half's start, and after D
    assert edge["end at a stoppage"] == [0.0, 0.0]          # into D, and the half's end


def test_a_missed_stoppage_stretches_the_possession_before_it():
    truth = seq((H, 100), (D, 50), (A, 100))
    pred = seq((H, 150), (A, 100))
    tax = errors.taxonomy(truth, pred, [(0, 250)])
    assert (tax["good"], tax["edges"]) == (1, 1)            # H: IoU 100/150
    assert errors.edges(truth, pred, [(0, 250)])["end at a stoppage"][0] == 2.0


def test_a_short_possession_given_to_the_other_team_merges_its_neighbours():
    truth = seq((H, 100), (A, 10), (H, 100))
    pred = seq((H, 210))
    tax = errors.taxonomy(truth, pred, [(0, 210)])
    assert tax["missed"] == 1 and tax["missed_by"]["other team"] == 1
    assert tax["merged"] == 1 and tax["edges"] == 2


def test_an_invented_possession_inside_the_other_teams_splits_it():
    truth = seq((H, 200))
    pred = seq((H, 90), (A, 20), (H, 90))
    tax = errors.taxonomy(truth, pred, [(0, 200)])
    assert tax["invented"] == 1 and tax["invented_by"]["in the other team's"] == 1
    assert tax["split"] == 1 and tax["edges"] == 1


def test_reasons_for_missing_called_dead_and_swallowed():
    truth = seq((H, 100), (A, 50), (H, 100))
    tax = errors.taxonomy(truth, seq((H, 100), (D, 50), (H, 100)), [(0, 250)])
    assert tax["missed_by"]["called dead"] == 1 and tax["good"] == 2

    truth = seq((H, 30), (D, 10), (H, 160))
    tax = errors.taxonomy(truth, seq((H, 200)), [(0, 200)])
    assert tax["missed_by"]["inside a longer one"] == 1 and tax["good"] == 1


def test_halves_are_scored_separately():
    truth = seq((H, 100), (H, 100))
    tax = errors.taxonomy(truth, truth.copy(), [(0, 100), (100, 200)])
    assert tax["true"] == 2 and tax["good"] == 2


def test_merging_counts_from_several_games():
    truth = seq((H, 100), (D, 50), (A, 100))
    one = errors.taxonomy(truth, truth.copy(), [(0, 250)])
    total = errors.merge_taxonomies([one, one])
    assert total["true"] == 4 and total["by_length"]["true"] == [2 * n for n in one["by_length"]["true"]]
    edges = errors.merge_edges([errors.edges(truth, truth.copy(), [(0, 250)])] * 2)
    assert len(edges["start after a restart"]) == 4
