import numpy as np
import pytest

from python.audit_duel_source_uncertainty import Position, confidence, route


def test_confidence_normalizes_for_legal_action_count() -> None:
    top, entropy, margin = confidence(np.asarray([0.5, 0.5]))
    assert top == 0.5
    assert entropy == pytest.approx(1.0)
    assert margin == 0.0
    assert confidence(np.asarray([1.0])) == (1.0, 0.0, 1.0)


def test_fixed_query_budget_captures_disagreements_without_relabeling() -> None:
    positions = [
        Position(10, 0, 20, True, 0.3, 0.9, 0.1),
        Position(10, 1, 20, True, 0.4, 0.7, 0.2),
        Position(11, 0, 2, False, 0.8, 0.2, 0.3),
        Position(11, 1, 2, False, 0.9, 0.1, 0.4),
    ]
    quarter = route(positions, 0.25, "normalized_entropy")
    half = route(positions, 0.5, "top_probability_margin")
    assert quarter["query_count"] == 1
    assert quarter["overall"]["captured_disagreements"] == 1
    assert quarter["overall"]["residual_disagreement_fraction"] == pytest.approx(1 / 3)
    assert half["query_count"] == 2
    assert half["overall"]["captured_disagreements"] == 2
    assert half["overall"]["residual_disagreement_fraction"] == 0
    assert half["by_seat"]["0"]["captured_disagreements"] == 1
