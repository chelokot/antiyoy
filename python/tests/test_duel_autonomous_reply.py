from __future__ import annotations

import json

import numpy as np

from python.audit_duel_autonomous_reply import (
    compare_map_errors,
    selected_candidate,
    transformed_error,
)


def test_transformed_error_keeps_terminal_scores_finite() -> None:
    assert transformed_error(5, 5) == 0.0
    assert np.isfinite(transformed_error(1_000_000_000_000, -1_000_000_000_000))
    assert transformed_error(100, 200) == transformed_error(200, 100)


def test_selected_candidate_uses_native_static_and_rank_tie_breaks() -> None:
    assert selected_candidate([5, 5, 4], [1, 2, 100]) == 1
    assert selected_candidate([5, 5, 5], [2, 2, 2]) == 0


def test_map_error_comparison_serializes_numpy_comparisons() -> None:
    comparison, ledger = compare_map_errors(
        {3: [(1.0, 0.5)], 4: [(0.25, 0.5)], 5: [(1.0, 1.0)]}
    )

    assert comparison["candidate_better"] == 1
    assert comparison["baseline_better"] == 1
    assert comparison["same"] == 1
    assert ledger[0]["seed"] == 3
    json.dumps({"comparison": comparison, "ledger": ledger})
