from __future__ import annotations

import json

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import UniversalPolicy, encode_rules_batch

from python.audit_duel_autonomous_reply import (
    candidate_score_record,
    compare_map_errors,
    searched_reply_diagnostic,
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


def test_candidate_score_record_preserves_paired_rankings_and_censoring() -> None:
    record = {
        "seat": 1,
        "round": 8,
        "selected_index": 2,
        "static_scores": [10, 20, 20],
        "reply_scores": [5, 7, 9],
    }

    exported = candidate_score_record(
        71, record, {"source": [2, 5, 5], "student": [3, None, 1]}
    )

    assert exported["autonomous_selected_indices"] == {"source": 1, "student": None}
    assert exported["native_reply_scores"] == [5, 7, 9]
    json.dumps(exported)


def test_map_error_comparison_serializes_numpy_comparisons() -> None:
    comparison, ledger = compare_map_errors(
        {3: [(1.0, 0.5)], 4: [(0.25, 0.5)], 5: [(1.0, 1.0)]}
    )

    assert comparison["candidate_better"] == 1
    assert comparison["baseline_better"] == 1
    assert comparison["same"] == 1
    assert ledger[0]["seed"] == 3
    json.dumps({"comparison": comparison, "ledger": ledger})


def test_native_teacher_reply_reconstruction_matches_its_terminal_score() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=47)
    reference = environment.fork(np.asarray([0], dtype=np.uint64))
    active = int(reference.observe()["active_players"][0])
    for _ in range(24):
        selected = reference.search_actions(node_budget=64)
        reference.step(selected)
        if (
            reference.done()[0]
            or int(reference.observe()["active_players"][0]) != active
        ):
            break
    score = int(reference.position_scores(1)[0])
    rules = encode_rules_batch([environment.rules_json()], torch.device("cpu"))
    policy = UniversalPolicy(hidden=16, layers=1).eval()

    diagnostic = searched_reply_diagnostic(
        environment, {"source": policy, "student": policy}, rules, 1, score
    )

    assert diagnostic["teacher_actions"] >= 1
    assert not diagnostic["first_actions_differ"]
    assert (
        diagnostic["first_mismatch"]["source"]
        == diagnostic["first_mismatch"]["student"]
    )
