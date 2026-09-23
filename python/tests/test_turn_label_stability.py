import numpy as np
import pytest

from antiyoy_rl.turn_credit import TurnCreditPosition
from python.audit_turn_label_stability import summarize


def position(
    seed: int,
    seat: int,
    greedy: list[int],
    opponent_search: list[int] | None,
    round_number: int = 4,
) -> TurnCreditPosition:
    return TurnCreditPosition(
        seed=seed,
        seat=seat,
        round=round_number,
        root={},
        post_turn={},
        root_rules_json=(),
        post_turn_rules_json=(),
        outcome_scores=np.asarray(greedy, dtype=np.int8),
        static_scores=np.zeros(len(greedy), dtype=np.int64),
        search_index=0,
        greedy_index=0,
        opponent_search_scores=(
            np.asarray(opponent_search, dtype=np.int8)
            if opponent_search is not None
            else None
        ),
        opponent_search_nodes=32,
    )


def test_audit_distinguishes_outcome_changes_from_preference_reversals() -> None:
    summary = summarize(
        [
            position(10, 0, [0, 2, -1], [2, 0, -1]),
            position(10, 1, [1, 2], [1, 2]),
            position(11, 2, [0, 0], [0, 2], round_number=14),
        ]
    )

    assert summary["independent_maps"] == 2
    assert summary["paired_state_outcomes"] == {
        "changed": 3,
        "censored": 1,
        "same": 3,
    }
    assert summary["candidate_vs_search"] == {
        "complete": 3,
        "informative": 3,
        "changed": 2,
        "strict_reversals": 1,
        "greedy_better_probe_not": 1,
        "probe_better_greedy_not": 1,
        "robust_better": 1,
        "same": 1,
        "censored": 1,
    }
    assert summary["maps_with_changed_state_outcome"] == 2
    assert summary["positions_with_changed_state_outcome"] == 2
    assert summary["positions_with_informative_candidate"] == 3
    assert summary["positions_with_changed_candidate_preference"] == 2
    assert summary["positions_with_strict_preference_reversal"] == 1
    assert summary["positions_with_robust_better_candidate"] == 1
    assert summary["maps_with_informative_candidate"] == 2
    assert summary["maps_with_changed_candidate_preference"] == 2
    assert summary["maps_with_strict_preference_reversal"] == 1
    assert summary["maps_with_robust_better_candidate"] == 1
    assert summary["maps_with_better_candidate"] == {
        "greedy_continuation": 1,
        "search_opponents": 2,
    }
    assert summary["by_seat"]["1"] == {"positions": 1}
    assert summary["by_round_decade"] == {
        "0-9": {
            "positions": 2,
            "informative_positions": 2,
            "changed_preference_positions": 1,
        },
        "10-19": {
            "positions": 1,
            "informative_positions": 1,
            "changed_preference_positions": 1,
        },
    }


def test_audit_requires_aligned_probe_labels() -> None:
    with pytest.raises(ValueError, match="align"):
        summarize([position(10, 0, [0, 2], None)])


def test_audit_round_bucket_includes_uninformative_positions() -> None:
    summary = summarize([position(12, 4, [0, 0], [0, 0], round_number=20)])

    assert summary["by_round_decade"] == {"20-29": {"positions": 1}}
    assert summary["positions_with_robust_better_candidate"] == 0
