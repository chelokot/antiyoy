from __future__ import annotations

import pytest

from python.audit_duel_hybrid_on_policy import audit_on_policy


def record(
    seed: int,
    seat: int,
    native: list[int],
    predicted: list[int],
    teacher: int,
    selected: int,
) -> dict[str, object]:
    return {
        "seed": seed,
        "root_seat": seat,
        "round": 8,
        "native_reply_scores": native,
        "autonomous_reply_scores": predicted,
        "static_scores": native,
        "native_selected_index": teacher,
        "autonomous_selected_index": selected,
    }


def test_on_policy_audit_groups_both_seats_by_independent_map() -> None:
    result = audit_on_policy(
        {
            "model_agent": "model_reply_search",
            "seed": 100,
            "winners": [0, 1, 0, 1],
            "seats": [],
            "model_reply_native_audit": [
                record(100, 0, [10, 0], [10, 0], 0, 0),
                record(100, 1, [10, 0], [0, 10], 0, 1),
                record(101, 0, [0, 10], [10, 0], 1, 0),
                record(101, 1, [0, 10], [0, 10], 1, 1),
            ],
        }
    )

    assert result["games"] == 4
    assert result["sampled_candidate_states"] == 8
    assert result["ranking"]["positions"] == 4
    assert result["paired_map_seat_mean_transformed_regret"] == {
        "paired_maps": 2,
        "seat_one_higher": 1,
        "seat_zero_higher": 1,
        "same": 0,
        "exact_two_sided_sign_test_p": 1.0,
    }


def test_on_policy_audit_rejects_unrelated_agent() -> None:
    with pytest.raises(ValueError, match="hybrid"):
        audit_on_policy({"model_agent": "policy"})


def test_on_policy_audit_rejects_inconsistent_hybrid_selection() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        audit_on_policy(
            {
                "model_agent": "model_reply_search",
                "model_reply_native_audit": [
                    record(100, 0, [10, 0], [10, 0], 0, 1)
                ],
            }
        )
