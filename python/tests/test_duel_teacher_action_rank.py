import torch

from python.audit_duel_teacher_action_rank import coverage, teacher_rank


def test_teacher_rank_uses_stable_legal_order_on_ties() -> None:
    logits = torch.tensor([1.0, 2.0, 2.0, -1.0])

    assert [teacher_rank(logits, action) for action in range(4)] == [3, 1, 2, 4]


def test_candidate_coverage_separates_source_disagreements() -> None:
    records = [
        {"rank": 1, "root_seat": 0},
        {"rank": 2, "root_seat": 1},
        {"rank": 8, "root_seat": 0},
        {"rank": 9, "root_seat": 1},
    ]

    result = coverage(records)

    assert result["positions"] == 4
    assert result["source_teacher_disagreements"] == 3
    assert result["teacher_in_source_top_k"] == {
        "1": 1,
        "2": 2,
        "4": 2,
        "8": 3,
        "16": 4,
        "32": 4,
    }
    assert result["teacher_in_source_top_k_on_disagreements"] == {
        "1": 0,
        "2": 1,
        "4": 1,
        "8": 2,
        "16": 3,
        "32": 3,
    }
