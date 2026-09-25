import pytest
import torch

from python.audit_duel_action_feature_alias import identical_action_features, summarize


def test_identical_action_features_requires_complete_row_equality() -> None:
    features = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 3.0]])

    assert identical_action_features(features, 0) == [0, 1]
    assert identical_action_features(features, 2) == [2]
    with pytest.raises(ValueError, match="outside"):
        identical_action_features(features, 3)


def test_alias_summary_separates_teacher_disagreements() -> None:
    records = [
        {"rank": 1, "alias_count": 2, "source_top1_in_alias": True},
        {"rank": 2, "alias_count": 2, "source_top1_in_alias": True},
        {"rank": 3, "alias_count": 1, "source_top1_in_alias": False},
    ]

    assert summarize(records) == {
        "positions": 3,
        "teacher_aliased_positions": 2,
        "source_teacher_disagreements": 2,
        "teacher_aliased_on_disagreements": 1,
        "source_top1_aliases_teacher_on_disagreements": 1,
    }
