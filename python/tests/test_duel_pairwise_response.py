import torch

from python.scout_duel_pairwise_response import (
    pair_features,
    qualifying_threshold,
    select_candidate,
)


def test_pair_representation_preserves_static_candidate_and_difference() -> None:
    baseline = torch.tensor([[2.0, 5.0], [2.0, 5.0]])
    alternatives = torch.tensor([[4.0, 1.0], [1.0, 7.0]])

    result = pair_features(baseline, alternatives)

    assert result.tolist() == [
        [2.0, 5.0, 4.0, 1.0, 2.0, -4.0],
        [2.0, 5.0, 1.0, 7.0, -1.0, 2.0],
    ]


def test_pair_selector_abstains_below_fixed_probability_threshold() -> None:
    logits = torch.tensor([-1.0, 0.4, 0.2])

    assert select_candidate(logits, 0.5) == 2
    assert select_candidate(logits, 0.7) == 0
    assert select_candidate(torch.empty(0), 0.5) == 0


def test_calibration_rejects_seat_regression_and_breaks_ties_conservatively() -> None:
    reports = [
        {
            "threshold": 0.5,
            "overrides": 30,
            "native_reply_vs_static": {
                "better": 15,
                "worse": 10,
                "by_seat": {
                    0: {"better": 12, "worse": 3},
                    1: {"better": 3, "worse": 7},
                },
            },
        },
        {
            "threshold": 0.7,
            "overrides": 25,
            "native_reply_vs_static": {
                "better": 12,
                "worse": 8,
                "by_seat": {
                    0: {"better": 7, "worse": 5},
                    1: {"better": 5, "worse": 3},
                },
            },
        },
        {
            "threshold": 0.9,
            "overrides": 21,
            "native_reply_vs_static": {
                "better": 10,
                "worse": 6,
                "by_seat": {
                    0: {"better": 6, "worse": 4},
                    1: {"better": 4, "worse": 2},
                },
            },
        },
    ]

    assert qualifying_threshold(reports) == 0.9
