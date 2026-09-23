import pytest

pytest.importorskip("numpy")

from python.audit_duel_pair_targets import PairTarget, summarize_pairs


def test_pair_prevalence_separates_static_and_response_effects() -> None:
    pairs = [
        PairTarget(1, 0, -20, 10, 30, False),
        PairTarget(1, 1, -30, -40, -10, False),
        PairTarget(2, 0, 0, 0, 0, True),
    ]

    report = summarize_pairs(pairs)

    assert report["pairs"] == 3
    assert report["reply_better"] == 1
    assert report["reply_worse"] == 1
    assert report["reply_same"] == 1
    assert report["static_ties"] == 1
    assert report["positive_response_swings"] == 1
    assert report["negative_response_swings"] == 1
    assert report["terminal_magnitude_pairs"] == 1
    assert report["ordinary_positive_static_gap_median"] == 20
    assert report["ordinary_positive_response_swing_median"] == 30


def test_empty_pair_prevalence_has_no_medians() -> None:
    report = summarize_pairs([])

    assert report["pairs"] == 0
    assert report["ordinary_positive_static_gap_median"] is None
