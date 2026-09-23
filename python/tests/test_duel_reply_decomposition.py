import pytest

pytest.importorskip("torch")

from python.audit_duel_reply_decomposition import ReplyDecomposition, summarize


def test_decomposition_separates_static_gap_from_response_swing() -> None:
    rows = [
        ReplyDecomposition(1, 0, True, -20, -30, -10, 0.1, False),
        ReplyDecomposition(1, 1, False, -40, 15, 55, 0.2, False),
        ReplyDecomposition(2, 0, False, 0, 0, 0, 0.3, True),
    ]

    report = summarize(rows)

    assert report["positions"] == 3
    assert report["better"] == 1
    assert report["worse"] == 1
    assert report["same"] == 1
    assert report["root_static_score_ties"] == 1
    assert report["terminal_magnitude_positions"] == 1
    assert report["ordinary_root_static_gap_median"] == -30
    assert report["ordinary_reply_gap_median"] == -7.5
    assert report["ordinary_response_swing_median"] == 22.5
    assert report["ordinary_positive_response_swings"] == 1
    assert report["ordinary_negative_response_swings"] == 1


def test_empty_decomposition_has_explicit_missing_medians() -> None:
    report = summarize([])

    assert report["positions"] == 0
    assert report["ordinary_root_static_gap_median"] is None
    assert report["predicted_margin_median"] is None
