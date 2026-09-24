from python.audit_three_turn_shared_stop_outcomes import (
    shared_prefix_length,
    summarize,
)


def terminal(winner: int) -> dict[str, object]:
    return {
        "terminal": True,
        "truncated": False,
        "winner": winner,
        "adjudicated_winner": None,
        "actions_after_intervention": 20,
    }


def truncated() -> dict[str, object]:
    return {
        "terminal": False,
        "truncated": True,
        "winner": None,
        "adjudicated_winner": 0,
        "actions_after_intervention": 2400,
    }


def test_shared_prefix_requires_an_earlier_stop_after_identical_actions() -> None:
    assert shared_prefix_length([[4, 7, 9, 0], [4, 7, 0]], 1) == 2
    assert shared_prefix_length([[4, 0], [4, 0]], 1) is None
    assert shared_prefix_length([[4, 7, 0], [4, 9, 0]], 1) is None


def test_summary_censors_adjudication_and_scores_terminal_pairs_by_root() -> None:
    samples = [
        {
            "seed": 1,
            "root_seat": 0,
            "outcomes": {"teacher": terminal(0), "static": terminal(1)},
        },
        {
            "seed": 2,
            "root_seat": 1,
            "outcomes": {"teacher": terminal(0), "static": terminal(1)},
        },
        {
            "seed": 3,
            "root_seat": 0,
            "outcomes": {"teacher": truncated(), "static": terminal(0)},
        },
    ]
    result = summarize(samples)
    assert result["independent_maps"] == 3
    assert result["teacher_better"] == 1
    assert result["static_better"] == 1
    assert result["censored"] == 1
    assert result["by_root_seat"][0]["censored"] == 1
