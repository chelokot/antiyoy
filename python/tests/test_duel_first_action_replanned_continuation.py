from python.audit_duel_first_action_replanned_continuation import summarize


def branch(winner: int | None, truncated: bool = False) -> dict[str, object]:
    return {
        "terminal": not truncated,
        "truncated": truncated,
        "winner": winner,
        "adjudicated_winner": 0 if truncated else None,
        "actions_after_intervention": 10,
    }


def test_first_action_summary_uses_terminal_root_perspective() -> None:
    samples = [
        {"root_seat": 0, "outcomes": {"teacher": branch(0), "source": branch(1)}},
        {"root_seat": 1, "outcomes": {"teacher": branch(0), "source": branch(1)}},
        {"root_seat": 0, "outcomes": {"teacher": branch(0), "source": branch(0)}},
        {
            "root_seat": 1,
            "outcomes": {"teacher": branch(None, True), "source": branch(0)},
        },
    ]

    summary = summarize(samples)

    assert summary["teacher_better"] == 1
    assert summary["source_better"] == 1
    assert summary["same"] == 1
    assert summary["censored"] == 1
    assert summary["by_root_seat"]["0"] == {
        "teacher_better": 1,
        "source_better": 0,
        "same": 1,
        "censored": 0,
    }
    assert summary["by_root_seat"]["1"] == {
        "teacher_better": 0,
        "source_better": 1,
        "same": 0,
        "censored": 1,
    }
    assert not summary["exploratory_advance_condition"]


def test_first_action_advance_requires_complete_map_signal() -> None:
    samples = [
        {
            "root_seat": index % 2,
            "outcomes": {
                "teacher": branch(index % 2),
                "source": branch(1 - index % 2),
            },
        }
        for index in range(10)
    ]
    samples.extend(
        {
            "root_seat": index % 2,
            "outcomes": {"teacher": branch(0), "source": branch(0)},
        }
        for index in range(10, 32)
    )

    summary = summarize(samples)

    assert summary["teacher_better"] == 10
    assert summary["source_better"] == 0
    assert summary["exact_two_sided_map_sign_test_p"] < 0.05
    assert summary["exploratory_advance_condition"]
