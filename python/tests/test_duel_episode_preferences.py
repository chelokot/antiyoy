from python.collect_duel_episode_preferences import MAPS, SEED_FIRST, summarize


def record(seed: int, teacher_seat: int | None, winner: int, terminal: bool = True):
    return {
        "seed": seed,
        "teacher_seat": teacher_seat,
        "outcome": {"terminal": terminal, "winner": winner if terminal else None},
    }


def test_episode_preference_pairs_both_seats_and_excludes_censoring() -> None:
    records = []
    for seed in range(SEED_FIRST, SEED_FIRST + MAPS):
        reference_winner = seed % 2
        records.extend(
            (
                record(seed, None, reference_winner),
                record(seed, 0, 0, terminal=seed != SEED_FIRST),
                record(seed, 1, 1),
            )
        )

    summary = summarize(records)

    assert summary["games"] == MAPS * 3
    assert summary["terminal_games"] == MAPS * 3 - 1
    assert summary["censored_pairs"] == 1
    assert summary["by_teacher_seat"][0] == {
        "teacher_preferred": 8,
        "source_preferred": 0,
        "same_terminal_outcome": 7,
        "censored": 1,
    }
    assert summary["by_teacher_seat"][1] == {
        "teacher_preferred": 8,
        "source_preferred": 0,
        "same_terminal_outcome": 8,
        "censored": 0,
    }
    assert not summary["data_gate_passed"]
