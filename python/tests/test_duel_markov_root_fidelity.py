from python.audit_duel_markov_root_fidelity import (
    empty_counts,
    legal_count_bin,
    record_counts,
)


def test_legal_action_count_bins_cover_fixed_boundaries() -> None:
    assert [legal_count_bin(count) for count in (1, 2, 4, 5, 16, 17)] == [
        "1",
        "2-4",
        "2-4",
        "5-16",
        "5-16",
        "17+",
    ]


def test_root_fidelity_counts_exact_correction_and_off_target_departures() -> None:
    counts = empty_counts()

    record_counts(counts, source=1, student=2, teacher=2)
    record_counts(counts, source=1, student=3, teacher=2)
    record_counts(counts, source=2, student=3, teacher=2)
    record_counts(counts, source=2, student=2, teacher=2)

    assert counts["decisions"] == 4
    assert counts["source_matches_teacher"] == 2
    assert counts["student_matches_teacher"] == 2
    assert counts["teacher_source_disagreements"] == 2
    assert counts["student_matches_teacher_on_disagreements"] == 1
    assert counts["student_matches_neither_on_disagreements"] == 1
    assert counts["student_deviates_when_teacher_matches_source"] == 1
    assert counts["student_matches_source"] == 1
