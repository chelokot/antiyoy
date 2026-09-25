from pathlib import Path

import pytest

from python.audit_duel_markov_root_fidelity import (
    empty_counts,
    empty_shadow_counts,
    legal_count_bin,
    record_counts,
    record_shadow_counts,
    run,
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


def test_root_fidelity_rejects_unexpected_checkpoint_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not a model")

    with pytest.raises(ValueError, match="predeclared hashes"):
        run(checkpoint, checkpoint, checkpoint, "wrong-student-hash")


def test_shadow_disagreement_distinguishes_missed_and_extra_queries() -> None:
    counts = empty_shadow_counts()

    record_shadow_counts(counts, source=0, student=1, shadow=1)
    record_shadow_counts(counts, source=0, student=1, shadow=0)
    record_shadow_counts(counts, source=0, student=0, shadow=2)
    record_shadow_counts(counts, source=0, student=1, shadow=2)

    assert counts == {
        "positions": 4,
        "student_disagreements": 3,
        "shadow_disagreements": 3,
        "same_action": 1,
        "same_trigger": 2,
        "missed_student_disagreements": 1,
        "extra_shadow_disagreements": 1,
    }
