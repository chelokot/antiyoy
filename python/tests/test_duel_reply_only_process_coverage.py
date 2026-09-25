import pytest

from python.audit_duel_reply_only_process_coverage import stage_score, summarize


def candidate(static: int, reply: int, final: int, first_action: int) -> dict:
    def components(value: int) -> list[int]:
        return [value] + [0] * 15

    return {
        "plan": [first_action],
        "reply_plan": [0, 1],
        "followup_plan": [0],
        "post_components": components(static),
        "reply_components": components(reply),
        "followup_components": components(final),
        "terminal_classes": [1, 1, 1],
        "static_score": 128 * static,
        "response_score": 128 * final,
    }


def test_reply_stage_can_help_or_hurt_full_teacher_agreement() -> None:
    samples = [
        {
            "root_seat": 0,
            "rollin_censored": False,
            "teacher_selected_index": 1,
            "candidates": [candidate(2, 1, 1, 0), candidate(1, 3, 3, 1)],
        },
        {
            "root_seat": 1,
            "rollin_censored": False,
            "teacher_selected_index": 0,
            "candidates": [candidate(2, 1, 4, 0), candidate(1, 3, 3, 1)],
        },
    ]
    report = summarize(samples)
    assert report["candidate_indices"] == {
        "static_match": 1,
        "reply_match": 1,
        "better": 1,
        "worse": 1,
        "same": 0,
    }
    assert report["first_actions"] == report["candidate_indices"]
    assert report["candidates"] == 4
    assert report["by_root_seat"]["0"]["reply_action_match"] == 1
    assert report["by_root_seat"]["1"]["static_action_match"] == 1


def test_terminal_score_is_exact_and_inconsistent_teacher_rank_fails() -> None:
    root = candidate(2, 1, 1, 0)
    root["terminal_classes"][1] = 2
    assert stage_score(root, 1) == 10**12
    root["terminal_classes"][1] = 0
    assert stage_score(root, 1) == -(10**12)
    sample = {
        "root_seat": 0,
        "rollin_censored": False,
        "teacher_selected_index": 1,
        "candidates": [candidate(2, 1, 4, 0), candidate(1, 3, 3, 1)],
    }
    with pytest.raises(ValueError, match="teacher choice"):
        summarize([sample])
