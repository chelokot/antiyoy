import pytest

pytest.importorskip("torch")

from python.audit_reply_slate import summarize


def branch(score: int, state_index: int, action: object) -> dict[str, object]:
    return {
        "static_score": score,
        "state_index": state_index,
        "actions": [action, "EndTurn"],
    }


def continuation(winner: int, reply_score: int | None) -> dict[str, object]:
    return {
        "winner": winner,
        "reply_score": reply_score,
        "truncated": False,
    }


def record(
    seed: int,
    seat: int,
    branches: list[dict[str, object]],
    continuations: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "seed": seed,
        "seat": seat,
        "search": branches[0],
        "beam_candidates": [
            {"rank": rank, "branch": candidate}
            for rank, candidate in enumerate(branches[1:], start=1)
        ],
        "opponent_search_continuations": continuations,
    }


def test_reply_slate_audit_distinguishes_score_from_outcome_preference() -> None:
    report = summarize(
        [
            record(
                1,
                0,
                [
                    branch(300, 0, {"Move": {"target": 1}}),
                    branch(200, 1, {"Recruit": {"target": 2}}),
                    branch(100, 2, "EndTurn"),
                ],
                [
                    continuation(1, 100),
                    continuation(0, 200),
                    continuation(1, 50),
                ],
            ),
            record(
                2,
                1,
                [
                    branch(300, 0, {"Move": {"target": 1}}),
                    branch(200, 1, {"Recruit": {"target": 2}}),
                ],
                [continuation(1, 100), continuation(0, 200)],
            ),
        ]
    )

    assert report["positions"] == 2
    assert report["candidate_states"] == 5
    assert report["positions_with_distinct_first_actions"] == 2
    assert report["positions_with_outcome_diversity"] == 2
    assert report["reply_choice_differs_from_static"] == 2
    assert report["reply_choice_changes_first_action"] == 2
    assert report["selected_outcome_better"] == 1
    assert report["selected_outcome_worse"] == 1
    assert report["pairwise_concordant"] == 2
    assert report["pairwise_discordant"] == 1
    assert report["selected_rank_counts"] == {"1": 2}
    assert report["independent_maps_selected_better"] == 1
    assert report["independent_maps_selected_worse"] == 1
    assert report["independent_map_sign_test_p"] == 1.0


def test_reply_slate_audit_does_not_infer_missing_reply_scores() -> None:
    report = summarize(
        [
            record(
                3,
                0,
                [branch(300, 0, "EndTurn"), branch(200, 1, {"Move": 1})],
                [continuation(1, None), continuation(0, 200)],
            )
        ]
    )

    assert report["positions_missing_reply_score"] == 1
    assert report["selected_outcome_better"] == 0
    assert report["selected_rank_counts"] == {}
    assert report["independent_maps_selected_better"] == 0
    assert report["independent_maps_censored"] == 1
