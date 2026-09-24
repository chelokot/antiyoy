import numpy as np
from pathlib import Path

from python.evaluate import evaluate, selective_reply_search_actions
from python.tests.test_bundle import write_checkpoint


class SearchEnvironment:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def reply_search_actions(self, **arguments: object) -> np.ndarray:
        self.calls.append(arguments)
        return np.asarray([9, 8, 7, 6], dtype=np.uint64)


def select(
    environment: SearchEnvironment,
    student: list[int],
    source: list[int],
    active: list[bool],
) -> tuple[np.ndarray, int]:
    return selective_reply_search_actions(
        environment,
        np.asarray(student, dtype=np.uint64),
        np.asarray(source, dtype=np.uint64),
        np.asarray(active, dtype=np.bool_),
        256,
        64,
        8,
        32,
        48,
        24,
        32,
    )


def test_selective_search_queries_only_active_disagreements() -> None:
    environment = SearchEnvironment()

    actions, queries = select(
        environment,
        student=[1, 2, 3, 4],
        source=[1, 5, 6, 7],
        active=[True, True, False, True],
    )

    assert queries == 2
    assert actions.tolist() == [1, 8, 6, 6]
    assert len(environment.calls) == 1
    assert environment.calls[0]["active_mask"].tolist() == [0, 1, 0, 1]
    assert environment.calls[0]["replan_each_action"] is True
    assert environment.calls[0]["followup_nodes"] == 32


def test_selective_search_skips_teacher_when_policies_agree() -> None:
    environment = SearchEnvironment()

    actions, queries = select(
        environment,
        student=[1, 2, 3, 4],
        source=[1, 2, 6, 4],
        active=[True, True, False, True],
    )

    assert queries == 0
    assert actions.tolist() == [1, 2, 6, 4]
    assert environment.calls == []


def test_selective_search_runs_against_full_teacher_with_separate_source(
    tmp_path: Path,
) -> None:
    student = tmp_path / "student.pt"
    source = tmp_path / "source.pt"
    write_checkpoint(student, 1.0)
    write_checkpoint(source, 2.0)

    result = evaluate(
        student,
        games=2,
        seed=96_000,
        device_name="cpu",
        baseline="reply_search",
        profile="classic_generic_2022",
        search_nodes=16,
        search_beam_width=8,
        search_branch_width=12,
        search_maximum_actions_per_turn=8,
        reply_search_nodes=8,
        reply_slate_size=4,
        followup_search_nodes=8,
        replan_reply_search=True,
        width=7,
        height=5,
        action_limit=40,
        procedural=True,
        generator_schema_version=2,
        players=2,
        model_agent="selective_reply_search",
        selective_source_checkpoint_path=source,
    )

    assert result["baseline"] == "reply_search"
    assert result["selective_source_checkpoint"] == str(source)
    assert result["selective_reply_search"]["candidate_decisions"] > 0
    assert (
        result["selective_reply_search"]["queries"]
        <= result["selective_reply_search"]["candidate_decisions"]
    )
    assert len(result["winners"]) == 2

    agreement = evaluate(
        student,
        games=2,
        seed=96_000,
        device_name="cpu",
        baseline="reply_search",
        profile="classic_generic_2022",
        search_nodes=16,
        search_beam_width=8,
        search_branch_width=12,
        search_maximum_actions_per_turn=8,
        reply_search_nodes=8,
        reply_slate_size=4,
        followup_search_nodes=8,
        replan_reply_search=True,
        width=7,
        height=5,
        action_limit=40,
        procedural=True,
        generator_schema_version=2,
        players=2,
        model_agent="selective_reply_search",
        selective_source_checkpoint_path=student,
    )

    assert agreement["selective_reply_search"]["queries"] == 0
    assert (
        agreement["selective_reply_search"]["final_actions_different_from_source"] == 0
    )
    assert agreement["baseline_self_play"] == result["baseline_self_play"]
