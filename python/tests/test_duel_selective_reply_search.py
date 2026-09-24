import numpy as np

from python.evaluate import selective_reply_search_actions


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
