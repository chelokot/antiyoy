import numpy as np
import torch

from python import audit_three_turn_shared_stop_outcomes
from python.audit_three_turn_shared_stop_outcomes import finish_persistently
from python.audit_three_turn_shared_stop_stability import (
    preference,
    summarize_stability,
)


def terminal(winner: int) -> dict[str, object]:
    return {
        "terminal": True,
        "truncated": False,
        "winner": winner,
        "adjudicated_winner": None,
        "actions_after_intervention": 10,
    }


def censored() -> dict[str, object]:
    return {
        "terminal": False,
        "truncated": True,
        "winner": None,
        "adjudicated_winner": 0,
        "actions_after_intervention": 2400,
    }


def sample(
    seat: int,
    direct: tuple[dict[str, object], dict[str, object]],
    two_turn: tuple[dict[str, object], dict[str, object]],
) -> dict[str, object]:
    return {
        "root_seat": seat,
        "outcomes": {
            "direct": {"teacher": direct[0], "static": direct[1]},
            "two_turn": {"teacher": two_turn[0], "static": two_turn[1]},
        },
    }


def test_preference_uses_root_perspective_and_censors_action_limit() -> None:
    assert preference({"teacher": terminal(1), "static": terminal(0)}, 1) == 1
    assert preference({"teacher": terminal(1), "static": terminal(0)}, 0) == -1
    assert preference({"teacher": censored(), "static": terminal(0)}, 0) is None


def test_summary_separates_reversal_from_one_policy_only_change() -> None:
    samples = [
        sample(0, (terminal(0), terminal(1)), (terminal(1), terminal(0))),
        sample(0, (terminal(0), terminal(1)), (terminal(0), terminal(0))),
        sample(1, (terminal(1), terminal(1)), (terminal(1), terminal(0))),
        sample(1, (terminal(1), terminal(1)), (terminal(0), terminal(0))),
        sample(0, (terminal(0), terminal(1)), (terminal(0), terminal(1))),
        sample(0, (censored(), terminal(1)), (terminal(0), terminal(1))),
    ]
    summary = summarize_stability(samples)
    cross = summary["cross_policy"]
    assert cross == {
        "strict_agreement": 1,
        "strict_reversal": 1,
        "direct_only_strict": 1,
        "two_turn_only_strict": 1,
        "both_tied": 1,
        "censored_in_either_mode": 1,
    }
    assert summary["by_opponent_policy"]["direct"]["censored"] == 1
    assert summary["by_opponent_policy"]["two_turn"]["censored"] == 0


def test_two_turn_mode_uses_native_opponent_search(monkeypatch) -> None:
    selected = []

    class Branch:
        finished = False

        def done(self):
            return np.asarray([self.finished])

        def observe(self):
            return {"active_players": np.asarray([1])}

        def step(self, action):
            selected.append(int(action[0]))
            self.finished = True
            return {
                "terminal": np.asarray([True]),
                "truncated": np.asarray([False]),
                "winners": np.asarray([0]),
            }

    class DirectOpponent:
        def actions(self, observation, rules):
            raise AssertionError("direct opponent must not act in two-turn mode")

    def native_action(environment, followup_nodes=0):
        assert followup_nodes == 0
        return np.asarray([7], dtype=np.uint64)

    monkeypatch.setattr(
        audit_three_turn_shared_stop_outcomes, "native_teacher_action", native_action
    )
    result = finish_persistently(
        Branch(),
        {},
        2,
        0,
        DirectOpponent(),
        torch.empty(0),
        "two_turn",
    )
    assert selected == [7]
    assert result["winner"] == 0
    assert result["actions_after_intervention"] == 3
