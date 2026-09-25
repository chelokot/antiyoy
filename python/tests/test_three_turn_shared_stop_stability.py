import numpy as np
import torch

from python import audit_three_turn_shared_stop_outcomes
from python.audit_duel_three_turn_intervention import FOLLOWUP_NODES
from python.audit_three_turn_shared_stop_outcomes import finish_persistently
from python.audit_three_turn_shared_stop_stability import (
    CONFIGURATIONS,
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
    conditional = summary["conditional_nonharmful_signal"]
    assert conditional["beneficial"] == 3
    assert conditional["harmful"] == 1
    assert conditional["neutral"] == 1
    assert conditional["censored"] == 1


def test_confirmation_uses_a_disjoint_predeclared_map_window() -> None:
    exploratory = CONFIGURATIONS["exploratory"]
    confirmation = CONFIGURATIONS["confirmation"]
    assert exploratory.first_seed + exploratory.maps <= confirmation.first_seed
    assert confirmation.maximum_samples == 48
    assert confirmation.protocol.endswith(
        "duel-shared-stop-opponent-confirmation-v1.json"
    )


def test_persistent_modes_use_requested_native_policy(monkeypatch) -> None:
    selected = []
    native_calls = []

    class Branch:
        def __init__(self, active: int) -> None:
            self.active = active
            self.finished = False

        def done(self):
            return np.asarray([self.finished])

        def observe(self):
            return {"active_players": np.asarray([self.active])}

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

    def native_action(environment, followup_nodes=0, replan_each_action=False):
        native_calls.append((followup_nodes, replan_each_action))
        return np.asarray([7], dtype=np.uint64)

    monkeypatch.setattr(
        audit_three_turn_shared_stop_outcomes, "native_teacher_action", native_action
    )
    result = finish_persistently(
        Branch(1),
        {},
        2,
        0,
        DirectOpponent(),
        torch.empty(0),
        "two_turn",
    )
    replanned = finish_persistently(
        Branch(0),
        {},
        2,
        0,
        DirectOpponent(),
        torch.empty(0),
        root_replan_each_action=True,
    )
    assert selected == [7, 7]
    assert native_calls == [(0, False), (FOLLOWUP_NODES, True)]
    assert result["winner"] == 0
    assert result["actions_after_intervention"] == 3
    assert replanned["winner"] == 0
