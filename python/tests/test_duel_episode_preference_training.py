import copy

import numpy as np
import torch

from antiyoy_rl.model import UniversalPolicy
from antiyoy_rl.routed import RoutedPolicy
from python.audit_duel_first_regret import create_environment
from python import train_duel_episode_preference as preference


def test_terminal_pairs_keep_both_preference_directions() -> None:
    dataset = {
        "first_seed": 6561000,
        "maps": 1,
        "records": [
            {"seed": 6561000, "teacher_seat": None, "outcome": {"terminal": True, "winner": 0}},
            {"seed": 6561000, "teacher_seat": 0, "outcome": {"terminal": True, "winner": 1}},
            {"seed": 6561000, "teacher_seat": 1, "outcome": {"terminal": True, "winner": 1}},
        ],
    }

    pairs = preference.informative_pairs(dataset)

    assert len(pairs[6561000]) == 2
    assert pairs[6561000][0].preferred["teacher_seat"] is None
    assert pairs[6561000][1].preferred["teacher_seat"] == 1


def test_streamed_complete_pair_gradient_matches_full_autograd(monkeypatch) -> None:
    torch.manual_seed(7)
    environment = create_environment(7)
    observation = {
        key: np.asarray(value).copy() for key, value in environment.observe().items()
    }
    assert np.diff(observation["action_offsets"])[0] > 1
    rules_json = environment.rules_jsons()[0]
    preferred = preference.DecisionTrace(
        (observation, observation), rules_json, (0, 0)
    )
    dispreferred = preference.DecisionTrace(
        (observation, observation), rules_json, (1, 1)
    )
    policy = RoutedPolicy(
        {"shared": UniversalPolicy(hidden=16, layers=1)}, ("shared", "shared")
    )
    reference = copy.deepcopy(policy)
    full = copy.deepcopy(policy)
    monkeypatch.setattr(preference, "CHUNK_SIZE", 1)

    loss, margin = preference.accumulate_pair_gradient(
        policy, reference, preferred, dispreferred, 1
    )
    full_margin = (
        preference.decision_log_probabilities(full, preferred, 0, 2).sum()
        - preference.decision_log_probabilities(full, dispreferred, 0, 2).sum()
    )
    with torch.no_grad():
        reference_margin = (
            preference.decision_log_probabilities(reference, preferred, 0, 2).sum()
            - preference.decision_log_probabilities(reference, dispreferred, 0, 2).sum()
        )
    full_margin = full_margin - reference_margin
    full_loss = preference.preference_loss(full_margin)
    full_loss.backward()

    assert abs(margin - float(full_margin.detach())) < 1e-5
    assert abs(loss - float(full_loss.detach())) < 1e-5
    for streamed, direct in zip(
        policy.models["shared"].parameters(), full.models["shared"].parameters(), strict=True
    ):
        if direct.grad is None:
            assert streamed.grad is None
        else:
            assert streamed.grad is not None
            torch.testing.assert_close(streamed.grad, direct.grad, atol=1e-5, rtol=1e-4)
