import numpy as np

from antiyoy_rl.model import UniversalPolicy
from antiyoy_rl.routed import RoutedPolicy
from python.audit_duel_first_regret import create_environment
from python.audit_duel_episode_preference_offline import (
    exact_two_sided_sign_p,
    margin_summary,
    score_trace,
)
from python.train_duel_episode_preference import DecisionTrace


def test_exact_map_sign_and_offline_gate() -> None:
    assert exact_two_sided_sign_p(5, 0) == 0.0625
    margins = {seed: [(seed % 2, 1.0)] for seed in range(24)}
    margins[24] = []

    result = margin_summary(margins, 24, 0.01, True)

    assert result["map_positive"] == 24
    assert result["map_zero"] == 1
    assert result["offline_gate_passed"]
    assert not margin_summary(margins, 24, 0.051, True)["offline_gate_passed"]
    assert not margin_summary(margins, 24, 0.01, False)["offline_gate_passed"]


def test_reference_scores_zero_episode_ratio_and_action_kl() -> None:
    environment = create_environment(7)
    observation = {
        key: np.asarray(value).copy() for key, value in environment.observe().items()
    }
    trace = DecisionTrace((observation,), environment.rules_jsons()[0], (0,))
    policy = RoutedPolicy(
        {"shared": UniversalPolicy(hidden=16, layers=1)}, ("shared", "shared")
    )

    ratio, divergence, decisions = score_trace(policy, policy, trace)

    assert ratio == 0.0
    assert divergence == 0.0
    assert decisions == 1
