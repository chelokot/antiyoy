from __future__ import annotations

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.browser_policy import BrowserPolicy
from antiyoy_rl.model import UniversalPolicy, encode_rules
from export_browser_policy import browser_inputs


def test_browser_policy_matches_universal_policy() -> None:
    torch.manual_seed(73)
    environment = VectorEnv(
        profile="classic_generic_2022",
        environments=1,
        width=11,
        height=9,
        seed=47,
        action_limit=1_000,
    )
    policy = UniversalPolicy(hidden=16, layers=2).eval()
    rules = encode_rules(environment.rules_json(), torch.device("cpu"))
    browser = BrowserPolicy(policy, rules, 11, 9).eval()
    for _ in range(4):
        observation = environment.observe()
        inputs = browser_inputs(observation)
        with torch.inference_mode():
            expected_logits, expected_value = policy(observation, rules)
            actual_logits, actual_value = browser(*inputs)
        torch.testing.assert_close(actual_logits, expected_logits)
        torch.testing.assert_close(actual_value, expected_value)
        environment.step(np.array([int(torch.argmax(actual_logits))], dtype=np.uint64))
