from __future__ import annotations

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.browser_policy import BrowserPolicy
from antiyoy_rl.model import UniversalPolicy, encode_rules
from export_browser_policy import browser_inputs, select_browser_policy_state


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
    browser = BrowserPolicy(policy, 11, 9).eval()
    for _ in range(4):
        observation = environment.observe()
        inputs = browser_inputs(observation, rules)
        with torch.inference_mode():
            expected_logits, expected_value = policy(observation, rules)
            actual_logits, actual_value = browser(*inputs)
        torch.testing.assert_close(actual_logits, expected_logits)
        torch.testing.assert_close(actual_value, expected_value)
        environment.step(np.array([int(torch.argmax(actual_logits))], dtype=np.uint64))


def test_browser_policy_selects_exact_seat_route() -> None:
    primary = UniversalPolicy(hidden=16, layers=1)
    second_seat = UniversalPolicy(hidden=16, layers=1)
    primary.missing_source.data.fill_(1.0)
    second_seat.missing_source.data.fill_(9.0)
    checkpoint = {
        "kind": "routed_policy_bundle",
        "bundle_version": 4,
        "config": {
            "hidden": 16,
            "layers": 1,
            "profiles": ["classic_generic_2022"],
        },
        "experts": {
            "primary": primary.state_dict(),
            "seat-one": second_seat.state_dict(),
        },
        "routes": {"classic_generic_2022": "primary"},
        "context_routes": [],
        "seat_context_routes": [
            {
                "profile": "classic_generic_2022",
                "generator": "symmetric_duel_v1",
                "players": 2,
                "seat": 1,
                "expert": "seat-one",
            }
        ],
        "domain_routes": [],
    }

    state, config = select_browser_policy_state(
        checkpoint, "classic_generic_2022", seat=1
    )

    assert config["selected_expert"] == "seat-one"
    assert torch.all(state["missing_source"] == 9.0)
