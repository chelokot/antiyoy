from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path

from antiyoy_rl import VectorEnv
from antiyoy_rl.browser_policy import BrowserPolicy
from antiyoy_rl.model import UniversalPolicy, encode_rules
from python.export_browser_policy import (
    browser_inputs,
    export_policy,
    select_browser_policy_state,
)
from python.tests.test_bundle import write_checkpoint


@pytest.mark.parametrize("action_residual_hidden", [0, 16])
def test_browser_policy_matches_universal_policy(action_residual_hidden: int) -> None:
    torch.manual_seed(73)
    environment = VectorEnv(
        profile="classic_generic_2022",
        environments=1,
        width=11,
        height=9,
        seed=47,
        action_limit=1_000,
    )
    policy = UniversalPolicy(
        hidden=16, layers=2, action_residual_hidden=action_residual_hidden
    ).eval()
    if policy.action_residual is not None:
        policy.action_residual.network[-1].weight.data.fill_(0.02)
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


def test_exported_residual_policy_matches_legal_actions(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from python.verify_browser_policy import verify_policy

    checkpoint_path = tmp_path / "residual.pt"
    model_path = tmp_path / "residual.onnx"
    write_checkpoint(checkpoint_path, 1.0, profiles=["classic_generic_2022"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    residual = UniversalPolicy(hidden=16, layers=1, action_residual_hidden=12)
    assert residual.action_residual is not None
    residual.action_residual.network[-1].weight.data.fill_(0.02)
    checkpoint["model"].update(
        {
            name: value
            for name, value in residual.state_dict().items()
            if name.startswith("action_residual.")
        }
    )
    torch.save(checkpoint, checkpoint_path)

    export_policy(
        checkpoint_path,
        model_path,
        "classic_generic_2022",
        11,
        9,
        47,
        0,
    )
    report = verify_policy(
        checkpoint_path,
        model_path,
        "classic_generic_2022",
        11,
        9,
        47,
        20,
        0,
    )

    assert report["actions_compared"] == 20
    assert report["maximum_logit_error"] < 1e-4
