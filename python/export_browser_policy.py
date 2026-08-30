from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.browser_policy import BrowserPolicy
from antiyoy_rl.model import UniversalPolicy, encode_rules, load_policy_state

try:
    from .evaluate import load_policy_checkpoint, select_policy_state
except ImportError:
    from evaluate import load_policy_checkpoint, select_policy_state


INPUT_NAMES = (
    "playable",
    "visible",
    "owners",
    "objects",
    "unit_strengths",
    "ready",
    "defenses",
    "province_features",
    "active_player",
    "player_count",
    "round_number",
    "action_sources",
    "action_targets",
    "action_kinds",
    "action_parameters",
)


def province_features(observation: dict[str, np.ndarray]) -> np.ndarray:
    province_ids = observation["province_ids"].astype(np.int64, copy=False)
    values = np.zeros((province_ids.size, 3), dtype=np.float32)
    present = province_ids != 65535
    selected = province_ids[present]
    numeric = np.stack(
        (
            observation["province_money"][selected],
            observation["province_profit"][selected],
            observation["province_sizes"][selected],
        ),
        axis=1,
    ).astype(np.float32)
    values[present] = np.sign(numeric) * np.log1p(np.abs(numeric))
    return values


def browser_inputs(observation: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    long_names = (
        "visible",
        "owners",
        "objects",
        "unit_strengths",
        "ready",
        "defenses",
    )
    inputs: list[torch.Tensor] = [
        torch.as_tensor(observation["playable"], dtype=torch.float32)
    ]
    inputs.extend(
        torch.as_tensor(observation[name], dtype=torch.long) for name in long_names
    )
    inputs.append(torch.from_numpy(province_features(observation)))
    inputs.extend(
        (
            torch.as_tensor(observation["active_players"], dtype=torch.long),
            torch.as_tensor(observation["player_counts"], dtype=torch.long),
            torch.as_tensor(observation["rounds"], dtype=torch.long),
            torch.as_tensor(observation["action_sources"], dtype=torch.long),
            torch.as_tensor(observation["action_targets"], dtype=torch.long),
            torch.as_tensor(observation["action_kinds"], dtype=torch.long),
            torch.as_tensor(observation["action_parameters"], dtype=torch.long),
        )
    )
    return tuple(inputs)


def export_policy(
    checkpoint_path: Path,
    output_path: Path,
    profile: str,
    width: int,
    height: int,
    seed: int,
) -> dict[str, object]:
    device = torch.device("cpu")
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    state, config = select_policy_state(
        checkpoint,
        profile=profile,
        generator="symmetric_duel",
        players=2,
        seat=1,
    )
    policy = UniversalPolicy(int(config["hidden"]), int(config["layers"]))
    load_policy_state(policy, state)
    policy.eval()
    environment = VectorEnv(
        profile=profile,
        environments=1,
        width=width,
        height=height,
        seed=seed,
        action_limit=1_000,
    )
    observation = environment.observe()
    if environment.rules_jsons()[0] != environment.rules_json():
        raise ValueError("single environment returned inconsistent rules")
    rules = json.loads(environment.rules_json())
    if rules["diplomacy"]["enabled"]:
        raise ValueError("browser policy export does not support diplomacy")
    wrapped = BrowserPolicy(
        policy,
        encode_rules(environment.rules_json(), device),
        width,
        height,
    ).eval()
    inputs = browser_inputs(observation)
    with torch.inference_mode():
        expected_logits, expected_value = policy(
            observation,
            encode_rules(environment.rules_json(), device),
        )
        actual_logits, actual_value = wrapped(*inputs)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_value, expected_value, rtol=1e-5, atol=1e-5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        inputs,
        output_path,
        input_names=INPUT_NAMES,
        output_names=("logits", "value"),
        dynamic_axes={
            "action_sources": {0: "actions"},
            "action_targets": {0: "actions"},
            "action_kinds": {0: "actions"},
            "action_parameters": {0: "actions"},
            "logits": {0: "actions"},
        },
        opset_version=18,
        dynamo=False,
    )
    return {
        "profile": profile,
        "width": width,
        "height": height,
        "seed": seed,
        "expert": config["selected_expert"],
        "actions": int(inputs[-1].shape[0]),
        "selected_action": int(torch.argmax(actual_logits).item()),
        "bytes": output_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", default="classic_generic_2022")
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--seed", type=int, default=47)
    arguments = parser.parse_args()
    summary = export_policy(
        arguments.checkpoint,
        arguments.output,
        arguments.profile,
        arguments.width,
        arguments.height,
        arguments.seed,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
