from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import UniversalPolicy, encode_rules, load_policy_state
from export_browser_policy import INPUT_NAMES, browser_inputs

try:
    from .evaluate import load_policy_checkpoint, select_policy_state
except ImportError:
    from evaluate import load_policy_checkpoint, select_policy_state


def verify_policy(
    checkpoint_path: Path,
    model_path: Path,
    profile: str,
    width: int,
    height: int,
    seed: int,
    maximum_actions: int,
    seat: int,
) -> dict[str, int | float | str]:
    device = torch.device("cpu")
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    state, config = select_policy_state(
        checkpoint,
        profile=profile,
        generator="symmetric_duel",
        players=2,
        seat=seat,
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
    rules = encode_rules(environment.rules_json(), device)
    session = onnxruntime.InferenceSession(
        model_path,
        providers=("CPUExecutionProvider",),
    )
    compared = 0
    maximum_logit_error = 0.0
    maximum_value_error = 0.0
    smallest_action_count = 2**63 - 1
    largest_action_count = 0
    while compared < maximum_actions:
        observation = environment.observe()
        inputs = browser_inputs(observation, rules)
        feeds = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(INPUT_NAMES, inputs, strict=True)
        }
        with torch.inference_mode():
            expected_logits, expected_value = policy(observation, rules)
        actual_logits, actual_value = session.run(None, feeds)
        expected_numpy = expected_logits.detach().cpu().numpy()
        logit_error = float(np.max(np.abs(expected_numpy - actual_logits)))
        value_error = float(
            np.max(np.abs(expected_value.detach().cpu().numpy() - actual_value))
        )
        maximum_logit_error = max(maximum_logit_error, logit_error)
        maximum_value_error = max(maximum_value_error, value_error)
        expected_action = int(np.argmax(expected_numpy))
        actual_action = int(np.argmax(actual_logits))
        if actual_action != expected_action:
            raise AssertionError(
                f"action mismatch at step {compared}: "
                f"PyTorch={expected_action}, ONNX={actual_action}"
            )
        action_count = actual_logits.size
        smallest_action_count = min(smallest_action_count, action_count)
        largest_action_count = max(largest_action_count, action_count)
        result = environment.step(np.array([actual_action], dtype=np.uint64))
        compared += 1
        if bool(result["terminal"][0]) or bool(result["truncated"][0]):
            break
    return {
        "profile": profile,
        "seat": seat,
        "expert": str(config["selected_expert"]),
        "actions_compared": compared,
        "smallest_legal_action_set": smallest_action_count,
        "largest_legal_action_set": largest_action_count,
        "maximum_logit_error": maximum_logit_error,
        "maximum_value_error": maximum_value_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--profile", default="classic_generic_2022")
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--maximum-actions", type=int, default=1_000)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=1)
    arguments = parser.parse_args()
    summary = verify_policy(
        arguments.checkpoint,
        arguments.model,
        arguments.profile,
        arguments.width,
        arguments.height,
        arguments.seed,
        arguments.maximum_actions,
        arguments.seat,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
