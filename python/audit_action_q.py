from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl.counterfactual import rollout_candidates
from antiyoy_rl.model import encode_rules_batch

try:
    from .build_bundle import digest
    from .collect_action_q import (
        DATASET_KIND,
        create_replay_environment,
        load_collection_policy,
        observation_fingerprint,
    )
    from .distill_puct import PuctDistillationConfig
    from .evaluate import load_policy_checkpoint, winner_score
except ImportError:
    from build_bundle import digest
    from collect_action_q import (
        DATASET_KIND,
        create_replay_environment,
        load_collection_policy,
        observation_fingerprint,
    )
    from distill_puct import PuctDistillationConfig
    from evaluate import load_policy_checkpoint, winner_score


def audit_action_q(
    dataset_path: Path,
    checkpoint_path: Path,
    max_examples: int,
    horizon: int | None,
    device_name: str,
) -> dict[str, object]:
    if max_examples < 1 or (horizon is not None and horizon < 1):
        raise ValueError("audit limits must be positive")
    started = time.perf_counter()
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if dataset["kind"] != DATASET_KIND:
        raise ValueError("audit requires an action-Q pair dataset")
    source_sha256 = digest(checkpoint_path)
    if dataset["source"]["sha256"] != source_sha256:
        raise ValueError("audit checkpoint does not match the dataset source")
    config = dataset["config"]
    descriptor = config["descriptor"]
    examples = dataset["examples"]
    if "direct_actions" not in examples or "search_actions" not in examples:
        raise ValueError("audit requires replayable action indices")
    device = torch.device(device_name)
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    policy_config = PuctDistillationConfig(
        profile=str(config["profile"]),
        generator=str(config["generator"]),
        route_generator=str(config["route_generator"]),
        players=int(config["players"]),
        device=device_name,
    )
    policy, selected_experts, _, _ = load_collection_policy(
        checkpoint, policy_config, descriptor, device
    )
    if selected_experts != dataset["source"]["seat_experts"]:
        raise ValueError("audit selected different policy experts")
    available = int(examples["episode_seeds"].numel())
    if available == 0:
        raise ValueError("audit dataset has no action pairs")
    selected = np.linspace(
        0, available - 1, min(max_examples, available), dtype=np.int64
    )
    records: list[dict[str, object]] = []
    for example in selected:
        index = int(example)
        episode_seed = int(examples["episode_seeds"][index])
        environment = create_replay_environment(config, episode_seed)
        start = int(examples["replay_offsets"][index])
        end = int(examples["replay_offsets"][index + 1])
        for action in examples["replay_actions"][start:end]:
            result = environment.step(np.asarray([int(action)], dtype=np.uint64))
            if bool(result["terminal"][0]) or bool(result["truncated"][0]):
                raise ValueError("audit replay terminates before the sampled state")
        observation = environment.observe()
        if (
            observation_fingerprint(observation, 0)
            != examples["state_fingerprints"][index]
        ):
            raise ValueError("audit replay fingerprint mismatch")
        seat = int(examples["seats"][index])
        if int(observation["active_players"][0]) != seat:
            raise ValueError("audit replay active seat mismatch")
        direct_action = int(examples["direct_actions"][index])
        search_action = int(examples["search_actions"][index])
        if direct_action == search_action:
            raise ValueError("audit pair does not contain a search disagreement")
        rules = encode_rules_batch(environment.rules_jsons(), device)
        direct, search = rollout_candidates(
            environment,
            0,
            np.asarray([direct_action, search_action], dtype=np.uint64),
            policy,
            rules,
            horizon,
        )
        score_delta = (
            None
            if direct.censored or search.censored
            else winner_score(int(search.winner), seat)
            - winner_score(int(direct.winner), seat)
        )
        records.append(
            {
                "example": index,
                "episode_seed": episode_seed,
                "seat": seat,
                "round": int(examples["rounds"][index]),
                "direct_action": direct_action,
                "search_action": search_action,
                "root_value_delta": float(
                    examples["search_values"][index] - examples["direct_values"][index]
                ),
                "policy_logit_margin": float(examples["baseline_margins"][index]),
                "score_delta": score_delta,
                "direct_winner": direct.winner,
                "search_winner": search.winner,
                "direct_truncated": direct.truncated,
                "search_truncated": search.truncated,
                "direct_censored": direct.censored,
                "search_censored": search.censored,
                "direct_rollout_steps": direct.rollout_steps,
                "search_rollout_steps": search.rollout_steps,
            }
        )
    complete = [record for record in records if record["score_delta"] is not None]
    decisive = [record for record in complete if record["score_delta"] != 0]
    return {
        "schema_version": 1,
        "kind": "puct_action_q_terminal_audit",
        "dataset_sha256": digest(dataset_path),
        "checkpoint_sha256": source_sha256,
        "generator": config["generator"],
        "route_generator": config["route_generator"],
        "continuation": "frozen direct policy for every seat after candidate action",
        "horizon": horizon,
        "sampled_positions": len(records),
        "distinct_episode_seeds": len({record["episode_seed"] for record in records}),
        "complete_positions": len(complete),
        "censored_positions": len(records) - len(complete),
        "search_better": sum(record["score_delta"] > 0 for record in complete),
        "direct_better": sum(record["score_delta"] < 0 for record in complete),
        "same_outcome": sum(record["score_delta"] == 0 for record in complete),
        "root_value_ordering_correct": int(
            sum(
                np.sign(record["root_value_delta"]) == np.sign(record["score_delta"])
                for record in decisive
            )
        ),
        "root_value_ordering_incorrect": int(
            sum(
                np.sign(record["root_value_delta"]) != np.sign(record["score_delta"])
                for record in decisive
            )
        ),
        "adjudicated_branches": sum(
            int(record["direct_truncated"]) + int(record["search_truncated"])
            for record in records
        ),
        "seconds": time.perf_counter() - started,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit_action_q(
                arguments.dataset,
                arguments.checkpoint,
                arguments.max_examples,
                arguments.horizon,
                arguments.device,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
