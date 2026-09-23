from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl.counterfactual import CandidateOutcome, rollout_candidates
from antiyoy_rl.model import action_distribution, encode_rules_batch

try:
    from .benchmark_counterfactual import outcome_record
    from .build_bundle import digest
    from .collect_action_q import (
        add_collection_arguments,
        collection_config,
        load_collection_policy,
    )
    from .distill_puct import (
        PuctDistillationConfig,
        create_environment,
        domain_descriptor,
        validate_config,
    )
    from .evaluate import (
        load_policy_checkpoint,
        paired_method_comparison,
        winner_score,
    )
except ImportError:
    from benchmark_counterfactual import outcome_record
    from build_bundle import digest
    from collect_action_q import (
        add_collection_arguments,
        collection_config,
        load_collection_policy,
    )
    from distill_puct import (
        PuctDistillationConfig,
        create_environment,
        domain_descriptor,
        validate_config,
    )
    from evaluate import (
        load_policy_checkpoint,
        paired_method_comparison,
        winner_score,
    )


def select_intervention(outcomes: list[CandidateOutcome], seat: int) -> int:
    return max(
        range(len(outcomes)),
        key=lambda index: (
            0.5
            if outcomes[index].winner is None
            else winner_score(outcomes[index].winner, seat),
            outcomes[index].territory,
        ),
    )


def evaluate_intervention(
    checkpoint_path: Path,
    config: PuctDistillationConfig,
    model_seat: int,
    candidate_count: int,
    rollout_horizon: int,
    intervention_round: int,
) -> dict[str, object]:
    validate_config(config)
    if not 0 <= model_seat < config.players:
        raise ValueError("model seat is outside the player range")
    if candidate_count < 2 or rollout_horizon < 1 or intervention_round < 0:
        raise ValueError("counterfactual evaluation settings are invalid")
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    checkpoint_config = dict(checkpoint["config"])
    descriptor = domain_descriptor(config, checkpoint_config)
    policy, selected_experts, _, _ = load_collection_policy(
        checkpoint, config, descriptor, device
    )
    started = time.perf_counter()

    def play_games(intervene: bool) -> tuple[np.ndarray, int, list[dict[str, object]]]:
        environment = create_environment(config, checkpoint_config)
        for index in range(config.environments):
            environment.reset(index, config.seed + index)
        rules = encode_rules_batch(environment.rules_jsons(), device)
        finished = np.zeros(config.environments, dtype=np.bool_)
        intervened = np.zeros(config.environments, dtype=np.bool_)
        episode_steps = np.zeros(config.environments, dtype=np.int32)
        winners = np.full(config.environments, 255, dtype=np.uint8)
        truncations = 0
        next_seed = config.seed + config.environments
        records: list[dict[str, object]] = []
        while not bool(finished.all()):
            observation = environment.observe()
            actions = policy.actions(observation, rules)
            if intervene:
                eligible = np.flatnonzero(
                    np.logical_and.reduce(
                        (
                            np.logical_not(finished),
                            np.logical_not(intervened),
                            observation["active_players"] == model_seat,
                            observation["rounds"] >= intervention_round,
                        )
                    )
                )
                if eligible.size:
                    with torch.no_grad():
                        logits, _ = policy(observation, rules)
                        probabilities = (
                            action_distribution(logits, observation["action_offsets"])
                            .probs.cpu()
                            .numpy()
                        )
                    offsets = np.asarray(observation["action_offsets"], dtype=np.int64)
                    for index in eligible:
                        count = int(offsets[index + 1] - offsets[index])
                        if count < 2:
                            continue
                        candidates = np.argsort(
                            -probabilities[index, :count], kind="stable"
                        )[:candidate_count].astype(np.uint64)
                        if candidates[0] != actions[index]:
                            raise RuntimeError(
                                "candidate order disagrees with the direct policy"
                            )
                        outcomes = rollout_candidates(
                            environment,
                            int(index),
                            candidates,
                            policy,
                            rules,
                            rollout_horizon,
                        )
                        selected = select_intervention(outcomes, model_seat)
                        actions[index] = outcomes[selected].action
                        intervened[index] = True
                        records.append(
                            {
                                "seed": config.seed + int(index),
                                "episode_step": int(episode_steps[index]),
                                "round": int(observation["rounds"][index]),
                                "direct_action": outcomes[0].action,
                                "chosen_action": outcomes[selected].action,
                                "candidates": [
                                    outcome_record(outcome, model_seat)
                                    for outcome in outcomes
                                ],
                            }
                        )
            result = environment.step(actions)
            episode_steps += 1
            done = np.logical_or(result["terminal"], result["truncated"])
            for index in np.flatnonzero(done):
                if not finished[index]:
                    truncated = bool(result["truncated"][index])
                    truncations += int(truncated)
                    winner_key = "adjudicated_winners" if truncated else "winners"
                    winners[index] = result[winner_key][index]
                    finished[index] = True
                environment.reset(int(index), next_seed)
                next_seed += 1
        return winners, truncations, records

    baseline_winners, baseline_truncations, _ = play_games(False)
    candidate_winners, candidate_truncations, interventions = play_games(True)
    baseline_scores = np.array(
        [winner_score(int(winner), model_seat) for winner in baseline_winners]
    )
    candidate_scores = np.array(
        [winner_score(int(winner), model_seat) for winner in candidate_winners]
    )
    paired = paired_method_comparison(candidate_scores, baseline_scores)
    return {
        "schema_version": 1,
        "kind": "single_intervention_counterfactual_evaluation",
        "source": {
            "path": str(checkpoint_path),
            "sha256": digest(checkpoint_path),
            "seat_experts": selected_experts,
        },
        "config": {
            "domain": asdict(config),
            "descriptor": descriptor,
            "model_seat": model_seat,
            "candidate_count": candidate_count,
            "rollout_horizon": rollout_horizon,
            "intervention_round": intervention_round,
        },
        "baseline_winners": baseline_winners.tolist(),
        "candidate_winners": candidate_winners.tolist(),
        "baseline_wins": int(np.count_nonzero(baseline_winners == model_seat)),
        "candidate_wins": int(np.count_nonzero(candidate_winners == model_seat)),
        "baseline_truncations": baseline_truncations,
        "candidate_truncations": candidate_truncations,
        "paired": paired,
        "interventions": interventions,
        "changed_actions": sum(
            record["chosen_action"] != record["direct_action"]
            for record in interventions
        ),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    add_collection_arguments(parser)
    parser.add_argument("--model-seat", type=int, required=True)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--rollout-horizon", type=int, default=96)
    parser.add_argument("--intervention-round", type=int, default=0)
    arguments = parser.parse_args()
    report = evaluate_intervention(
        arguments.checkpoint,
        collection_config(arguments),
        arguments.model_seat,
        arguments.candidate_count,
        arguments.rollout_horizon,
        arguments.intervention_round,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
