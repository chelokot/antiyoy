from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import concatenate_observations, encode_rules_batch

from .audit_duel_first_regret import (
    CHECKPOINT_SHA256,
    create_environment,
    load_routed_policy,
    native_teacher_action,
)
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .build_bundle import digest


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-spatial-process-feasibility-v1.json"
FIRST_SEED = 6570000
MAPS = 8


def replay_candidates(
    environment: VectorEnv,
    plans: list[list[int]],
    static_scores: list[int],
    root: int,
) -> tuple[list[dict[str, np.ndarray]], int]:
    observations = []
    terminal = 0
    for plan, static_score in zip(plans, static_scores, strict=True):
        branch = environment.fork(np.asarray([0], dtype=np.uint64))
        for index in plan:
            current = branch.observe()
            legal = int(np.diff(current["action_offsets"])[0])
            if not 0 <= index < legal:
                raise ValueError("native plan action index is not legal on replay")
            result = branch.step(np.asarray([index], dtype=np.uint64))
            if bool(result["truncated"][0]):
                raise ValueError("native plan was action-limit censored")
        if int(branch.position_scores(root)[0]) != static_score:
            raise ValueError("native candidate static score disagrees with replay")
        if branch.done()[0]:
            terminal += 1
            continue
        observed = branch.observe()
        if int(observed["active_players"][0]) == root:
            raise ValueError("native plan did not complete the root turn")
        observations.append(observed)
    return observations, terminal


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def summarize(samples: list[dict[str, float | int]]) -> dict[str, object]:
    teacher = [float(sample["teacher_search_seconds"]) for sample in samples]
    proposed = [
        sum(
            float(sample[key])
            for key in (
                "root_slate_seconds",
                "candidate_replay_seconds",
                "batched_encoder_seconds",
            )
        )
        for sample in samples
    ]
    return {
        "samples": len(samples),
        "median_teacher_search_seconds": statistics.median(teacher),
        "p95_teacher_search_seconds": percentile(teacher, 0.95),
        "median_root_slate_replay_encoder_seconds": statistics.median(proposed),
        "p95_root_slate_replay_encoder_seconds": percentile(proposed, 0.95),
        "total_candidate_plans": sum(int(sample["candidate_plans"]) for sample in samples),
        "terminal_candidate_plans": sum(
            int(sample["terminal_candidates"]) for sample in samples
        ),
        "advance_gate_passed": (
            len(samples) == MAPS * 2
            and statistics.median(proposed) <= 0.75 * statistics.median(teacher)
            and percentile(proposed, 0.95) <= percentile(teacher, 0.95)
        ),
    }


def audit(source_path: Path) -> dict[str, object]:
    if digest(source_path) != CHECKPOINT_SHA256:
        raise ValueError("frozen source checkpoint disagrees with the protocol")
    torch.set_num_threads(1)
    policy, experts = load_routed_policy(source_path)
    samples = []
    with torch.inference_mode():
        for seed in range(FIRST_SEED, FIRST_SEED + MAPS):
            environment = create_environment(seed)
            seen = set()
            while not environment.done()[0] and len(seen) < 2:
                observation = environment.observe()
                seat = int(observation["active_players"][0])

                def native_action() -> tuple[np.ndarray, float]:
                    started = time.perf_counter()
                    selected = native_teacher_action(
                        environment, FOLLOWUP_NODES, True
                    )
                    return selected, time.perf_counter() - started

                def candidate_features() -> dict[str, float | int]:
                    started = time.perf_counter()
                    plans, scores = environment.search_turn_plans(
                        node_budget=256,
                        slate_size=8,
                        beam_width=32,
                        branch_width=48,
                        maximum_actions_per_turn=24,
                    )
                    slate_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    post_turn, terminal = replay_candidates(
                        environment, plans[0], scores[0], seat
                    )
                    replay_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    encoded = concatenate_observations([observation, *post_turn])
                    rules = encode_rules_batch(
                        environment.rules_jsons() * (len(post_turn) + 1),
                        torch.device("cpu"),
                    )
                    model = policy.models[experts[seat]]
                    _, _, spatial = model.forward_with_value_features(encoded, rules)
                    if spatial.shape != (len(post_turn) + 1, model.hidden * 2):
                        raise ValueError("batched spatial features have an unexpected shape")
                    encoder_seconds = time.perf_counter() - started
                    return {
                        "root_slate_seconds": slate_seconds,
                        "candidate_replay_seconds": replay_seconds,
                        "batched_encoder_seconds": encoder_seconds,
                        "candidate_plans": len(plans[0]),
                        "terminal_candidates": terminal,
                    }

                if seat not in seen:
                    if seed % 2:
                        candidate = candidate_features()
                        action, teacher_seconds = native_action()
                    else:
                        action, teacher_seconds = native_action()
                        candidate = candidate_features()
                    samples.append(
                        {
                            "seed": seed,
                            "seat": seat,
                            "teacher_search_seconds": teacher_seconds,
                            **candidate,
                        }
                    )
                    seen.add(seat)
                else:
                    action, _ = native_action()
                result = environment.step(action)
                if bool(result["truncated"][0]):
                    raise ValueError("teacher self-play was action-limit censored")
            if seen != {0, 1}:
                raise ValueError("teacher game did not expose both root seats")
    summary = summarize(samples)
    return {
        "kind": "read_only_spatial_turn_process_feasibility",
        "protocol": PROTOCOL,
        "source_sha256": CHECKPOINT_SHA256,
        "selected_experts_by_seat": experts,
        "summary": summary,
        "samples": samples,
        "qualification": "Same-state CPU interface and latency feasibility only, not a trained selector, game outcome or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.source), sort_keys=True))


if __name__ == "__main__":
    main()
