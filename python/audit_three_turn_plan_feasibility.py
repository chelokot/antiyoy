from __future__ import annotations

import argparse
import gzip
import json
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import torch

from antiyoy_rl.model import (
    UniversalPolicy,
    action_distribution,
    concatenate_observations,
    encode_rules,
    select_environments,
)
from antiyoy_rl.slate_dataset import ReplayedSlatePosition, replay_slate_positions

from .build_bundle import digest
from .evaluate import load_policy


def candidate_decisions(
    position: ReplayedSlatePosition,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    plans = cast(list[list[int]], position.record["candidate_action_indices"])
    observations = []
    indices = []
    plan_offsets = [0]
    for candidate, actions in enumerate(plans):
        branch = position.root.fork(
            np.asarray([0], dtype=np.uint64), action_limit=2**32 - 1
        )
        for index in actions:
            observation = branch.observe()
            legal = int(np.diff(observation["action_offsets"])[0])
            if index < 0 or index >= legal:
                raise ValueError("exported plan action is not locally legal")
            observations.append(observation)
            indices.append(index)
            if bool(branch.step(np.asarray([index], dtype=np.uint64))["truncated"][0]):
                raise ValueError("exported candidate was action-limit adjudicated")
        expected = select_environments(position.post_turn, [candidate])
        actual = branch.observe()
        if any(
            not np.array_equal(actual[key], value) for key, value in expected.items()
        ):
            raise ValueError("exported complete plan disagrees with post-turn state")
        plan_offsets.append(len(indices))
    return (
        concatenate_observations(observations),
        np.asarray(indices, dtype=np.int64),
        np.asarray(plan_offsets, dtype=np.int64),
    )


def action_log_probabilities(
    model: UniversalPolicy,
    observations: dict[str, np.ndarray],
    action_indices: np.ndarray,
    rules_json: str,
) -> np.ndarray:
    rules = encode_rules(rules_json, torch.device("cpu"))
    with torch.inference_mode():
        logits, _ = model(observations, rules)
        distribution = action_distribution(logits, observations["action_offsets"])
        labels = torch.as_tensor(action_indices, dtype=torch.long)
        log_probabilities = distribution.log_prob(labels).numpy()
    return log_probabilities


def mean_plan_log_probabilities(
    action_log_probs: np.ndarray, plan_offsets: np.ndarray
) -> np.ndarray:
    return np.asarray(
        [
            action_log_probs[start:end].mean()
            for start, end in zip(plan_offsets[:-1], plan_offsets[1:], strict=True)
        ],
        dtype=np.float64,
    )


def plan_log_probabilities(
    model: UniversalPolicy,
    observations: dict[str, np.ndarray],
    action_indices: np.ndarray,
    plan_offsets: np.ndarray,
    rules_json: str,
) -> np.ndarray:
    return mean_plan_log_probabilities(
        action_log_probabilities(model, observations, action_indices, rules_json),
        plan_offsets,
    )


def selected_plan(scores: np.ndarray, static_scores: list[int]) -> int:
    return max(
        range(len(scores)),
        key=lambda index: (scores[index], static_scores[index], -index),
    )


def audit(dataset_path: Path, source_path: Path) -> dict[str, object]:
    with gzip.open(dataset_path, "rt", encoding="utf-8") as source:
        dataset = cast(dict[str, object], json.load(source))
    if dataset["followup_search_nodes"] != 32:
        raise ValueError("feasibility dataset must use the validated three-turn teacher")
    model, selected_expert = load_policy(
        source_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    torch.set_num_threads(1)
    positions = plans = decisions = matches = 0
    replay_seconds = inference_seconds = 0.0
    seat_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for position in replay_slate_positions(dataset):
        started = time.perf_counter()
        observations, indices, plan_offsets = candidate_decisions(position)
        replay_seconds += time.perf_counter() - started
        started = time.perf_counter()
        scores = plan_log_probabilities(
            model,
            observations,
            indices,
            plan_offsets,
            position.root.rules_json(),
        )
        inference_seconds += time.perf_counter() - started
        record = position.record
        seat = cast(int, record["seat"])
        teacher = cast(int, record["selected_index"])
        match = selected_plan(scores, cast(list[int], record["static_scores"])) == teacher
        positions += 1
        plans += len(scores)
        decisions += len(indices)
        matches += match
        seat_counts[seat][0] += 1
        seat_counts[seat][1] += match
    return {
        "kind": "three_turn_plan_feasibility",
        "dataset_sha256": digest(dataset_path),
        "source_sha256": digest(source_path),
        "selected_expert": selected_expert["selected_expert"],
        "maps": dataset["maps"],
        "completed_maps": dataset["completed_maps"],
        "truncated_maps": dataset["truncated_maps"],
        "positions": positions,
        "candidate_plans": plans,
        "candidate_decisions": decisions,
        "source_teacher_plan_matches": matches,
        "seat_positions_and_matches": dict(sorted(seat_counts.items())),
        "collection_seconds": dataset["elapsed_seconds"],
        "replay_seconds": replay_seconds,
        "inference_seconds": inference_seconds,
        "gzip_bytes": dataset_path.stat().st_size,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.dataset, arguments.source), sort_keys=True))


if __name__ == "__main__":
    main()
