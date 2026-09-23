from __future__ import annotations

import argparse
import copy
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl import ProceduralConfig, VectorEnv
from antiyoy_rl.model import UniversalPolicy, encode_rules_batch, select_environments
from antiyoy_rl.turn_credit import model_observation

from .build_bundle import digest
from .evaluate import load_policy, paired_comparison_summary


MAXIMUM_RESPONSE_ACTIONS = 24


def step_index(environment: VectorEnv, index: int) -> bool:
    result = environment.step(np.asarray([index], dtype=np.uint64))
    return bool(result["truncated"][0])


def response_score(
    start: VectorEnv,
    model: UniversalPolicy,
    rules: torch.Tensor,
    root_seat: int,
) -> tuple[int | None, int]:
    environment = start.fork(np.asarray([0], dtype=np.uint64))
    if environment.done()[0]:
        return int(environment.position_scores(root_seat)[0]), 0
    opponent = int(environment.observe()["active_players"][0])
    for depth in range(MAXIMUM_RESPONSE_ACTIONS):
        observation = environment.observe()
        if int(observation["active_players"][0]) != opponent:
            return int(environment.position_scores(root_seat)[0]), depth
        with torch.inference_mode():
            logits, _ = model(observation, rules)
        action = int(logits.argmax())
        if step_index(environment, action):
            return None, depth + 1
        if environment.done()[0]:
            return int(environment.position_scores(root_seat)[0]), depth + 1
    observation = environment.observe()
    if int(observation["active_players"][0]) == opponent:
        return None, MAXIMUM_RESPONSE_ACTIONS
    return int(environment.position_scores(root_seat)[0]), MAXIMUM_RESPONSE_ACTIONS


def transformed_error(actual: int, expected: int) -> float:
    return abs(float(np.arcsinh(actual / 1000)) - float(np.arcsinh(expected / 1000)))


def selected_candidate(scores: list[int], static_scores: list[int]) -> int:
    return max(
        range(len(scores)),
        key=lambda index: (scores[index], static_scores[index], -index),
    )


def compare_map_errors(
    response_errors: dict[int, list[tuple[float, float]]],
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    map_ledger = []
    better = worse = same = 0
    for seed, pairs in sorted(response_errors.items()):
        before, after = np.asarray(pairs, dtype=np.float64).mean(axis=0)
        better += int(after < before)
        worse += int(after > before)
        same += int(after == before)
        map_ledger.append(
            {
                "seed": seed,
                "paired_candidates": len(pairs),
                "source_mean_transformed_error": float(before),
                "student_mean_transformed_error": float(after),
            }
        )
    return paired_comparison_summary(better, worse, same), map_ledger


def audit(
    dataset_path: Path,
    source_path: Path,
    head_path: Path,
) -> dict[str, object]:
    torch.set_num_threads(1)
    with gzip.open(dataset_path, "rt", encoding="utf-8") as source_file:
        dataset = json.load(source_file)
    generator = dataset["generator"]
    if dataset["schema_version"] != 1 or generator["players"] != 2:
        raise ValueError(
            "autonomous reply audit requires version-one two-player slates"
        )
    source, config = load_policy(
        source_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    saved = torch.load(head_path, map_location="cpu", weights_only=True)
    if (
        saved["kind"] != "opponent_action_head_scout"
        or saved["source_sha256"] != digest(source_path)
        or saved["selected_expert"] != config["selected_expert"]
    ):
        raise ValueError("student head does not match the frozen source expert")
    student = copy.deepcopy(source)
    student.action_head.load_state_dict(saved["action_head"])
    if student.action_residual is not None:
        student.action_residual.load_state_dict(saved["action_residual"])
    source.eval()
    student.eval()
    policies = {"source": source, "student": student}
    response_errors: dict[int, list[tuple[float, float]]] = defaultdict(list)
    seat_errors: dict[int, list[tuple[float, float]]] = defaultdict(list)
    action_counts = {name: [] for name in policies}
    censored = {name: 0 for name in policies}
    selected_matches = {name: 0 for name in policies}
    selected_complete = 0
    candidate_count = 0
    checked_positions = 0
    by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in dataset["records"]:
        by_seed[record["seed"]].append(record)
    for seed, records in sorted(by_seed.items()):
        map_config = ProceduralConfig(
            width=generator["width"],
            height=generator["height"],
            players=2,
            seed=seed,
            land_density_per_million=generator["land_density_per_million"],
            starting_province_size=generator["starting_province_size"],
            starting_money=generator["starting_money"],
            tree_density_per_million=generator["tree_density_per_million"],
            neutral_tower_density_per_million=generator[
                "neutral_tower_density_per_million"
            ],
            neutral_capital_density_per_million=generator[
                "neutral_capital_density_per_million"
            ],
            grave_density_per_million=generator["grave_density_per_million"],
            schema_version=generator["schema_version"],
        )
        environment = VectorEnv.procedural(
            1,
            map_config,
            action_limit=dataset["action_limit"],
            profile=dataset["rules"],
        )
        rules = encode_rules_batch([environment.rules_json()], torch.device("cpu"))
        applied = 0
        for record in records:
            rollin = record["rollin_indices"]
            for index in rollin[applied:]:
                if step_index(environment, index):
                    raise ValueError("searched roll-in was action-limit adjudicated")
            applied = len(rollin)
            observed = environment.observe()
            if (
                int(observed["active_players"][0]) != record["seat"]
                or int(observed["rounds"][0]) != record["round"]
            ):
                raise ValueError("sampled root state disagrees with replayed roll-in")
            expected, _ = model_observation(record["post_turn"])
            root_seat = record["seat"]
            autonomous_scores: dict[str, list[int | None]] = {
                name: [] for name in policies
            }
            for candidate, indices in enumerate(record["candidate_action_indices"]):
                branch = environment.fork(np.asarray([0], dtype=np.uint64))
                for index in indices:
                    if step_index(branch, index):
                        raise ValueError("searched root candidate was adjudicated")
                actual_observation = branch.observe()
                expected_observation = select_environments(expected, [candidate])
                if any(
                    not np.array_equal(actual_observation[key], values)
                    for key, values in expected_observation.items()
                ):
                    raise ValueError("searched candidate state disagrees with replay")
                candidate_count += 1
                for name, model in policies.items():
                    score, actions = response_score(branch, model, rules, root_seat)
                    autonomous_scores[name].append(score)
                    action_counts[name].append(actions)
                    censored[name] += score is None
                first = autonomous_scores["source"][-1]
                second = autonomous_scores["student"][-1]
                if first is not None and second is not None:
                    expected_score = record["reply_scores"][candidate]
                    pair = (
                        transformed_error(first, expected_score),
                        transformed_error(second, expected_score),
                    )
                    response_errors[seed].append(pair)
                    seat_errors[root_seat].append(pair)
            if all(
                all(score is not None for score in scores)
                for scores in autonomous_scores.values()
            ):
                selected_complete += 1
                for name, scores in autonomous_scores.items():
                    choice = selected_candidate(
                        [score for score in scores if score is not None],
                        record["static_scores"],
                    )
                    selected_matches[name] += choice == record["selected_index"]
            checked_positions += 1
    comparison, map_ledger = compare_map_errors(response_errors)
    by_seat = {}
    for seat, pairs in sorted(seat_errors.items()):
        errors = np.asarray(pairs, dtype=np.float64).mean(axis=0)
        by_seat[str(seat)] = {
            "paired_candidates": len(pairs),
            "source_mean_transformed_error": float(errors[0]),
            "student_mean_transformed_error": float(errors[1]),
        }
    return {
        "kind": "procedural_duel_autonomous_opponent_reply_probe",
        "dataset_sha256": digest(dataset_path),
        "source_sha256": digest(source_path),
        "student_head_sha256": digest(head_path),
        "maps": dataset["maps"],
        "completed_maps": dataset["completed_maps"],
        "truncated_maps": dataset["truncated_maps"],
        "checked_positions": checked_positions,
        "candidate_states": candidate_count,
        "paired_non_censored_candidates": sum(map(len, response_errors.values())),
        "response_censored": censored,
        "mean_reply_actions": {
            name: float(np.mean(counts)) for name, counts in action_counts.items()
        },
        "independent_map_error_comparison": comparison,
        "map_ledger": map_ledger,
        "by_root_seat": by_seat,
        "slates_with_all_responses": selected_complete,
        "native_teacher_selected_turn_matches": selected_matches,
        "qualification": "Autonomous one-turn reply successor-score fidelity, not complete-game strength or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("head", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(arguments.dataset, arguments.source, arguments.head), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
