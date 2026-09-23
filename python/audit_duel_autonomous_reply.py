from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import (
    ACTION_KIND_NAMES,
    UniversalPolicy,
    encode_rules_batch,
)
from antiyoy_rl.slate_dataset import (
    replay_slate_candidate,
    replay_slate_positions,
    step_action_index,
)

from .build_bundle import digest
from .evaluate import paired_comparison_summary
from .train_duel_opponent_plan import load_head_student


MAXIMUM_RESPONSE_ACTIONS = 24


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
        if step_action_index(environment, action):
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


def candidate_score_record(
    seed: int,
    record: dict[str, object],
    autonomous_scores: dict[str, list[int | None]],
) -> dict[str, object]:
    static_scores = record["static_scores"]
    return {
        "seed": seed,
        "root_seat": record["seat"],
        "round": record["round"],
        "native_selected_index": record["selected_index"],
        "static_scores": static_scores,
        "native_reply_scores": record["reply_scores"],
        "autonomous_reply_scores": autonomous_scores,
        "autonomous_selected_indices": {
            name: (
                selected_candidate([int(score) for score in scores], static_scores)
                if all(score is not None for score in scores)
                else None
            )
            for name, scores in autonomous_scores.items()
        },
    }


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


def searched_reply_diagnostic(
    start: VectorEnv,
    policies: dict[str, UniversalPolicy],
    rules: torch.Tensor,
    root_seat: int,
    expected_score: int,
) -> dict[str, object]:
    environment = start.fork(np.asarray([0], dtype=np.uint64))
    first_mismatch: dict[str, int | None] = {name: None for name in policies}
    first_mismatch_kind: dict[str, str | None] = {name: None for name in policies}
    first_actions: dict[str, int | None] = {name: None for name in policies}
    teacher_actions = 0
    if not environment.done()[0]:
        opponent = int(environment.observe()["active_players"][0])
        for depth in range(MAXIMUM_RESPONSE_ACTIONS):
            observation = environment.observe()
            if int(observation["active_players"][0]) != opponent:
                break
            selected = int(environment.search_actions(node_budget=64)[0])
            action_kind = ACTION_KIND_NAMES[int(observation["action_kinds"][selected])]
            for name, model in policies.items():
                with torch.inference_mode():
                    logits, _ = model(observation, rules)
                predicted = int(logits.argmax())
                if depth == 0:
                    first_actions[name] = predicted
                if first_mismatch[name] is None and predicted != selected:
                    first_mismatch[name] = depth
                    first_mismatch_kind[name] = action_kind
            if step_action_index(environment, selected):
                raise ValueError("native searched reply was action-limit adjudicated")
            teacher_actions += 1
            if environment.done()[0]:
                break
        if (
            not environment.done()[0]
            and int(environment.observe()["active_players"][0]) == opponent
        ):
            raise ValueError(
                "native searched reply exceeded the full-turn action bound"
            )
    actual_score = int(environment.position_scores(root_seat)[0])
    if actual_score != expected_score:
        raise ValueError(
            "native searched reply score disagrees with stored teacher score"
        )
    return {
        "teacher_actions": teacher_actions,
        "first_mismatch": first_mismatch,
        "first_mismatch_kind": first_mismatch_kind,
        "first_actions_differ": first_actions["source"] != first_actions["student"],
    }


def audit(
    dataset_path: Path,
    source_path: Path,
    head_path: Path,
    diagnose: bool = False,
    include_candidate_scores: bool = False,
) -> dict[str, object]:
    torch.set_num_threads(1)
    with gzip.open(dataset_path, "rt", encoding="utf-8") as source_file:
        dataset = json.load(source_file)
    generator = dataset["generator"]
    if dataset["schema_version"] != 1 or generator["players"] != 2:
        raise ValueError(
            "autonomous reply audit requires version-one two-player slates"
        )
    source, student = load_head_student(source_path, head_path)
    policies = {"source": source, "student": student}
    response_errors: dict[int, list[tuple[float, float]]] = defaultdict(list)
    seat_errors: dict[int, list[tuple[float, float]]] = defaultdict(list)
    action_counts = {name: [] for name in policies}
    censored = {name: 0 for name in policies}
    selected_matches = {name: 0 for name in policies}
    selected_complete = 0
    candidate_count = 0
    checked_positions = 0
    divergence = []
    candidate_scores = []
    for position in replay_slate_positions(dataset):
        seed, record = position.seed, position.record
        rules = encode_rules_batch([position.root.rules_json()], torch.device("cpu"))
        root_seat = int(record["seat"])
        autonomous_scores: dict[str, list[int | None]] = {name: [] for name in policies}
        for candidate in range(len(record["candidate_action_indices"])):
            branch = replay_slate_candidate(position, candidate)
            candidate_count += 1
            teacher_diagnostic = (
                searched_reply_diagnostic(
                    branch,
                    policies,
                    rules,
                    root_seat,
                    record["reply_scores"][candidate],
                )
                if diagnose
                else None
            )
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
                if teacher_diagnostic is not None:
                    divergence.append(
                        {
                            "seed": seed,
                            "root_seat": root_seat,
                            "round": record["round"],
                            "candidate": candidate,
                            "native_selected_index": record["selected_index"],
                            "native_reply_score": expected_score,
                            "source_reply_score": first,
                            "student_reply_score": second,
                            **teacher_diagnostic,
                            "source_reply_actions": action_counts["source"][-1],
                            "student_reply_actions": action_counts["student"][-1],
                            "source_score_error": pair[0],
                            "student_score_error": pair[1],
                        }
                    )
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
        if include_candidate_scores:
            candidate_scores.append(
                candidate_score_record(seed, record, autonomous_scores)
            )
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
    report = {
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
    if diagnose:
        report["divergence"] = divergence
    if include_candidate_scores:
        report["candidate_score_records"] = candidate_scores
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--include-candidate-scores", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(
                arguments.dataset,
                arguments.source,
                arguments.head,
                diagnose=arguments.diagnose,
                include_candidate_scores=arguments.include_candidate_scores,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
