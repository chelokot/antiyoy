from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import (
    UniversalPolicy,
    concatenate_observations,
    encode_rules_batch,
)
from antiyoy_rl.slate_dataset import (
    replay_slate_candidate,
    replay_slate_positions,
    step_action_index,
)

from .audit_duel_autonomous_reply import MAXIMUM_RESPONSE_ACTIONS
from .build_bundle import digest
from .train_duel_opponent_plan import load_head_student


def collect_candidate(
    environment: VectorEnv,
    student: UniversalPolicy,
    rules: torch.Tensor,
) -> tuple[list[dict[str, np.ndarray]], list[int], list[int], bool]:
    observations: list[dict[str, np.ndarray]] = []
    labels: list[int] = []
    student_actions: list[int] = []
    if environment.done()[0]:
        return observations, labels, student_actions, False
    opponent = int(environment.observe()["active_players"][0])
    for _ in range(MAXIMUM_RESPONSE_ACTIONS):
        observation = environment.observe()
        if int(observation["active_players"][0]) != opponent:
            return observations, labels, student_actions, False
        teacher = int(environment.search_actions_replanned(node_budget=64)[0])
        with torch.inference_mode():
            logits, _ = student(observation, rules)
        chosen = int(logits.argmax())
        legal_count = len(observation["action_kinds"])
        if not (0 <= teacher < legal_count and 0 <= chosen < legal_count):
            raise ValueError("corrective action is outside the legal-action slate")
        observations.append(observation)
        labels.append(teacher)
        student_actions.append(chosen)
        if step_action_index(environment, chosen):
            return observations, labels, student_actions, True
        if environment.done()[0]:
            return observations, labels, student_actions, False
    still_active = int(environment.observe()["active_players"][0]) == opponent
    return observations, labels, student_actions, still_active


def collect(
    slate_path: Path,
    source_path: Path,
    head_path: Path,
    output_path: Path,
) -> dict[str, object]:
    torch.set_num_threads(1)
    with gzip.open(slate_path, "rt", encoding="utf-8") as source_file:
        dataset = json.load(source_file)
    _, student = load_head_student(source_path, head_path)
    records = []
    corrective_actions = matched_actions = censored_candidates = candidates = 0
    sampled_maps = set()
    for position in replay_slate_positions(dataset):
        sampled_maps.add(position.seed)
        rules_json = position.root.rules_json()
        rules = encode_rules_batch([rules_json], torch.device("cpu"))
        offsets = [0]
        observations = []
        labels = []
        chosen = []
        for candidate in range(len(position.record["candidate_action_indices"])):
            branch = replay_slate_candidate(position, candidate)
            states, teacher_actions, student_actions, censored = collect_candidate(
                branch, student, rules
            )
            observations.extend(states)
            labels.extend(teacher_actions)
            chosen.extend(student_actions)
            offsets.append(len(labels))
            censored_candidates += censored
            candidates += 1
        corrective_actions += len(labels)
        matched_actions += sum(
            teacher == action for teacher, action in zip(labels, chosen, strict=True)
        )
        records.append(
            {
                "seed": position.seed,
                "seat": position.record["seat"],
                "round": position.record["round"],
                "candidate_offsets": offsets,
                "observation": {
                    key: values.tolist()
                    for key, values in concatenate_observations(observations).items()
                }
                if observations
                else None,
                "rules_json": [rules_json] * len(labels),
                "teacher_action_indices": labels,
                "student_action_indices": chosen,
            }
        )
    report = {
        "schema_version": 1,
        "kind": "procedural_duel_corrective_opponent_decisions",
        "slate_sha256": digest(slate_path),
        "source_sha256": digest(source_path),
        "rollout_student_head_sha256": digest(head_path),
        "maps": dataset["maps"],
        "completed_maps": dataset["completed_maps"],
        "truncated_maps": dataset["truncated_maps"],
        "sampled_maps": len(sampled_maps),
        "positions": len(records),
        "candidate_states": candidates,
        "corrective_actions": corrective_actions,
        "student_actions_matching_corrective_teacher": matched_actions,
        "censored_candidate_replies": censored_candidates,
        "records": records,
    }
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as output:
        json.dump(report, output, separators=(",", ":"))
    return {key: value for key, value in report.items() if key != "records"} | {
        "output_sha256": digest(output_path),
        "output_bytes": output_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("slates", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            collect(
                arguments.slates, arguments.source, arguments.head, arguments.output
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
