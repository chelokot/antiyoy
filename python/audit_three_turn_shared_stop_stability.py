from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch

from antiyoy_rl.model import encode_rules_batch
from antiyoy_rl.slate_dataset import replay_slate_positions

from .audit_duel_first_regret import (
    CHECKPOINT_SHA256,
    BranchOutcome,
    load_routed_policy,
    score,
)
from .audit_three_turn_shared_stop_outcomes import (
    finish_persistently,
    force_plan,
    shared_prefix_length,
    summarize,
)
from .build_bundle import digest
from .train_three_turn_plan import checked_dataset


PROTOCOL = "benchmarks/protocols/2026-09-24-duel-shared-stop-opponent-stability-v1.json"
FIRST_SEED = 6460000
MAPS = 64
MAXIMUM_SAMPLES = 24


def preference(outcomes: dict[str, BranchOutcome], root_seat: int) -> int | None:
    teacher = score(outcomes["teacher"], root_seat)
    static = score(outcomes["static"], root_seat)
    if teacher is None or static is None:
        return None
    return int(teacher > static) - int(teacher < static)


def summarize_stability(samples: list[dict[str, object]]) -> dict[str, object]:
    modes = {}
    for mode in ("direct", "two_turn"):
        modes[mode] = summarize(
            [
                {
                    "root_seat": sample["root_seat"],
                    "outcomes": cast(dict[str, object], sample["outcomes"])[mode],
                }
                for sample in samples
            ]
        )
    cross = {
        "strict_agreement": 0,
        "strict_reversal": 0,
        "direct_only_strict": 0,
        "two_turn_only_strict": 0,
        "both_tied": 0,
        "censored_in_either_mode": 0,
    }
    for sample in samples:
        root = cast(int, sample["root_seat"])
        outcomes = cast(dict[str, dict[str, BranchOutcome]], sample["outcomes"])
        direct = preference(outcomes["direct"], root)
        two_turn = preference(outcomes["two_turn"], root)
        if direct is None or two_turn is None:
            cross["censored_in_either_mode"] += 1
        elif direct == two_turn == 0:
            cross["both_tied"] += 1
        elif direct == two_turn:
            cross["strict_agreement"] += 1
        elif direct == -two_turn:
            cross["strict_reversal"] += 1
        elif direct != 0:
            cross["direct_only_strict"] += 1
        else:
            cross["two_turn_only_strict"] += 1
    return {"by_opponent_policy": modes, "cross_policy": cross}


def audit(dataset_path: Path, checkpoint_path: Path) -> dict[str, object]:
    if digest(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("opponent checkpoint disagrees with the fixed protocol")
    dataset = checked_dataset(dataset_path, FIRST_SEED, MAPS)
    if dataset["completed_maps"] != MAPS or dataset["truncated_maps"] != 0:
        raise ValueError("fresh searched roll-in includes incomplete maps")
    torch.set_num_threads(1)
    opponent, experts = load_routed_policy(checkpoint_path)
    samples = []
    selected_maps: set[int] = set()
    for position in replay_slate_positions(dataset):
        if position.seed in selected_maps:
            continue
        record = position.record
        teacher = cast(int, record["selected_index"])
        if teacher == 0:
            continue
        plans = cast(list[list[int]], record["candidate_action_indices"])
        prefix = shared_prefix_length(plans, teacher)
        if prefix is None:
            continue
        root = cast(int, record["seat"])
        rules = encode_rules_batch(position.root.rules_jsons(), torch.device("cpu"))
        outcomes: dict[str, dict[str, BranchOutcome]] = {}
        for mode in ("direct", "two_turn"):
            branches = {}
            for label, candidate in (("teacher", teacher), ("static", 0)):
                branch, result, actions = force_plan(position, candidate)
                branches[label] = finish_persistently(
                    branch, result, actions, root, opponent, rules, mode
                )
            outcomes[mode] = branches
        followup = cast(list[int], record["followup_scores"])
        samples.append(
            {
                "seed": position.seed,
                "root_seat": root,
                "round": record["round"],
                "identical_prefix_actions": prefix,
                "teacher_plan_actions": len(plans[teacher]),
                "static_plan_actions": len(plans[0]),
                "native_three_turn_score_gap": followup[teacher] - followup[0],
                "outcomes": outcomes,
            }
        )
        selected_maps.add(position.seed)
        if len(samples) == MAXIMUM_SAMPLES:
            break
    return {
        "kind": "three_turn_shared_stop_opponent_stability_probe",
        "protocol": PROTOCOL,
        "dataset_sha256": digest(dataset_path),
        "checkpoint_sha256": digest(checkpoint_path),
        "selected_experts": experts,
        "dataset_completed_maps": dataset["completed_maps"],
        "eligible_sampled_maps": len(samples),
        "samples": samples,
        "summary": summarize_stability(samples),
        "qualification": "Fresh conditional terminal labels under two frozen continuation policies, not a trained-policy strength test or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.dataset, arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
