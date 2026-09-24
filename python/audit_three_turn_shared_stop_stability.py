from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

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
from .evaluate import paired_comparison_summary
from .train_three_turn_plan import checked_dataset


Phase = Literal["exploratory", "confirmation"]


@dataclass(frozen=True)
class ProbeConfiguration:
    first_seed: int
    maps: int
    maximum_samples: int
    protocol: str


CONFIGURATIONS: dict[Phase, ProbeConfiguration] = {
    "exploratory": ProbeConfiguration(
        6460000,
        64,
        24,
        "benchmarks/protocols/2026-09-24-duel-shared-stop-opponent-stability-v1.json",
    ),
    "confirmation": ProbeConfiguration(
        6461000,
        128,
        48,
        "benchmarks/protocols/2026-09-24-duel-shared-stop-opponent-confirmation-v1.json",
    ),
}


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
    conditional = {"beneficial": 0, "harmful": 0, "neutral": 0, "censored": 0}
    by_seat = {
        seat: {"beneficial": 0, "harmful": 0, "neutral": 0, "censored": 0}
        for seat in (0, 1)
    }
    for sample in samples:
        root = cast(int, sample["root_seat"])
        outcomes = cast(dict[str, dict[str, BranchOutcome]], sample["outcomes"])
        direct = preference(outcomes["direct"], root)
        two_turn = preference(outcomes["two_turn"], root)
        if direct is None or two_turn is None:
            cross["censored_in_either_mode"] += 1
            label = "censored"
        elif direct == two_turn == 0:
            cross["both_tied"] += 1
            label = "neutral"
        elif direct == two_turn:
            cross["strict_agreement"] += 1
            label = "beneficial" if direct > 0 else "harmful"
        elif direct == -two_turn:
            cross["strict_reversal"] += 1
            label = "harmful"
        elif direct != 0:
            cross["direct_only_strict"] += 1
            label = "beneficial" if direct > 0 else "harmful"
        else:
            cross["two_turn_only_strict"] += 1
            label = "beneficial" if two_turn > 0 else "harmful"
        conditional[label] += 1
        by_seat[root][label] += 1
    sign = paired_comparison_summary(
        conditional["beneficial"], conditional["harmful"], conditional["neutral"]
    )
    worst_case = paired_comparison_summary(
        conditional["beneficial"],
        conditional["harmful"] + conditional["censored"],
        conditional["neutral"],
    )
    return {
        "by_opponent_policy": modes,
        "cross_policy": cross,
        "conditional_nonharmful_signal": {
            **conditional,
            "by_root_seat": by_seat,
            "exact_two_sided_map_sign_test_p": sign["exact_two_sided_sign_test_p"],
            "censored_as_harmful_sign_test_p": worst_case[
                "exact_two_sided_sign_test_p"
            ],
        },
    }


def audit(
    dataset_path: Path,
    checkpoint_path: Path,
    phase: Phase = "exploratory",
) -> dict[str, object]:
    configuration = CONFIGURATIONS[phase]
    if digest(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("opponent checkpoint disagrees with the fixed protocol")
    dataset = checked_dataset(
        dataset_path, configuration.first_seed, configuration.maps
    )
    if (
        dataset["completed_maps"] != configuration.maps
        or dataset["truncated_maps"] != 0
    ):
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
        if len(samples) == configuration.maximum_samples:
            break
    return {
        "kind": "three_turn_shared_stop_opponent_stability_probe",
        "protocol": configuration.protocol,
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
    parser.add_argument("--phase", choices=tuple(CONFIGURATIONS), default="exploratory")
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(arguments.dataset, arguments.checkpoint, arguments.phase),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
