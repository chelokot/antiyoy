from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, TypedDict, cast

import numpy as np
import torch

from antiyoy_rl import ProceduralConfig, VectorEnv
from antiyoy_rl.model import RULE_FEATURES, UniversalPolicy, domain_key, encode_rules_batch
from antiyoy_rl.model_reply_search import ModelReplySearch
from antiyoy_rl.routed import RoutedPolicy

from .build_bundle import digest
from .evaluate import (
    instantiate_policy,
    load_policy_checkpoint,
    paired_comparison_summary,
    select_policy_state,
)


PROFILE = "classic_generic_2022"
GENERATOR = "procedural_v1"
CHECKPOINT_SHA256 = "68549a5af87e4d2d065a164e39bc515c2c1d48276f437494d757e25c3ef28867"
ACTION_LIMIT = 2400
SEARCH_NODES = 256
REPLY_NODES = 64
SLATE_SIZE = 8
BEAM_WIDTH = 32
BRANCH_WIDTH = 48
MAXIMUM_ACTIONS_PER_TURN = 24


class BranchOutcome(TypedDict):
    terminal: bool
    truncated: bool
    winner: int | None
    adjudicated_winner: int | None
    actions_after_intervention: int


def create_environment(seed: int) -> VectorEnv:
    config = ProceduralConfig(
        width=11,
        height=9,
        players=2,
        seed=seed,
        schema_version=2,
    )
    return VectorEnv.procedural(1, config, action_limit=ACTION_LIMIT, profile=PROFILE)


def load_routed_policy(checkpoint_path: Path) -> tuple[RoutedPolicy, list[str]]:
    device = torch.device("cpu")
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    config = cast(dict[str, object], checkpoint["config"])
    descriptor: dict[str, object] = {
        "width": 11,
        "height": 9,
        "players": 2,
        "action_limit": ACTION_LIMIT,
        "fog": config["fog"],
        "diplomacy": config.get("diplomacy", False),
        "initial_relation": config.get("initial_relation", "neutral"),
        "land_density_per_million": 650_000,
        "starting_province_size": 5,
        "starting_money": 10,
        "tree_density_per_million": 150_000,
        "neutral_tower_density_per_million": 20_000,
        "neutral_capital_density_per_million": 10_000,
        "grave_density_per_million": 15_000,
    }
    route_domain = domain_key(GENERATOR, descriptor)
    models: dict[str, UniversalPolicy] = {}
    experts = []
    for seat in range(2):
        state, seat_config = select_policy_state(
            checkpoint, PROFILE, GENERATOR, 2, seat, route_domain
        )
        expert = str(seat_config["selected_expert"])
        if expert not in models:
            models[expert] = instantiate_policy(state, seat_config, device)
        experts.append(expert)
    return RoutedPolicy(models, experts), experts


def new_hybrid(audit_native_replies: bool) -> ModelReplySearch:
    return ModelReplySearch(
        node_budget=SEARCH_NODES,
        slate_size=SLATE_SIZE,
        beam_width=BEAM_WIDTH,
        branch_width=BRANCH_WIDTH,
        maximum_actions_per_turn=MAXIMUM_ACTIONS_PER_TURN,
        audit_native_replies=audit_native_replies,
        audit_reply_nodes=REPLY_NODES,
        audit_round_modulus=1,
    )


def native_teacher_action(environment: VectorEnv) -> np.ndarray:
    return np.asarray(
        environment.reply_search_actions(
            node_budget=SEARCH_NODES,
            reply_nodes=REPLY_NODES,
            slate_size=SLATE_SIZE,
            beam_width=BEAM_WIDTH,
            branch_width=BRANCH_WIDTH,
            maximum_actions_per_turn=MAXIMUM_ACTIONS_PER_TURN,
        ),
        dtype=np.uint64,
    )


def first_strict_regret(record: Mapping[str, object]) -> tuple[int, int, int] | None:
    native = cast(list[int], record["native_reply_scores"])
    selected = cast(int, record["autonomous_selected_index"])
    teacher = cast(int, record["native_selected_index"])
    gap = native[teacher] - native[selected]
    return (selected, teacher, gap) if gap > 0 else None


def outcome(result: Mapping[str, np.ndarray], actions: int) -> BranchOutcome:
    truncated = bool(result["truncated"][0])
    terminal = bool(result["terminal"][0])
    if not terminal and not truncated:
        raise ValueError("outcome requires a finished branch")
    return {
        "terminal": terminal,
        "truncated": truncated,
        "winner": None if truncated else int(result["winners"][0]),
        "adjudicated_winner": (
            int(result["adjudicated_winners"][0]) if truncated else None
        ),
        "actions_after_intervention": actions,
    }


def force_completed_turn(
    environment: VectorEnv, root: int, plan: list[int]
) -> Mapping[str, np.ndarray] | None:
    result = None
    for index, action in enumerate(plan):
        result = environment.step(np.asarray([action], dtype=np.uint64))
        if environment.done()[0]:
            if index + 1 != len(plan):
                raise RuntimeError("completed root plan continued after game end")
            return result
    if result is None or int(environment.observe()["active_players"][0]) == root:
        raise RuntimeError("root candidate did not complete its turn")
    return None


def rollout(
    environment: VectorEnv,
    root: int,
    plan: list[int],
    policy: RoutedPolicy,
    rule_features: torch.Tensor,
) -> BranchOutcome:
    result = force_completed_turn(environment, root, plan)
    actions = len(plan)
    if result is not None:
        return outcome(result, actions)
    hybrid = new_hybrid(False)
    active = np.asarray([True], dtype=np.bool_)
    while not environment.done()[0]:
        observation = environment.observe()
        if int(observation["active_players"][0]) == root:
            chosen = hybrid.actions(environment, policy, rule_features, active)
        else:
            chosen = native_teacher_action(environment)
        result = environment.step(chosen)
        actions += 1
    return outcome(result, actions)


def score(branch: BranchOutcome, root: int) -> float | None:
    if branch["truncated"]:
        return None
    if branch["winner"] == 255:
        return 0.5
    return float(branch["winner"] == root)


def sample_first_regret(
    seed: int,
    root: int,
    policy: RoutedPolicy,
    maximum_round: int,
) -> dict[str, object]:
    environment = create_environment(seed)
    rule_features = encode_rules_batch(environment.rules_jsons(), torch.device("cpu"))
    hybrid = new_hybrid(True)
    active = np.asarray([True], dtype=np.bool_)
    observed_root_turns = 0
    last_result = None
    while not environment.done()[0]:
        observation = environment.observe()
        round_number = int(observation["rounds"][0])
        if round_number > maximum_round:
            break
        if int(observation["active_players"][0]) == root:
            before = len(hybrid.native_reply_records)
            chosen = hybrid.actions(environment, policy, rule_features, active)
            if len(hybrid.native_reply_records) > before:
                observed_root_turns += 1
                record = hybrid.native_reply_records[-1]
                regret = first_strict_regret(record)
                if regret is not None:
                    selected, teacher, gap = regret
                    plans = cast(list[list[int]], record["candidate_plans"])
                    outcomes = {
                        "hybrid": rollout(
                            environment.fork(np.asarray([0], dtype=np.uint64)),
                            root,
                            plans[selected],
                            policy,
                            rule_features,
                        ),
                        "native_best": rollout(
                            environment.fork(np.asarray([0], dtype=np.uint64)),
                            root,
                            plans[teacher],
                            policy,
                            rule_features,
                        ),
                    }
                    return {
                        "seed": seed,
                        "root_seat": root,
                        "round": round_number,
                        "observed_root_turns": observed_root_turns,
                        "hybrid_selected_index": selected,
                        "native_selected_index": teacher,
                        "native_reply_score_gap": gap,
                        "hybrid_plan_length": len(plans[selected]),
                        "native_plan_length": len(plans[teacher]),
                        "outcomes": outcomes,
                    }
        else:
            chosen = native_teacher_action(environment)
        last_result = environment.step(chosen)
    if not environment.done()[0]:
        stop_reason = "round_limit"
    elif bool(last_result["truncated"][0]):
        stop_reason = "action_limit"
    else:
        stop_reason = "terminal"
    return {
        "seed": seed,
        "root_seat": root,
        "observed_root_turns": observed_root_turns,
        "no_intervention_reason": stop_reason,
    }


def summarize(samples: list[dict[str, object]]) -> dict[str, object]:
    by_map: dict[int, list[float]] = defaultdict(list)
    seat_counts = {
        seat: {"native_better": 0, "hybrid_better": 0, "same": 0, "censored": 0}
        for seat in (0, 1)
    }
    sampled = 0
    for sample in samples:
        if "outcomes" not in sample:
            continue
        sampled += 1
        seat = cast(int, sample["root_seat"])
        branches = cast(dict[str, BranchOutcome], sample["outcomes"])
        hybrid = score(branches["hybrid"], seat)
        native = score(branches["native_best"], seat)
        if hybrid is None or native is None:
            seat_counts[seat]["censored"] += 1
            continue
        difference = native - hybrid
        by_map[cast(int, sample["seed"])].append(difference)
        if difference > 0:
            key = "native_better"
        elif difference < 0:
            key = "hybrid_better"
        else:
            key = "same"
        seat_counts[seat][key] += 1
    map_net = {seed: sum(values) for seed, values in by_map.items()}
    native_better = sum(value > 0 for value in map_net.values())
    hybrid_better = sum(value < 0 for value in map_net.values())
    same = len(map_net) - native_better - hybrid_better
    return {
        "games_examined": len(samples),
        "positions_with_strict_regret": sampled,
        "by_root_seat": seat_counts,
        "independent_maps_with_uncensored_pairs": len(map_net),
        "maps_with_both_seats_uncensored": sum(len(values) == 2 for values in by_map.values()),
        "map_net_outcome": {
            "native_better": native_better,
            "hybrid_better": hybrid_better,
            "same": same,
            "exact_two_sided_sign_test_p": paired_comparison_summary(
                native_better, hybrid_better, same
            )["exact_two_sided_sign_test_p"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=6370000)
    parser.add_argument("--maps", type=int, default=16)
    parser.add_argument("--maximum-round", type=int, default=32)
    arguments = parser.parse_args()
    checkpoint_sha256 = digest(arguments.checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("first-regret protocol requires the frozen routed-v6 checkpoint")
    policy, experts = load_routed_policy(arguments.checkpoint)
    samples = [
        sample_first_regret(seed, seat, policy, arguments.maximum_round)
        for seed in range(arguments.seed, arguments.seed + arguments.maps)
        for seat in (0, 1)
    ]
    print(
        json.dumps(
            {
                "kind": "exploratory_whole_turn_first_regret_intervention",
                "protocol": "benchmarks/protocols/2026-09-24-duel-first-regret-counterfactual-v1.json",
                "checkpoint_sha256": checkpoint_sha256,
                "selected_experts": experts,
                "seed": arguments.seed,
                "maps": arguments.maps,
                "maximum_round": arguments.maximum_round,
                "rule_features": RULE_FEATURES,
                "samples": samples,
                "summary": summarize(samples),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
