from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl import ProceduralConfig, VectorEnv
from antiyoy_rl.model import domain_key, encode_rules_batch
from antiyoy_rl.routed import RoutedPolicy

try:
    from .build_bundle import digest
    from .evaluate import (
        instantiate_policy,
        load_policy_checkpoint,
        select_policy_state,
    )
except ImportError:
    from build_bundle import digest
    from evaluate import instantiate_policy, load_policy_checkpoint, select_policy_state


@dataclass(frozen=True)
class AuditConfig:
    checkpoint: Path
    maps: int = 16
    seed: int = 6240000
    target_round: int = 8
    horizon: int | None = None
    profile: str = "classic_generic_2022"
    width: int = 11
    height: int = 9
    action_limit: int = 1000
    route_generator: str = "procedural_v1"
    search_nodes: int = 256
    reply_search_nodes: int = 64
    reply_slate_size: int = 8
    search_beam_width: int = 32
    search_branch_width: int = 48
    search_maximum_actions_per_turn: int = 24
    device: str = "cpu"


@dataclass(frozen=True)
class BranchOutcome:
    winner: int | None
    terminal: bool
    truncated: bool
    censored: bool
    actions: int
    first_turn_actions: tuple[int, ...]


def outcome_score(outcome: BranchOutcome, seat: int) -> int | None:
    if outcome.censored:
        return None
    if outcome.winner is None:
        return 1
    return 2 if outcome.winner == seat else 0


def rollout_branch(
    environment: VectorEnv,
    index: int,
    policy: RoutedPolicy,
    rules: torch.Tensor,
    config: AuditConfig,
    teacher_turn: bool,
    source_outcome: BranchOutcome | None = None,
) -> BranchOutcome:
    branch = environment.fork(np.array([index], dtype=np.uint64))
    branch_rules = rules[index : index + 1]
    root_player = int(branch.observe()["active_players"][0])
    first_turn = True
    first_turn_actions: list[int] = []
    actions = 0
    while True:
        observation = branch.observe()
        if teacher_turn and first_turn:
            selected = branch.reply_search_actions(
                node_budget=config.search_nodes,
                reply_nodes=config.reply_search_nodes,
                slate_size=config.reply_slate_size,
                beam_width=config.search_beam_width,
                branch_width=config.search_branch_width,
                maximum_actions_per_turn=config.search_maximum_actions_per_turn,
            )
        else:
            selected = policy.actions(observation, branch_rules)
        action = int(selected[0])
        if first_turn:
            first_turn_actions.append(action)
        result = branch.step(np.array([action], dtype=np.uint64))
        actions += 1
        truncated = bool(result["truncated"][0])
        terminal = bool(result["terminal"][0])
        if terminal or truncated:
            winner_key = "adjudicated_winners" if truncated else "winners"
            raw_winner = int(result[winner_key][0])
            return BranchOutcome(
                winner=None if raw_winner == 255 else raw_winner,
                terminal=terminal,
                truncated=truncated,
                censored=False,
                actions=actions,
                first_turn_actions=tuple(first_turn_actions),
            )
        if config.horizon is not None and actions >= config.horizon:
            return BranchOutcome(
                winner=None,
                terminal=False,
                truncated=False,
                censored=True,
                actions=actions,
                first_turn_actions=tuple(first_turn_actions),
            )
        if first_turn and int(branch.observe()["active_players"][0]) != root_player:
            first_turn = False
            if (
                teacher_turn
                and source_outcome is not None
                and tuple(first_turn_actions) == source_outcome.first_turn_actions
            ):
                return source_outcome


def audit(config: AuditConfig) -> dict[str, object]:
    if config.maps < 1 or config.width < 3 or config.height < 3:
        raise ValueError("audit requires positive maps and valid dimensions")
    if config.target_round < 0 or config.action_limit < 1:
        raise ValueError("audit round and action limit are invalid")
    if config.horizon is not None and config.horizon < 1:
        raise ValueError("audit horizon must be positive")
    if config.reply_search_nodes < 2 or config.reply_slate_size < 1:
        raise ValueError("audit reply search budget is invalid")
    started = time.perf_counter()
    device = torch.device(config.device)
    checkpoint = load_policy_checkpoint(config.checkpoint, device)
    checkpoint_config = dict(checkpoint["config"])
    if checkpoint_config["fog"]:
        raise ValueError("turn intervention audit requires full information")
    generator = ProceduralConfig(
        width=config.width,
        height=config.height,
        players=2,
        seed=config.seed,
        schema_version=2,
    )
    generator_config = json.loads(generator.to_json())
    descriptor = {
        key: generator_config[key]
        for key in (
            "width",
            "height",
            "players",
            "land_density_per_million",
            "starting_province_size",
            "starting_money",
            "tree_density_per_million",
            "neutral_tower_density_per_million",
            "neutral_capital_density_per_million",
            "grave_density_per_million",
        )
    }
    descriptor.update(
        action_limit=config.action_limit,
        fog=False,
        diplomacy=checkpoint_config.get("diplomacy", False),
        initial_relation=checkpoint_config.get("initial_relation", "neutral"),
    )
    route_domain = domain_key(config.route_generator, descriptor)
    models = {}
    experts = []
    for seat in range(2):
        state, selected_config = select_policy_state(
            checkpoint,
            config.profile,
            config.route_generator,
            2,
            seat,
            route_domain,
        )
        expert = str(selected_config["selected_expert"])
        if expert not in models:
            models[expert] = instantiate_policy(state, selected_config, device)
        experts.append(expert)
    policy = RoutedPolicy(models, experts)
    environment = VectorEnv.procedural(
        config.maps,
        generator,
        action_limit=config.action_limit,
        profile=config.profile,
        fog=False,
        diplomacy=bool(checkpoint_config.get("diplomacy", False)),
        initial_relation=str(checkpoint_config.get("initial_relation", "neutral")),
    )
    for index in range(config.maps):
        environment.reset(index, config.seed + index)
    rules = encode_rules_batch(environment.rules_jsons(), device)
    map_indices = np.arange(config.maps, dtype=np.int64)
    sampled = np.zeros((config.maps, 2), dtype=np.bool_)
    records: list[dict[str, object]] = []
    while len(map_indices) > 0:
        observation = environment.observe()
        for local_index, map_index in enumerate(map_indices):
            seat = int(observation["active_players"][local_index])
            if int(observation["rounds"][local_index]) < config.target_round:
                continue
            if sampled[map_index, seat]:
                continue
            source = rollout_branch(
                environment, local_index, policy, rules, config, teacher_turn=False
            )
            teacher = rollout_branch(
                environment,
                local_index,
                policy,
                rules,
                config,
                teacher_turn=True,
                source_outcome=source,
            )
            source_score = outcome_score(source, seat)
            teacher_score = outcome_score(teacher, seat)
            records.append(
                {
                    "seed": config.seed + int(map_index),
                    "seat": seat,
                    "round": int(observation["rounds"][local_index]),
                    "same_first_turn": (
                        source.first_turn_actions == teacher.first_turn_actions
                    ),
                    "source": asdict(source),
                    "teacher": asdict(teacher),
                    "outcome_delta": (
                        None
                        if source_score is None or teacher_score is None
                        else teacher_score - source_score
                    ),
                }
            )
            sampled[map_index, seat] = True
        selected = policy.actions(observation, rules)
        result = environment.step(selected)
        remaining = np.flatnonzero(
            ~np.logical_or(result["terminal"], result["truncated"])
        )
        if len(remaining) == 0:
            break
        environment = environment.fork(remaining.astype(np.uint64))
        rules = rules[remaining]
        map_indices = map_indices[remaining]
    deltas = [record["outcome_delta"] for record in records]
    completed = [delta for delta in deltas if delta is not None]
    map_deltas: dict[int, int] = {}
    for record in records:
        if record["outcome_delta"] is not None:
            seed = int(record["seed"])
            map_deltas[seed] = map_deltas.get(seed, 0) + int(record["outcome_delta"])
    return {
        "kind": "reply_search_whole_turn_intervention_audit",
        "schema_version": 1,
        "source_checkpoint_sha256": digest(config.checkpoint),
        "generator": generator_config,
        "profile": config.profile,
        "route_generator": config.route_generator,
        "selected_experts": experts,
        "seed_first": config.seed,
        "maps": config.maps,
        "target_round": config.target_round,
        "horizon": config.horizon,
        "positions": len(records),
        "same_first_turn": sum(record["same_first_turn"] for record in records),
        "better": sum(delta > 0 for delta in completed),
        "worse": sum(delta < 0 for delta in completed),
        "same_outcome": sum(delta == 0 for delta in completed),
        "censored": len(records) - len(completed),
        "independent_maps_better": sum(delta > 0 for delta in map_deltas.values()),
        "independent_maps_worse": sum(delta < 0 for delta in map_deltas.values()),
        "independent_maps_same": sum(delta == 0 for delta in map_deltas.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--maps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=6240000)
    parser.add_argument("--target-round", type=int, default=8)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--profile", default="classic_generic_2022")
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--action-limit", type=int, default=1000)
    parser.add_argument("--route-generator", default="procedural_v1")
    parser.add_argument("--search-nodes", type=int, default=256)
    parser.add_argument("--reply-search-nodes", type=int, default=64)
    parser.add_argument("--reply-slate-size", type=int, default=8)
    parser.add_argument("--search-beam-width", type=int, default=32)
    parser.add_argument("--search-branch-width", type=int, default=48)
    parser.add_argument("--search-maximum-actions-per-turn", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    print(json.dumps(audit(AuditConfig(**vars(parser.parse_args()))), sort_keys=True))


if __name__ == "__main__":
    main()
