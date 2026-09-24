from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch

from antiyoy_rl.model import encode_rules_batch
from antiyoy_rl.routed import RoutedPolicy

from .audit_duel_first_regret import (
    create_environment,
    load_routed_policy,
    native_teacher_action,
)
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .build_bundle import digest
from .evaluate import selective_reply_search_actions


PROTOCOL = "benchmarks/protocols/2026-09-24-duel-selective-replan-latency-v1.json"
GATE_SEED = 6532000
GATE_MAPS = 16
SOURCE_SHA256 = "68549a5af87e4d2d065a164e39bc515c2c1d48276f437494d757e25c3ef28867"
STUDENT_SHA256 = "1d6e6530bb2f16d273910fdeb4b2de491d2e115017468eb3f31d3aea43f64fe4"
CONTROLLERS = ("source", "selective", "teacher")


def timing(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("latency summary requires sampled actions and turns")
    return {
        "samples": len(values),
        "median_milliseconds": float(np.median(values)),
        "p95_milliseconds": float(np.percentile(values, 95)),
    }


def play_rollin(
    seed: int,
    root_seat: int,
    turns_per_game: int,
    source: RoutedPolicy,
    student: RoutedPolicy,
) -> dict[str, object]:
    environment = create_environment(seed)
    rules = encode_rules_batch(environment.rules_jsons(), torch.device("cpu"))
    action_times = {controller: {"cpu": [], "wall": []} for controller in CONTROLLERS}
    turn_times = {controller: {"cpu": [], "wall": []} for controller in CONTROLLERS}
    current_turn = {controller: {"cpu": 0.0, "wall": 0.0} for controller in CONTROLLERS}
    completed_turns = 0
    in_source_turn = False
    decisions = 0
    queries = 0
    overrides = 0
    steps = 0
    last_result = None
    while not environment.done()[0] and completed_turns < turns_per_game:
        observation = environment.observe()
        active = int(observation["active_players"][0])
        if in_source_turn and active != root_seat:
            for controller in CONTROLLERS:
                for clock in ("cpu", "wall"):
                    turn_times[controller][clock].append(
                        current_turn[controller][clock]
                    )
                    current_turn[controller][clock] = 0.0
            completed_turns += 1
            in_source_turn = False
            if completed_turns == turns_per_game:
                break
        if active == root_seat:
            in_source_turn = True
            selective_fork = environment.fork(np.asarray([0], dtype=np.uint64))
            teacher_fork = environment.fork(np.asarray([0], dtype=np.uint64))
            for fork in (selective_fork, teacher_fork):
                fork_observation = fork.observe()
                for name, values in observation.items():
                    np.testing.assert_array_equal(values, fork_observation[name])
            actions: dict[str, np.ndarray] = {}
            for offset in range(len(CONTROLLERS)):
                controller = CONTROLLERS[
                    (seed + root_seat + decisions + offset) % len(CONTROLLERS)
                ]
                cpu_started = time.process_time_ns()
                wall_started = time.perf_counter_ns()
                if controller == "source":
                    actions[controller] = source.actions(observation, rules)
                elif controller == "teacher":
                    actions[controller] = native_teacher_action(
                        teacher_fork, FOLLOWUP_NODES, replan_each_action=True
                    )
                else:
                    student_actions = student.actions(observation, rules)
                    source_actions = source.actions(observation, rules)
                    actions[controller], searched = selective_reply_search_actions(
                        selective_fork,
                        student_actions,
                        source_actions,
                        np.asarray([True]),
                        256,
                        64,
                        8,
                        32,
                        48,
                        24,
                        FOLLOWUP_NODES,
                    )
                    queries += searched
                    overrides += int(actions[controller][0] != source_actions[0])
                cpu_milliseconds = (time.process_time_ns() - cpu_started) / 1e6
                wall_milliseconds = (time.perf_counter_ns() - wall_started) / 1e6
                action_times[controller]["cpu"].append(cpu_milliseconds)
                action_times[controller]["wall"].append(wall_milliseconds)
                current_turn[controller]["cpu"] += cpu_milliseconds
                current_turn[controller]["wall"] += wall_milliseconds
            legal_actions = int(np.diff(observation["action_offsets"])[0])
            if any(
                not 0 <= int(actions[controller][0]) < legal_actions
                for controller in CONTROLLERS
            ):
                raise ValueError("matched selector returned an illegal action")
            selected = actions["source"]
            decisions += 1
        else:
            selected = native_teacher_action(
                environment, FOLLOWUP_NODES, replan_each_action=True
            )
        last_result = environment.step(selected)
        steps += 1
    if in_source_turn:
        for controller in CONTROLLERS:
            for clock in ("cpu", "wall"):
                turn_times[controller][clock].append(current_turn[controller][clock])
        completed_turns += 1
    if last_result is None:
        raise ValueError("runtime roll-in ended without an action")
    return {
        "seed": seed,
        "root_seat": root_seat,
        "sampled_turns": completed_turns,
        "sampled_decisions": decisions,
        "teacher_queries": queries,
        "final_actions_different_from_source": overrides,
        "rollin_steps": steps,
        "rollin_terminal": bool(environment.done()[0]),
        "rollin_truncated": bool(last_result["truncated"][0]),
        "action_milliseconds": action_times,
        "source_turn_milliseconds": turn_times,
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    def summaries(selected: list[dict[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for controller in CONTROLLERS:
            result[controller] = {
                kind: {
                    clock: timing(
                        [
                            value
                            for record in selected
                            for value in cast(
                                dict[str, dict[str, list[float]]], record[kind]
                            )[controller][clock]
                        ]
                    )
                    for clock in ("cpu", "wall")
                }
                for kind in ("action_milliseconds", "source_turn_milliseconds")
            }
        return result

    decisions = sum(cast(int, record["sampled_decisions"]) for record in records)
    queries = sum(cast(int, record["teacher_queries"]) for record in records)
    seeds = {record["seed"] for record in records}
    pairs = {(record["seed"], record["root_seat"]) for record in records}
    if (
        decisions < 1
        or len(records) != 2 * len(seeds)
        or len(pairs) != len(records)
        or any((seed, seat) not in pairs for seed in seeds for seat in (0, 1))
    ):
        raise ValueError("runtime sample lacks both seats or decisions")
    if any(cast(int, record["sampled_turns"]) < 1 for record in records):
        raise ValueError("runtime sample lacks a completed source turn")
    by_controller = summaries(records)
    selective = cast(dict[str, object], by_controller["selective"])
    teacher = cast(dict[str, object], by_controller["teacher"])
    selective_cpu = cast(dict[str, dict[str, float]], selective["action_milliseconds"])[
        "cpu"
    ]
    teacher_cpu = cast(dict[str, dict[str, float]], teacher["action_milliseconds"])[
        "cpu"
    ]
    gate = {
        "zero_censored_rollins": not any(
            record["rollin_truncated"] for record in records
        ),
        "teacher_query_fraction_at_most_half": queries / decisions <= 0.5,
        "selective_median_cpu_below_teacher": selective_cpu["median_milliseconds"]
        < teacher_cpu["median_milliseconds"],
        "selective_p95_cpu_at_most_teacher": selective_cpu["p95_milliseconds"]
        <= teacher_cpu["p95_milliseconds"],
        "selective_p95_cpu_below_250ms": selective_cpu["p95_milliseconds"] < 250,
    }
    return {
        "sampled_decisions": decisions,
        "sampled_source_turns": sum(
            cast(int, record["sampled_turns"]) for record in records
        ),
        "teacher_queries": queries,
        "teacher_query_fraction": queries / decisions,
        "final_actions_different_from_source": sum(
            cast(int, record["final_actions_different_from_source"])
            for record in records
        ),
        "censored_rollins": sum(bool(record["rollin_truncated"]) for record in records),
        "matched_timing": by_controller,
        "by_root_seat": {
            str(seat): summaries(
                [record for record in records if record["root_seat"] == seat]
            )
            for seat in (0, 1)
        },
        "gate": gate,
        "runtime_gate_passed": all(gate.values()),
    }


def audit(
    first_seed: int,
    maps: int,
    turns_per_game: int,
    source_path: Path,
    student_path: Path,
) -> dict[str, object]:
    if maps < 1 or turns_per_game < 1:
        raise ValueError("runtime audit requires maps and candidate turns")
    if first_seed == GATE_SEED and (maps != GATE_MAPS or turns_per_game != 12):
        raise ValueError("reserved runtime window requires 16 maps and 12 turns")
    hashes = {"source": digest(source_path), "student": digest(student_path)}
    if hashes != {"source": SOURCE_SHA256, "student": STUDENT_SHA256}:
        raise ValueError("runtime audit checkpoints differ from the frozen controller")
    torch.set_num_threads(1)
    source, source_experts = load_routed_policy(source_path)
    student, student_experts = load_routed_policy(student_path)
    records = [
        play_rollin(seed, seat, turns_per_game, source, student)
        for seed in range(first_seed, first_seed + maps)
        for seat in (0, 1)
    ]
    summary = summarize(records)
    gate_window = first_seed == GATE_SEED
    if not gate_window:
        summary["runtime_gate_passed"] = None
    return {
        "kind": "selective_replan_matched_source_state_cpu_latency",
        "protocol": PROTOCOL,
        "first_seed": first_seed,
        "independent_maps": maps,
        "maximum_source_turns_per_game": turns_per_game,
        "checkpoint_hashes": hashes,
        "source_experts": source_experts,
        "student_experts": student_experts,
        "records": records,
        "summary": summary,
        "gate_window": gate_window,
        "qualification": "Matched one-core source-visited selector cost, not candidate on-policy or browser latency, game strength, Elo or standalone distillation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--maps", type=int, required=True)
    parser.add_argument("--turns-per-game", type=int, default=12)
    parser.add_argument("--source", dest="source_path", type=Path, required=True)
    parser.add_argument("--student", dest="student_path", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(audit(**vars(arguments)), sort_keys=True))


if __name__ == "__main__":
    main()
