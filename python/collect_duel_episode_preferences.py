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
    CHECKPOINT_SHA256,
    create_environment,
    load_routed_policy,
    native_teacher_action,
    outcome,
)
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .build_bundle import digest


PROTOCOL = (
    "benchmarks/protocols/2026-09-25-duel-terminal-episode-preference-feasibility-v1.json"
)
SEED_FIRST = 6560000
MAPS = 16


def play(seed: int, teacher_seat: int | None, source: RoutedPolicy) -> dict[str, object]:
    environment = create_environment(seed)
    rules = encode_rules_batch(environment.rules_jsons(), torch.device("cpu"))
    actors: list[int] = []
    actions: list[int] = []
    result = None
    while not environment.done()[0]:
        observation = environment.observe()
        actor = int(observation["active_players"][0])
        selected = (
            native_teacher_action(environment, FOLLOWUP_NODES, True)
            if actor == teacher_seat
            else source.actions(observation, rules)
        )
        action = int(selected[0])
        legal = int(np.diff(observation["action_offsets"])[0])
        if not 0 <= action < legal:
            raise ValueError("controller selected an illegal action index")
        actors.append(actor)
        actions.append(action)
        result = environment.step(np.asarray([action], dtype=np.uint64))
    if result is None:
        raise ValueError("game ended before any action")
    return {
        "seed": seed,
        "teacher_seat": teacher_seat,
        "actors": actors,
        "actions": actions,
        "outcome": outcome(result, len(actions)),
    }


def verify_trace(record: dict[str, object]) -> None:
    environment = create_environment(cast(int, record["seed"]))
    actors = cast(list[int], record["actors"])
    actions = cast(list[int], record["actions"])
    if not actions or len(actors) != len(actions):
        raise ValueError("trace has no aligned actions")
    result = None
    for index, (actor, action) in enumerate(zip(actors, actions, strict=True)):
        observation = environment.observe()
        legal = int(np.diff(observation["action_offsets"])[0])
        if int(observation["active_players"][0]) != actor or not 0 <= action < legal:
            raise ValueError("trace action disagrees with reconstructed state")
        result = environment.step(np.asarray([action], dtype=np.uint64))
        if environment.done()[0] != (index + 1 == len(actions)):
            raise ValueError("trace ended at a different action")
    if result is None or outcome(result, len(actions)) != record["outcome"]:
        raise ValueError("replayed final outcome differs")


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    indexed = {
        (cast(int, record["seed"]), record["teacher_seat"]): record
        for record in records
    }
    if len(records) != MAPS * 3 or len(indexed) != MAPS * 3:
        raise ValueError("expected exactly one source and two teacher games per map")
    by_seat = []
    for seat in (0, 1):
        counts = {
            "teacher_preferred": 0,
            "source_preferred": 0,
            "same_terminal_outcome": 0,
            "censored": 0,
        }
        for seed in range(SEED_FIRST, SEED_FIRST + MAPS):
            reference = cast(dict[str, object], indexed[seed, None]["outcome"])
            teacher = cast(dict[str, object], indexed[seed, seat]["outcome"])
            if not reference["terminal"] or not teacher["terminal"]:
                counts["censored"] += 1
                continue
            reference_wins = reference["winner"] == seat
            teacher_wins = teacher["winner"] == seat
            if teacher_wins and not reference_wins:
                counts["teacher_preferred"] += 1
            elif reference_wins and not teacher_wins:
                counts["source_preferred"] += 1
            else:
                counts["same_terminal_outcome"] += 1
        by_seat.append(counts)
    informative = sum(
        item["teacher_preferred"] + item["source_preferred"] for item in by_seat
    )
    censored = sum(item["censored"] for item in by_seat)
    terminal_games = sum(
        bool(cast(dict[str, object], record["outcome"])["terminal"])
        for record in records
    )
    return {
        "maps": MAPS,
        "games": len(records),
        "terminal_games": terminal_games,
        "pairs": MAPS * 2,
        "informative_pairs": informative,
        "censored_pairs": censored,
        "by_teacher_seat": by_seat,
        "data_gate_passed": (
            terminal_games == MAPS * 3
            and censored == 0
            and informative >= 8
            and all(item["teacher_preferred"] + item["source_preferred"] >= 2 for item in by_seat)
        ),
    }


def collect(source_path: Path) -> dict[str, object]:
    if digest(source_path) != CHECKPOINT_SHA256:
        raise ValueError("frozen source checkpoint disagrees with the protocol")
    torch.set_num_threads(1)
    source, experts = load_routed_policy(source_path)
    started = time.perf_counter()
    records = []
    with torch.inference_mode():
        for seed in range(SEED_FIRST, SEED_FIRST + MAPS):
            for teacher_seat in (None, 0, 1):
                record = play(seed, teacher_seat, source)
                verify_trace(record)
                records.append(record)
    elapsed = time.perf_counter() - started
    summary = summarize(records)
    summary["collection_elapsed_seconds"] = elapsed
    summary["advance_gate_passed"] = bool(summary["data_gate_passed"]) and elapsed <= 600
    return {
        "kind": "terminal_whole_episode_preference_feasibility",
        "protocol": PROTOCOL,
        "source_sha256": CHECKPOINT_SHA256,
        "source_experts": experts,
        "records": records,
        "summary": summary,
        "qualification": "Feasibility only, no student training, terminal credit for individual actions, strength or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(collect(arguments.source), sort_keys=True))


if __name__ == "__main__":
    main()
