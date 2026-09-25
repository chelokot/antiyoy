from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import resource
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

from antiyoy_rl.model import (
    action_distribution,
    concatenate_observations,
    encode_rules_batch,
)
from antiyoy_rl.routed import RoutedPolicy

from .audit_duel_first_regret import (
    CHECKPOINT_SHA256,
    create_environment,
    load_routed_policy,
    outcome,
)
from .build_bundle import digest
from .collect_duel_episode_preference_window import PROTOCOL, WINDOWS
from .evaluate import load_policy_checkpoint


BETA = 0.02
EPOCHS = 2
LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.01
CHUNK_SIZE = 16


@dataclass(frozen=True)
class DecisionTrace:
    observations: tuple[dict[str, np.ndarray], ...]
    rules_json: str
    actions: tuple[int, ...]


@dataclass(frozen=True)
class PreferencePair:
    seat: int
    preferred: dict[str, object]
    dispreferred: dict[str, object]


def checked_dataset(path: Path, window: str) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        dataset = cast(dict[str, object], json.load(source))
    first_seed, maps, _, _ = WINDOWS[window]
    records = cast(list[dict[str, object]], dataset["records"])
    keys = [(record["seed"], record["teacher_seat"]) for record in records]
    expected = {
        (seed, seat)
        for seed in range(first_seed, first_seed + maps)
        for seat in (None, 0, 1)
    }
    if (
        dataset["kind"] != "terminal_whole_episode_preference_window"
        or dataset["protocol"] != PROTOCOL
        or dataset["window"] != window
        or dataset["first_seed"] != first_seed
        or dataset["maps"] != maps
        or dataset["source_sha256"] != CHECKPOINT_SHA256
        or len(keys) != len(expected)
        or set(keys) != expected
    ):
        raise ValueError("episode-preference dataset disagrees with the protocol")
    return dataset


def informative_pairs(dataset: dict[str, object]) -> dict[int, list[PreferencePair]]:
    records = cast(list[dict[str, object]], dataset["records"])
    indexed = {
        (cast(int, record["seed"]), record["teacher_seat"]): record
        for record in records
    }
    pairs: dict[int, list[PreferencePair]] = defaultdict(list)
    for seed in range(
        cast(int, dataset["first_seed"]),
        cast(int, dataset["first_seed"]) + cast(int, dataset["maps"]),
    ):
        source = indexed[seed, None]
        source_outcome = cast(dict[str, object], source["outcome"])
        for seat in (0, 1):
            teacher = indexed[seed, seat]
            teacher_outcome = cast(dict[str, object], teacher["outcome"])
            if not source_outcome["terminal"] or not teacher_outcome["terminal"]:
                continue
            source_wins = source_outcome["winner"] == seat
            teacher_wins = teacher_outcome["winner"] == seat
            if source_wins != teacher_wins:
                preferred, dispreferred = (
                    (teacher, source) if teacher_wins else (source, teacher)
                )
                pairs[seed].append(PreferencePair(seat, preferred, dispreferred))
    return pairs


def replay_decisions(record: dict[str, object], seat: int) -> DecisionTrace:
    environment = create_environment(cast(int, record["seed"]))
    rules_json = environment.rules_jsons()[0]
    actors = cast(list[int], record["actors"])
    actions = cast(list[int], record["actions"])
    if not actions or len(actors) != len(actions):
        raise ValueError("episode trace has no aligned actions")
    observations = []
    selected = []
    result = None
    for actor, action in zip(actors, actions, strict=True):
        observed = environment.observe()
        legal = int(np.diff(observed["action_offsets"])[0])
        if int(observed["active_players"][0]) != actor or not 0 <= action < legal:
            raise ValueError("episode trace disagrees with its reconstructed state")
        if actor == seat:
            observations.append(
                {key: np.asarray(value).copy() for key, value in observed.items()}
            )
            selected.append(action)
        result = environment.step(np.asarray([action], dtype=np.uint64))
    if result is None or outcome(result, len(actions)) != record["outcome"]:
        raise ValueError("episode replay final outcome differs")
    if not selected:
        raise ValueError("root seat has no decision in its episode")
    return DecisionTrace(tuple(observations), rules_json, tuple(selected))


def decision_log_probabilities(
    policy: RoutedPolicy, trace: DecisionTrace, start: int, end: int
) -> Tensor:
    observations = concatenate_observations(trace.observations[start:end])
    rules = encode_rules_batch(
        [trace.rules_json] * (end - start), torch.device("cpu")
    )
    logits, _ = policy(observations, rules)
    distribution = action_distribution(logits, observations["action_offsets"])
    actions = torch.as_tensor(trace.actions[start:end], dtype=torch.long)
    return distribution.log_prob(actions)


def trajectory_log_ratio(
    student: RoutedPolicy, reference: RoutedPolicy, trace: DecisionTrace
) -> float:
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(trace.actions), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(trace.actions))
            student_log = decision_log_probabilities(student, trace, start, end)
            reference_log = decision_log_probabilities(reference, trace, start, end)
            total += float((student_log - reference_log).sum())
    return total


def preference_loss(margin: Tensor) -> Tensor:
    return functional.softplus(-BETA * margin)


def accumulate_trajectory_gradient(
    student: RoutedPolicy, trace: DecisionTrace, weight: float
) -> None:
    for start in range(0, len(trace.actions), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(trace.actions))
        log_probabilities = decision_log_probabilities(student, trace, start, end)
        (weight * log_probabilities.sum()).backward()


def accumulate_pair_gradient(
    student: RoutedPolicy,
    reference: RoutedPolicy,
    preferred: DecisionTrace,
    dispreferred: DecisionTrace,
    pair_count: int,
) -> tuple[float, float]:
    margin = trajectory_log_ratio(student, reference, preferred)
    margin -= trajectory_log_ratio(student, reference, dispreferred)
    margin_tensor = torch.tensor(margin)
    coefficient = BETA * float(torch.sigmoid(-BETA * margin_tensor))
    scale = coefficient / pair_count
    accumulate_trajectory_gradient(student, preferred, -scale)
    accumulate_trajectory_gradient(student, dispreferred, scale)
    return float(preference_loss(margin_tensor)), margin


def fit_map(
    student: RoutedPolicy,
    reference: RoutedPolicy,
    pairs: list[PreferencePair],
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    losses = []
    margins = []
    for pair in pairs:
        preferred = replay_decisions(pair.preferred, pair.seat)
        dispreferred = replay_decisions(pair.dispreferred, pair.seat)
        loss, margin = accumulate_pair_gradient(
            student, reference, preferred, dispreferred, len(pairs)
        )
        losses.append(loss)
        margins.append(margin)
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    return float(np.mean(losses)), float(np.mean(margins))


def train(dataset: dict[str, object], student: RoutedPolicy, reference: RoutedPolicy) -> list[dict[str, float | int]]:
    pairs = informative_pairs(dataset)
    parameters = [
        parameter
        for model in student.models.values()
        for name, parameter in model.named_parameters()
        if not name.startswith("value_head.")
    ]
    for model in student.models.values():
        for parameter in model.value_head.parameters():
            parameter.requires_grad_(False)
    for model in reference.models.values():
        model.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    order = sorted(pairs)
    randomizer = random.Random(WINDOWS["fit"][0])
    history = []
    for epoch in range(EPOCHS):
        randomizer.shuffle(order)
        losses = []
        margins = []
        for seed in order:
            loss, margin = fit_map(
                student, reference, pairs[seed], optimizer, parameters
            )
            losses.append(loss)
            margins.append(margin)
        history.append(
            {
                "epoch": epoch + 1,
                "map_updates": len(order),
                "mean_map_loss": float(np.mean(losses)),
                "mean_map_log_ratio_margin": float(np.mean(margins)),
            }
        )
    return history


def run(source_path: Path, fit_path: Path, output_path: Path) -> dict[str, object]:
    if digest(source_path) != CHECKPOINT_SHA256:
        raise ValueError("frozen source checkpoint disagrees with the protocol")
    dataset = checked_dataset(fit_path, "fit")
    summary = cast(dict[str, object], dataset["summary"])
    if not summary["data_gate_passed"]:
        raise ValueError("fit data availability gate failed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    student, student_experts = load_routed_policy(source_path)
    reference, reference_experts = load_routed_policy(source_path)
    if student_experts != reference_experts:
        raise ValueError("student and reference routes differ")
    started = time.perf_counter()
    history = train(dataset, student, reference)
    elapsed = time.perf_counter() - started
    peak_resident_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if elapsed > 3600 or peak_resident_bytes > 4 * 1024**3:
        raise ValueError("predeclared fit resource gate failed")
    checkpoint = copy.deepcopy(load_policy_checkpoint(source_path, torch.device("cpu")))
    expert_states = cast(dict[str, dict[str, Tensor]], checkpoint["experts"])
    for name, model in student.models.items():
        expert_states[name] = model.state_dict()
    torch.save(checkpoint, output_path)
    return {
        "kind": "terminal_whole_episode_preference_fit",
        "protocol": PROTOCOL,
        "source_sha256": CHECKPOINT_SHA256,
        "fit_dataset_sha256": digest(fit_path),
        "student_checkpoint_sha256": digest(output_path),
        "informative_pairs": summary["informative_pairs"],
        "map_updates_per_epoch": history[0]["map_updates"],
        "history": history,
        "training_elapsed_seconds": elapsed,
        "peak_resident_bytes_before_checkpoint_save": peak_resident_bytes,
        "qualification": "Fit only, no offline validation, complete-game strength, Elo or deployment",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("fit_dataset", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.source, arguments.fit_dataset, arguments.output), sort_keys=True))


if __name__ == "__main__":
    main()
