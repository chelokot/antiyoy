from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from antiyoy_rl.model import (
    action_distribution,
    concatenate_observations,
    encode_rules_batch,
)
from antiyoy_rl.slate_dataset import replay_slate_positions

from .build_bundle import digest
from .evaluate import load_policy
from .train_three_turn_plan import FIT_MAPS, FIT_SEED, SOURCE_SHA256, checked_dataset


FIT_SHA256 = "252f82f070c4fee9c707365d72c96766385afb4827276ce7030d837f8e7a7191"
UPDATES = 1024
BATCH_SIZE = 16
READOUTS = (0, 128, 256, 512, 1024)


@dataclass(frozen=True)
class Snapshot:
    seed: int
    seat: int
    observation: dict[str, np.ndarray]
    rules: str
    label: int


def first_records(
    dataset: dict[str, object], both_seats: bool = False
) -> list[dict[str, object]]:
    selected: dict[tuple[int, int], dict[str, object]] = {}
    for record in cast(list[dict[str, object]], dataset["records"]):
        seed = cast(int, record["seed"])
        seat = cast(int, record["seat"]) if both_seats else 0
        key = (seed, seat)
        if key not in selected:
            selected[key] = record
    expected = {
        (seed, seat)
        for seed in range(FIT_SEED, FIT_SEED + FIT_MAPS)
        for seat in (range(2) if both_seats else range(1))
    }
    if selected.keys() != expected:
        raise ValueError("fit data must contain every predeclared map and seat")
    return [selected[key] for key in sorted(selected)]


def fixed_snapshots(
    dataset: dict[str, object], both_seats: bool = False
) -> list[Snapshot]:
    selected = dict(dataset)
    selected["records"] = first_records(dataset, both_seats)
    snapshots = []
    for position in replay_slate_positions(selected):
        record = position.record
        plans = cast(list[list[int]], record["candidate_action_indices"])
        label = plans[cast(int, record["selected_index"])][0]
        observation = {
            key: np.asarray(value).copy()
            for key, value in position.root.observe().items()
        }
        legal = int(np.diff(observation["action_offsets"])[0])
        if label < 0 or label >= legal:
            raise ValueError(
                "teacher first action is not legal in the fixed observation"
            )
        snapshots.append(
            Snapshot(
                seed=position.seed,
                seat=cast(int, record["seat"]),
                observation=observation,
                rules=position.root.rules_json(),
                label=label,
            )
        )
    return snapshots


def batch_input(
    snapshots: list[Snapshot],
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    observations = concatenate_observations([item.observation for item in snapshots])
    rules = encode_rules_batch([item.rules for item in snapshots], torch.device("cpu"))
    labels = torch.tensor([item.label for item in snapshots], dtype=torch.long)
    return observations, rules, labels


def measure(
    model: torch.nn.Module, snapshots: list[Snapshot], update: int
) -> dict[str, object]:
    model.eval()
    correct = Counter[int]()
    totals = Counter[int]()
    total_nll = 0.0
    with torch.inference_mode():
        for start in range(0, len(snapshots), BATCH_SIZE):
            batch = snapshots[start : start + BATCH_SIZE]
            observations, rules, labels = batch_input(batch)
            logits, _ = model(observations, rules)
            distribution = action_distribution(logits, observations["action_offsets"])
            predictions = distribution.probs.argmax(dim=1)
            total_nll -= float(distribution.log_prob(labels).sum())
            for item, predicted in zip(batch, predictions.tolist(), strict=True):
                totals[item.seat] += 1
                correct[item.seat] += int(predicted == item.label)
    return {
        "update": update,
        "correct": sum(correct.values()),
        "total": len(snapshots),
        "mean_teacher_nll": total_nll / len(snapshots),
        "seats": {
            str(seat): {"correct": correct[seat], "total": totals[seat]}
            for seat in sorted(totals)
        },
    }


def run(
    fit_path: Path, source_path: Path, both_seats: bool = False
) -> dict[str, object]:
    if digest(fit_path) != FIT_SHA256 or digest(source_path) != SOURCE_SHA256:
        raise ValueError("diagnostic inputs disagree with the predeclared hashes")
    torch.set_num_threads(1)
    torch.manual_seed(FIT_SEED)
    dataset = checked_dataset(fit_path, FIT_SEED, FIT_MAPS)
    snapshots = fixed_snapshots(dataset, both_seats)
    model, config = load_policy(
        source_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0)
    shuffle = random.Random(FIT_SEED)
    readouts = [measure(model, snapshots, 0)]
    started = time.perf_counter()
    update = 0
    while update < UPDATES:
        order = list(range(len(snapshots)))
        shuffle.shuffle(order)
        for start in range(0, len(order), BATCH_SIZE):
            batch = [snapshots[index] for index in order[start : start + BATCH_SIZE]]
            observations, rules, labels = batch_input(batch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(observations, rules)
            distribution = action_distribution(logits, observations["action_offsets"])
            loss = -distribution.log_prob(labels).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update += 1
            if update in READOUTS:
                readouts.append(measure(model, snapshots, update))
    return {
        "kind": (
            "fixed_snapshot_teacher_action_memorization_both_seats"
            if both_seats
            else "fixed_snapshot_teacher_action_memorization_diagnostic"
        ),
        "fit_sha256": digest(fit_path),
        "source_sha256": digest(source_path),
        "selected_expert": config["selected_expert"],
        "maps": len(snapshots),
        "readouts": readouts,
        "training_seconds": time.perf_counter() - started,
        "qualification": "Same-training-snapshot memorization only; not generalization, autonomous strength, Elo, or a deployable model",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--both-seats", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(arguments.fit, arguments.source, arguments.both_seats), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
