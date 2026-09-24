from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from antiyoy_rl.model import action_distribution, encode_rules
from antiyoy_rl.slate_dataset import replay_slate_positions

from .audit_duel_markov_root_fidelity import legal_count_bin
from .build_bundle import digest
from .evaluate import load_policy
from .train_three_turn_plan import FIT_MAPS, FIT_SEED, SOURCE_SHA256, checked_dataset


FIT_SHA256 = "252f82f070c4fee9c707365d72c96766385afb4827276ce7030d837f8e7a7191"
PROTOCOL = "benchmarks/protocols/2026-09-25-duel-source-uncertainty-routing-feasibility-v1.json"


@dataclass(frozen=True)
class Position:
    seed: int
    seat: int
    legal_count: int
    mismatch: bool
    top_probability: float
    normalized_entropy: float
    margin: float


def confidence(probabilities: np.ndarray) -> tuple[float, float, float]:
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("action probabilities must be a nonempty vector")
    ordered = np.sort(probabilities)
    top = float(ordered[-1])
    if probabilities.size == 1:
        return top, 0.0, 1.0
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(probabilities.size)
    return top, entropy, top - float(ordered[-2])


def subgroup(
    positions: list[Position], selected: set[int], indices: list[int]
) -> dict[str, int | float | None]:
    disagreements = sum(positions[index].mismatch for index in indices)
    queried = sum(index in selected for index in indices)
    captured = sum(index in selected and positions[index].mismatch for index in indices)
    skipped = len(indices) - queried
    return {
        "positions": len(indices),
        "teacher_disagreements": disagreements,
        "queried": queried,
        "captured_disagreements": captured,
        "disagreement_capture_fraction": (
            captured / disagreements if disagreements else None
        ),
        "skipped": skipped,
        "skipped_disagreements": disagreements - captured,
        "residual_disagreement_fraction": (
            (disagreements - captured) / skipped if skipped else None
        ),
    }


def route(positions: list[Position], fraction: float, signal: str) -> dict[str, object]:
    order = sorted(
        range(len(positions)),
        key=lambda index: (
            -positions[index].normalized_entropy
            if signal == "normalized_entropy"
            else positions[index].margin,
            positions[index].seed,
            index,
        ),
    )
    budget = math.floor(len(positions) * fraction)
    selected = set(order[:budget])
    return {
        "query_budget_fraction": fraction,
        "query_count": budget,
        "uniform_expected_capture_fraction": budget / len(positions),
        "overall": subgroup(positions, selected, list(range(len(positions)))),
        "by_seat": {
            str(seat): subgroup(
                positions,
                selected,
                [
                    index
                    for index, position in enumerate(positions)
                    if position.seat == seat
                ],
            )
            for seat in range(2)
        },
        "by_legal_count": {
            name: subgroup(
                positions,
                selected,
                [
                    index
                    for index, position in enumerate(positions)
                    if legal_count_bin(position.legal_count) == name
                ],
            )
            for name in ("1", "2-4", "5-16", "17+")
        },
    }


def run(fit_path: Path, source_path: Path) -> dict[str, object]:
    if digest(fit_path) != FIT_SHA256 or digest(source_path) != SOURCE_SHA256:
        raise ValueError("source uncertainty inputs differ from predeclared hashes")
    torch.set_num_threads(1)
    dataset = checked_dataset(fit_path, FIT_SEED, FIT_MAPS)
    source, source_config = load_policy(
        source_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    positions = []
    with torch.inference_mode():
        for replayed in replay_slate_positions(dataset):
            observation = replayed.root.observe()
            legal_count = int(np.diff(observation["action_offsets"])[0])
            plans = cast(list[list[int]], replayed.record["candidate_action_indices"])
            teacher = plans[cast(int, replayed.record["selected_index"])][0]
            if teacher < 0 or teacher >= legal_count:
                raise ValueError("selected teacher action is not locally legal")
            rules = encode_rules(replayed.root.rules_json(), torch.device("cpu"))
            logits, _ = source(observation, rules)
            distribution = action_distribution(logits, observation["action_offsets"])
            probabilities = distribution.probs[0, :legal_count].cpu().numpy()
            top, entropy, margin = confidence(probabilities)
            positions.append(
                Position(
                    seed=replayed.seed,
                    seat=cast(int, replayed.record["seat"]),
                    legal_count=legal_count,
                    mismatch=int(np.argmax(probabilities)) != teacher,
                    top_probability=top,
                    normalized_entropy=entropy,
                    margin=margin,
                )
            )
    if len({position.seed for position in positions}) != FIT_MAPS:
        raise ValueError("source uncertainty audit did not cover every fit map")
    return {
        "kind": "source_uncertainty_search_routing_read_only",
        "protocol": PROTOCOL,
        "fit_sha256": FIT_SHA256,
        "source_sha256": SOURCE_SHA256,
        "source_expert": source_config["selected_expert"],
        "independent_maps": FIT_MAPS,
        "root_positions": len(positions),
        "source_teacher_first_action_disagreements": sum(
            position.mismatch for position in positions
        ),
        "median_top_probability": {
            label: float(
                np.median(
                    [
                        position.top_probability
                        for position in positions
                        if position.mismatch == mismatch
                    ]
                )
            )
            for label, mismatch in (
                ("source_matches_teacher", False),
                ("source_disagrees_with_teacher", True),
            )
        },
        "routes": {
            signal: {
                str(fraction): route(positions, fraction, signal)
                for fraction in (0.25, 0.5)
            }
            for signal in ("normalized_entropy", "top_probability_margin")
        },
        "qualification": "Post hoc teacher first-action disagreement on previously inspected roots, not outcome benefit, controller speed, Elo or promotion",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit", type=Path)
    parser.add_argument("source", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.fit, arguments.source), sort_keys=True))


if __name__ == "__main__":
    main()
