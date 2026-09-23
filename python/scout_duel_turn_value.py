from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from antiyoy_rl.slate_dataset import TeacherSlatePosition
from antiyoy_rl.turn_credit import TurnCreditPosition, load_turn_credit_positions

from .build_bundle import digest
from .evaluate import paired_comparison_summary
from .scout_turn_value import EmbeddedPosition, pairwise_examples, train_head


FEATURE_NAMES = (
    "static_score",
    "territory_delta",
    "unit_one_delta",
    "unit_two_delta",
    "unit_three_delta",
    "unit_four_delta",
    "ready_strength_delta",
    "capital_delta",
    "farm_delta",
    "tower_delta",
    "strong_tower_delta",
    "tree_delta",
    "defense_delta",
    "treasury_delta",
    "province_profit_delta",
    "province_count_delta",
    "largest_province_delta",
)


@dataclass(frozen=True)
class DuelEmbedding:
    position: EmbeddedPosition
    reply_index: int
    first_actions: tuple[str, ...]


def side_features(
    observation: dict[str, np.ndarray], state: int, owner: int
) -> np.ndarray:
    cell_start, cell_end = observation["cell_offsets"][state : state + 2]
    province_start, province_end = observation["province_offsets"][state : state + 2]
    cells = slice(int(cell_start), int(cell_end))
    provinces = slice(int(province_start), int(province_end))
    owned = observation["owners"][cells] == owner
    objects = observation["objects"][cells][owned]
    strengths = observation["unit_strengths"][cells][owned]
    ready = observation["ready"][cells][owned]
    defenses = observation["defenses"][cells][owned]
    owned_provinces = observation["province_owners"][provinces] == owner
    sizes = observation["province_sizes"][provinces][owned_provinces]
    return np.asarray(
        [
            owned.sum(),
            (strengths == 1).sum(),
            (strengths == 2).sum(),
            (strengths == 3).sum(),
            (strengths == 4).sum(),
            (strengths * ready).sum(),
            (objects == 1).sum(),
            (objects == 2).sum(),
            (objects == 3).sum(),
            (objects == 4).sum(),
            ((objects == 5) | (objects == 6)).sum(),
            defenses.sum(),
            observation["province_money"][provinces][owned_provinces].sum(),
            observation["province_profit"][provinces][owned_provinces].sum(),
            owned_provinces.sum(),
            sizes.max(initial=0),
        ],
        dtype=np.float32,
    )


def embed_position(source: TurnCreditPosition | TeacherSlatePosition) -> DuelEmbedding:
    if (
        not source.slate_indices
        or len(source.slate_indices) != len(source.slate_first_actions)
        or not np.all(source.post_turn["player_counts"] == 2)
    ):
        raise ValueError("duel scout requires two-player completed-turn slates")
    if source.opponent_reply_scores is None:
        raise ValueError("duel scout requires opponent reply scores")
    indices = np.asarray(source.slate_indices, dtype=np.int64)
    reply_scores = source.opponent_reply_scores[indices]
    if not np.isfinite(reply_scores).all():
        raise ValueError("duel scout requires complete opponent replies")
    rows = []
    for state in indices:
        own = side_features(source.post_turn, int(state), source.seat)
        opponent = side_features(source.post_turn, int(state), 1 - source.seat)
        rows.append(
            np.concatenate(
                (
                    np.asarray([source.static_scores[state] / 1000], dtype=np.float32),
                    own - opponent,
                )
            )
        )
    static_scores = source.static_scores[indices]
    reply_index = max(
        range(len(indices)),
        key=lambda index: (reply_scores[index], static_scores[index], -index),
    )
    return DuelEmbedding(
        position=EmbeddedPosition(
            seed=source.seed,
            seat=source.seat,
            features=torch.as_tensor(np.stack(rows)),
            outcomes=source.outcome_scores[indices],
            search_index=0,
        ),
        reply_index=reply_index,
        first_actions=source.slate_first_actions,
    )


def compare_choices(
    positions: list[DuelEmbedding], selected: list[int], baseline: list[int]
) -> dict[str, object]:
    if len(positions) != len(selected) or len(positions) != len(baseline):
        raise ValueError("selected choices must align with positions")
    better = worse = same = censored = changed = 0
    map_deltas: dict[int, int] = {}
    censored_maps: set[int] = set()
    by_seat: dict[int, dict[str, int]] = {}
    for position, choice, reference in zip(positions, selected, baseline, strict=True):
        example = position.position
        changed += choice != reference
        outcomes = example.outcomes
        seat = by_seat.setdefault(
            example.seat, {"better": 0, "worse": 0, "same": 0, "censored": 0}
        )
        if outcomes[choice] < 0 or outcomes[reference] < 0:
            censored += 1
            seat["censored"] += 1
            censored_maps.add(example.seed)
            continue
        difference = int(outcomes[choice] - outcomes[reference])
        map_deltas[example.seed] = map_deltas.get(example.seed, 0) + difference
        if difference > 0:
            better += 1
            seat["better"] += 1
        elif difference < 0:
            worse += 1
            seat["worse"] += 1
        else:
            same += 1
            seat["same"] += 1
    complete_maps = [
        delta for seed, delta in map_deltas.items() if seed not in censored_maps
    ]
    grouped = paired_comparison_summary(
        sum(delta > 0 for delta in complete_maps),
        sum(delta < 0 for delta in complete_maps),
        sum(delta == 0 for delta in complete_maps),
    )
    grouped["censored"] = len(censored_maps)
    return {
        "positions": len(positions),
        "changed_choices": changed,
        "better": better,
        "worse": worse,
        "same": same,
        "censored": censored,
        "by_seat": by_seat,
        "independent_maps": grouped,
    }


def evaluate(
    positions: list[DuelEmbedding], weight: Tensor, scales: Tensor
) -> dict[str, object]:
    choices = [
        int(prediction_scores(position, weight, scales).argmax())
        for position in positions
    ]
    static = [0] * len(positions)
    reply = [position.reply_index for position in positions]
    return {
        "versus_static": compare_choices(positions, choices, static),
        "versus_reply_search": compare_choices(positions, choices, reply),
        "reply_search_versus_static": compare_choices(positions, reply, static),
    }


def prediction_scores(
    position: DuelEmbedding, weight: Tensor, scales: Tensor
) -> Tensor:
    features = position.position.features
    scores = ((features / scales) @ weight).clone()
    for index in range(len(scores)):
        for previous in range(index):
            if torch.equal(features[index], features[previous]):
                scores[index] = scores[previous]
                break
    return scores


def pairwise_agreement(
    positions: list[DuelEmbedding], weight: Tensor, scales: Tensor
) -> dict[str, int]:
    informative = concordant = discordant = tied = indistinguishable = 0
    for position in positions:
        outcomes = position.position.outcomes
        scores = prediction_scores(position, weight, scales).numpy()
        for left in range(len(outcomes)):
            for right in range(left + 1, len(outcomes)):
                if min(outcomes[left], outcomes[right]) < 0:
                    continue
                preference = int(np.sign(int(outcomes[left] - outcomes[right])))
                if preference == 0:
                    continue
                informative += 1
                indistinguishable += bool(
                    torch.equal(
                        position.position.features[left],
                        position.position.features[right],
                    )
                )
                prediction = int(np.sign(scores[left] - scores[right]))
                concordant += prediction == preference
                discordant += prediction == -preference
                tied += prediction == 0
    return {
        "informative_pairs": informative,
        "concordant": concordant,
        "discordant": discordant,
        "tied": tied,
        "indistinguishable_features": indistinguishable,
    }


def scout(
    training_paths: list[Path], validation_paths: list[Path]
) -> dict[str, object]:
    torch.set_num_threads(1)
    training = [
        embed_position(position)
        for path in training_paths
        for position in load_turn_credit_positions(path, "teacher")
    ]
    validation = [
        embed_position(position)
        for path in validation_paths
        for position in load_turn_credit_positions(path, "teacher")
    ]
    training_maps = {position.position.seed for position in training}
    validation_maps = {position.position.seed for position in validation}
    if training_maps & validation_maps:
        raise ValueError("training and validation maps overlap")
    differences, weights = pairwise_examples(
        [position.position for position in training]
    )
    weight, scales = train_head(differences, weights)
    return {
        "kind": "procedural_duel_persistent_teacher_turn_value_scout",
        "feature_names": FEATURE_NAMES,
        "training_files": [
            {"name": path.name, "sha256": digest(path)} for path in training_paths
        ],
        "validation_files": [
            {"name": path.name, "sha256": digest(path)} for path in validation_paths
        ],
        "training_maps": len(training_maps),
        "validation_maps": len(validation_maps),
        "training_informative_pairs": len(differences),
        "training_pairwise_agreement": pairwise_agreement(training, weight, scales),
        "validation_pairwise_agreement": pairwise_agreement(validation, weight, scales),
        "weights": weight.tolist(),
        "feature_scales": scales.tolist(),
        "training": evaluate(training, weight, scales),
        "validation": evaluate(validation, weight, scales),
        "qualification": "Offline deterministic teacher continuations only; no full-game or policy-strength claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, action="append", type=Path)
    parser.add_argument("--validation", required=True, action="append", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(scout(arguments.train, arguments.validation), sort_keys=True))


if __name__ == "__main__":
    main()
