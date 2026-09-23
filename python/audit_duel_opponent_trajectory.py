from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl.model import ACTION_KIND_NAMES, encode_rules_batch
from antiyoy_rl.slate_dataset import load_teacher_slates

from .build_bundle import digest
from .evaluate import load_policy


def first_mismatch(expected: np.ndarray, predicted: np.ndarray) -> int | None:
    disagreement = np.flatnonzero(expected != predicted)
    return int(disagreement[0]) if len(disagreement) else None


def summarize(counts: Counter[str]) -> dict[str, float | int]:
    decisions = counts["decisions"]
    plans = counts["plans"]
    return {
        "decisions": decisions,
        "top_one_matches": counts["top_one_matches"],
        "top_three_matches": counts["top_three_matches"],
        "top_one_rate": counts["top_one_matches"] / decisions if decisions else 0.0,
        "top_three_rate": counts["top_three_matches"] / decisions if decisions else 0.0,
        "mean_teacher_action_probability": (
            counts["teacher_probability"] / decisions if decisions else 0.0
        ),
        "plans": plans,
        "whole_plan_matches": counts["whole_plan_matches"],
        "whole_plan_rate": counts["whole_plan_matches"] / plans if plans else 0.0,
        "first_mismatch_depth_0": counts["first_mismatch_depth_0"],
        "first_mismatch_depth_1": counts["first_mismatch_depth_1"],
        "first_mismatch_depth_2_plus": counts["first_mismatch_depth_2_plus"],
    }


def audit(dataset_path: Path, checkpoint_path: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    model, config = load_policy(
        checkpoint_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    model.eval()
    groups: dict[str, Counter[str]] = {}
    seeds: set[int] = set()
    positions = terminal_candidates = 0
    with torch.inference_mode():
        for position in load_teacher_slates(dataset_path):
            positions += 1
            seeds.add(position.seed)
            trace = position.opponent_decisions
            if trace is None:
                raise ValueError("opponent trajectory audit requires decision traces")
            observation = trace.observation
            count = len(trace.action_indices)
            selected = max(
                range(len(position.slate_indices)),
                key=lambda index: (
                    position.opponent_reply_scores[index],
                    position.static_scores[index],
                    -index,
                ),
            )
            if count:
                logits, _ = model(
                    observation,
                    encode_rules_batch(list(trace.rules_json), torch.device("cpu")),
                )
            for candidate in range(len(position.slate_indices)):
                start, end = trace.candidate_offsets[candidate : candidate + 2]
                if start == end:
                    terminal_candidates += 1
                    continue
                keys = [
                    "all",
                    f"root_seat_{position.seat}",
                    "selected" if candidate == selected else "unselected",
                ]
                if candidate > 0 and (
                    position.opponent_reply_scores[candidate]
                    > position.opponent_reply_scores[0]
                ):
                    keys.append("beneficial_alternative")
                expected = trace.action_indices[start:end]
                predicted = np.empty(end - start, dtype=np.int64)
                for depth, state in enumerate(range(start, end)):
                    action_start, action_end = observation["action_offsets"][
                        state : state + 2
                    ]
                    local_logits = logits[action_start:action_end]
                    label = int(expected[depth])
                    predicted[depth] = int(local_logits.argmax())
                    ranking = torch.argsort(local_logits, descending=True)
                    rank = int((ranking == label).nonzero(as_tuple=True)[0][0])
                    probability = float(local_logits.softmax(dim=0)[label])
                    kind = ACTION_KIND_NAMES[
                        int(observation["action_kinds"][action_start + label])
                    ]
                    decision_keys = [
                        *keys,
                        f"depth_{depth}" if depth < 2 else "depth_2_plus",
                        f"action_{kind}",
                    ]
                    for key in decision_keys:
                        counts = groups.setdefault(key, Counter())
                        counts["decisions"] += 1
                        counts["top_one_matches"] += rank == 0
                        counts["top_three_matches"] += rank < 3
                        counts["teacher_probability"] += probability
                mismatch = first_mismatch(expected, predicted)
                for key in keys:
                    counts = groups.setdefault(key, Counter())
                    counts["plans"] += 1
                    if mismatch is None:
                        counts["whole_plan_matches"] += 1
                    elif mismatch < 2:
                        counts[f"first_mismatch_depth_{mismatch}"] += 1
                    else:
                        counts["first_mismatch_depth_2_plus"] += 1
    return {
        "kind": "procedural_duel_opponent_trajectory_predictability",
        "dataset_sha256": digest(dataset_path),
        "checkpoint_sha256": digest(checkpoint_path),
        "selected_expert": config["selected_expert"],
        "independent_maps": len(seeds),
        "positions": positions,
        "terminal_candidates_without_opponent_plan": terminal_candidates,
        "groups": {key: summarize(counts) for key, counts in sorted(groups.items())},
        "qualification": "Teacher-forced action agreement on correlated searched candidate plans, not autonomous rollout, game strength, or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.dataset, arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
