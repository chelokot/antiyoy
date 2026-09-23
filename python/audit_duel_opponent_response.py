from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import cast

import torch

from antiyoy_rl.model import encode_rules_batch
from antiyoy_rl.turn_credit import model_observation

from .build_bundle import digest
from .evaluate import load_policy


INVALID_HEX = 65535
STRUCTURE_CODES = {"Farm": 0, "Tower": 1, "StrongTower": 2}
COMMAND_CODES = {
    "DeclareWar": 0,
    "ProposeNeutral": 1,
    "ProposeFriendship": 2,
    "ProposeAlliance": 3,
    "Accept": 4,
    "Reject": 5,
}


def action_feature(action: object) -> tuple[str, int, int, int]:
    if action == "EndTurn":
        return "EndTurn", INVALID_HEX, INVALID_HEX, 0
    variant = cast(dict[str, dict[str, object]], action)
    kind, fields = next(iter(variant.items()))
    if kind == "Move":
        return kind, int(fields["source"]), int(fields["target"]), 0
    if kind == "Recruit":
        return (
            kind,
            int(fields["province"]),
            int(fields["target"]),
            int(fields["strength"]),
        )
    if kind == "Build":
        return (
            kind,
            INVALID_HEX,
            int(fields["target"]),
            STRUCTURE_CODES[str(fields["structure"])],
        )
    if kind == "PlantTree":
        return kind, INVALID_HEX, int(fields["target"]), 0
    if kind == "Diplomacy":
        return (
            kind,
            INVALID_HEX,
            int(fields["target"]),
            COMMAND_CODES[str(fields["command"])],
        )
    raise ValueError(f"unsupported action kind: {kind}")


def legal_action_index(action: object, legal: list[dict[str, object]]) -> int:
    target = action_feature(action)
    matches = [
        index
        for index, feature in enumerate(legal)
        if (
            feature["kind"],
            int(feature["source"]),
            int(feature["target"]),
            int(feature["parameter"]),
        )
        == target
    ]
    if len(matches) != 1:
        raise ValueError("searched opponent action has no unique legal action")
    return matches[0]


def has_opponent_response(plan: list[object], root_seat: int, active_seat: int) -> bool:
    if not plan:
        return False
    if active_seat != 1 - root_seat:
        raise ValueError("post-turn observation has the wrong opponent seat")
    return True


def summarize(counts: Counter[str]) -> dict[str, float | int]:
    candidates = counts["candidates"]
    return {
        "candidates": candidates,
        "top_one_matches": counts["top_one_matches"],
        "top_three_matches": counts["top_three_matches"],
        "top_one_rate": counts["top_one_matches"] / candidates if candidates else 0.0,
        "top_three_rate": counts["top_three_matches"] / candidates if candidates else 0.0,
        "mean_teacher_action_probability": (
            counts["teacher_action_probability"] / candidates if candidates else 0.0
        ),
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
    with gzip.open(dataset_path, "rt", encoding="utf-8") as source:
        dataset = cast(dict[str, object], json.load(source))
    records = cast(list[dict[str, object]], dataset["records"])
    if dataset["schema_version"] != 1:
        raise ValueError("unsupported opponent plan dataset")
    groups: dict[str, Counter[str]] = {}
    maps: set[int] = set()
    skipped_terminal = 0
    with torch.inference_mode():
        for record in records:
            seat = int(record["seat"])
            seed = int(record["seed"])
            maps.add(seed)
            serialized = cast(dict[str, object], record["post_turn"])
            observation, rules = model_observation(serialized)
            logits, _ = model(observation, encode_rules_batch(list(rules), torch.device("cpu")))
            offsets = cast(list[int], serialized["action_offsets"])
            legal = cast(list[dict[str, object]], serialized["actions"])
            plans = cast(list[list[object]], record["opponent_actions"])
            selected = int(record["selected_index"])
            reply_scores = cast(list[int], record["reply_scores"])
            if len(plans) != len(offsets) - 1:
                raise ValueError("opponent plans and candidate states disagree")
            for index, plan in enumerate(plans):
                if not has_opponent_response(
                    plan, seat, int(observation["active_players"][index])
                ):
                    skipped_terminal += 1
                    continue
                start, end = offsets[index : index + 2]
                label = legal_action_index(plan[0], legal[start:end])
                probabilities = logits[start:end].softmax(dim=0)
                ranking = torch.argsort(logits[start:end], descending=True)
                rank = int((ranking == label).nonzero(as_tuple=True)[0][0])
                keys = [
                    "all",
                    f"root_seat_{seat}",
                    "selected" if index == selected else "unselected",
                    "static" if index == 0 else "alternative",
                    f"action_{action_feature(plan[0])[0]}",
                ]
                if index > 0 and reply_scores[index] > reply_scores[0]:
                    keys.append("beneficial_alternative")
                for key in keys:
                    counts = groups.setdefault(key, Counter())
                    counts["candidates"] += 1
                    counts["top_one_matches"] += rank == 0
                    counts["top_three_matches"] += rank < 3
                    counts["teacher_action_probability"] += float(probabilities[label])
    return {
        "kind": "procedural_duel_opponent_first_action_predictability_audit",
        "dataset_sha256": digest(dataset_path),
        "checkpoint_sha256": digest(checkpoint_path),
        "selected_expert": config["selected_expert"],
        "independent_maps": len(maps),
        "positions": len(records),
        "terminal_candidates_without_opponent_action": skipped_terminal,
        "groups": {key: summarize(counts) for key, counts in sorted(groups.items())},
        "qualification": "Read-only action agreement on searched post-turn states; correlated candidates are not games or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.dataset, arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
