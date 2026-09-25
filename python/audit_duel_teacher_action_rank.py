from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch

from antiyoy_rl.model import ACTION_KIND_NAMES, encode_rules
from antiyoy_rl.slate_dataset import replay_slate_positions

from .audit_duel_markov_root_fidelity import FIT_SHA256
from .build_bundle import digest
from .evaluate import load_policy
from .train_three_turn_plan import FIT_MAPS, FIT_SEED, SOURCE_SHA256, checked_dataset


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-teacher-first-action-rank-v1.json"
TOP_K = (1, 2, 4, 8, 16, 32)


def teacher_rank(logits: torch.Tensor, teacher_action: int) -> int:
    order = torch.argsort(logits.reshape(-1), descending=True, stable=True).tolist()
    return order.index(teacher_action) + 1


def coverage(records: list[dict[str, int | str]]) -> dict[str, object]:
    disagreements = [record for record in records if cast(int, record["rank"]) > 1]
    return {
        "positions": len(records),
        "source_teacher_disagreements": len(disagreements),
        "teacher_in_source_top_k": {
            str(k): sum(cast(int, record["rank"]) <= k for record in records)
            for k in TOP_K
        },
        "teacher_in_source_top_k_on_disagreements": {
            str(k): sum(cast(int, record["rank"]) <= k for record in disagreements)
            for k in TOP_K
        },
    }


def audit(dataset_path: Path, checkpoint_path: Path) -> dict[str, object]:
    if digest(dataset_path) != FIT_SHA256 or digest(checkpoint_path) != SOURCE_SHA256:
        raise ValueError("teacher action rank inputs disagree with the protocol")
    dataset = checked_dataset(dataset_path, FIT_SEED, FIT_MAPS)
    torch.set_num_threads(1)
    source, config = load_policy(
        checkpoint_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    records: list[dict[str, int | str]] = []
    with torch.inference_mode():
        for position in replay_slate_positions(dataset):
            observation = position.root.observe()
            selected = cast(int, position.record["selected_index"])
            plans = cast(list[list[int]], position.record["candidate_action_indices"])
            teacher_action = plans[selected][0]
            legal_count = int(np.diff(observation["action_offsets"])[0])
            if not 0 <= teacher_action < legal_count:
                raise ValueError("teacher first action is not legal")
            rules = encode_rules(position.root.rules_json(), torch.device("cpu"))
            logits, _ = source(observation, rules)
            kind = ACTION_KIND_NAMES[int(observation["action_kinds"][teacher_action])]
            records.append(
                {
                    "seed": position.seed,
                    "root_seat": cast(int, position.record["seat"]),
                    "round": cast(int, position.record["round"]),
                    "teacher_action_kind": kind,
                    "legal_actions": legal_count,
                    "rank": teacher_rank(logits, teacher_action),
                }
            )
    if len({record["seed"] for record in records}) != FIT_MAPS:
        raise ValueError("teacher rank audit did not cover every fit map")
    return {
        "kind": "read_only_teacher_first_action_source_rank",
        "protocol": PROTOCOL,
        "dataset_sha256": FIT_SHA256,
        "checkpoint_sha256": SOURCE_SHA256,
        "source_expert": config["selected_expert"],
        "records": records,
        "overall": coverage(records),
        "by_root_seat": {
            str(seat): coverage(
                [record for record in records if record["root_seat"] == seat]
            )
            for seat in (0, 1)
        },
        "by_teacher_action_kind": {
            kind: coverage(
                [record for record in records if record["teacher_action_kind"] == kind]
            )
            for kind in ACTION_KIND_NAMES
        },
        "qualification": "Previously inspected fit-state candidate inclusion only; not terminal credit, fresh strength, Elo, policy training or promotion",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.dataset, arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
