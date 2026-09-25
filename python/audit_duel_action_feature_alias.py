from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl.model import ACTION_KIND_NAMES, encode_rules
from antiyoy_rl.slate_dataset import replay_slate_positions

from .audit_duel_markov_root_fidelity import FIT_SHA256
from .audit_duel_teacher_action_rank import teacher_rank
from .build_bundle import digest
from .evaluate import load_policy
from .train_three_turn_plan import FIT_MAPS, FIT_SEED, SOURCE_SHA256, checked_dataset


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-action-feature-alias-v1.json"


def identical_action_features(features: torch.Tensor, action: int) -> list[int]:
    if features.ndim != 2 or not 0 <= action < features.shape[0]:
        raise ValueError("selected legal action is outside the feature matrix")
    return torch.nonzero(
        torch.all(features == features[action], dim=1), as_tuple=False
    ).reshape(-1).tolist()


def summarize(records: list[dict[str, object]]) -> dict[str, int]:
    disagreements = [record for record in records if record["rank"] != 1]
    return {
        "positions": len(records),
        "teacher_aliased_positions": sum(record["alias_count"] > 1 for record in records),
        "source_teacher_disagreements": len(disagreements),
        "teacher_aliased_on_disagreements": sum(
            record["alias_count"] > 1 for record in disagreements
        ),
        "source_top1_aliases_teacher_on_disagreements": sum(
            record["source_top1_in_alias"] for record in disagreements
        ),
    }


def audit(dataset_path: Path, checkpoint_path: Path) -> dict[str, object]:
    if digest(dataset_path) != FIT_SHA256 or digest(checkpoint_path) != SOURCE_SHA256:
        raise ValueError("action-feature alias inputs disagree with the protocol")
    dataset = checked_dataset(dataset_path, FIT_SEED, FIT_MAPS)
    torch.set_num_threads(1)
    source, config = load_policy(
        checkpoint_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for position in replay_slate_positions(dataset):
            observation = position.root.observe()
            selected = int(position.record["selected_index"])
            plans = position.record["candidate_action_indices"]
            teacher = int(plans[selected][0])
            legal_count = int(np.diff(observation["action_offsets"])[0])
            rules = encode_rules(position.root.rules_json(), torch.device("cpu"))
            logits, _, features = source.forward_with_action_features(
                observation, rules
            )
            if logits.numel() != legal_count or features.shape[0] != legal_count:
                raise ValueError("model actions disagree with replayed legal actions")
            aliases = identical_action_features(features, teacher)
            if not torch.all(logits[aliases] == logits[teacher]):
                raise ValueError("identical action features produced different logits")
            top1 = int(torch.argmax(logits).item())
            records.append(
                {
                    "seed": position.seed,
                    "seat": int(position.record["seat"]),
                    "round": int(position.record["round"]),
                    "kind": ACTION_KIND_NAMES[
                        int(observation["action_kinds"][teacher])
                    ],
                    "rank": teacher_rank(logits, teacher),
                    "teacher_action": teacher,
                    "source_top1": top1,
                    "alias_count": len(aliases),
                    "aliases": aliases,
                    "source_top1_in_alias": top1 in aliases,
                }
            )
    if len({record["seed"] for record in records}) != FIT_MAPS:
        raise ValueError("action-feature alias audit missed a fit map")
    return {
        "kind": "read_only_frozen_encoder_action_feature_alias",
        "protocol": PROTOCOL,
        "dataset_sha256": FIT_SHA256,
        "checkpoint_sha256": SOURCE_SHA256,
        "source_expert": config["selected_expert"],
        "overall": summarize(records),
        "by_seat": {
            str(seat): summarize(
                [record for record in records if record["seat"] == seat]
            )
            for seat in (0, 1)
        },
        "by_kind": {
            kind: summarize([record for record in records if record["kind"] == kind])
            for kind in ACTION_KIND_NAMES
        },
        "records": records,
        "qualification": "Old fit states and a frozen encoder only; no game outcome, new student, strength, Elo or deployment claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.dataset, arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
