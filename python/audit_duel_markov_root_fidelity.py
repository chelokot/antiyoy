from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np
import torch

from antiyoy_rl.model import ACTION_KIND_NAMES, action_distribution, encode_rules
from antiyoy_rl.slate_dataset import replay_slate_positions

from .build_bundle import digest
from .evaluate import (
    empty_reply_teacher_agreement,
    load_policy,
    record_reply_teacher_action,
)
from .train_three_turn_plan import FIT_MAPS, FIT_SEED, SOURCE_SHA256, checked_dataset


FIT_SHA256 = "252f82f070c4fee9c707365d72c96766385afb4827276ce7030d837f8e7a7191"
STUDENT_SHA256 = "1d6e6530bb2f16d273910fdeb4b2de491d2e115017468eb3f31d3aea43f64fe4"


def legal_count_bin(count: int) -> str:
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 16:
        return "5-16"
    return "17+"


def empty_counts() -> dict[str, int]:
    return {**empty_reply_teacher_agreement(), "student_matches_source": 0}


def record_counts(
    counts: dict[str, int], source: int, student: int, teacher: int
) -> None:
    record_reply_teacher_action(counts, source, student, teacher)
    counts["student_matches_source"] += int(student == source)


def empty_shadow_counts() -> dict[str, int]:
    return {
        "positions": 0,
        "student_disagreements": 0,
        "shadow_disagreements": 0,
        "same_action": 0,
        "same_trigger": 0,
        "missed_student_disagreements": 0,
        "extra_shadow_disagreements": 0,
    }


def record_shadow_counts(
    counts: dict[str, int], source: int, student: int, shadow: int
) -> None:
    student_disagrees = student != source
    shadow_disagrees = shadow != source
    counts["positions"] += 1
    counts["student_disagreements"] += int(student_disagrees)
    counts["shadow_disagreements"] += int(shadow_disagrees)
    counts["same_action"] += int(student == shadow)
    counts["same_trigger"] += int(student_disagrees == shadow_disagrees)
    counts["missed_student_disagreements"] += int(
        student_disagrees and not shadow_disagrees
    )
    counts["extra_shadow_disagreements"] += int(
        shadow_disagrees and not student_disagrees
    )


def model_action(
    model: torch.nn.Module, observation: dict[str, np.ndarray], rules: torch.Tensor
) -> int:
    logits, _ = model(observation, rules)
    return action_from_logits(logits, observation)


def action_from_logits(logits: torch.Tensor, observation: dict[str, np.ndarray]) -> int:
    distribution = action_distribution(logits.reshape(-1), observation["action_offsets"])
    return int(distribution.probs.argmax(dim=1)[0])


def run(
    fit_path: Path,
    source_path: Path,
    student_path: Path,
    student_sha256: str = STUDENT_SHA256,
    shadow_disagreement: bool = False,
) -> dict[str, object]:
    hashes = {
        "dataset": digest(fit_path),
        "source": digest(source_path),
        "student": digest(student_path),
    }
    if hashes != {
        "dataset": FIT_SHA256,
        "source": SOURCE_SHA256,
        "student": student_sha256,
    }:
        raise ValueError("root fidelity inputs disagree with predeclared hashes")
    torch.set_num_threads(1)
    dataset = checked_dataset(fit_path, FIT_SEED, FIT_MAPS)
    source, source_config = load_policy(
        source_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    student, student_config = load_policy(student_path, torch.device("cpu"))
    total = empty_counts()
    by_seat = {str(seat): empty_counts() for seat in range(2)}
    by_kind = {kind: empty_counts() for kind in ACTION_KIND_NAMES}
    by_legal_count = {name: empty_counts() for name in ("1", "2-4", "5-16", "17+")}
    by_map: dict[int, int] = defaultdict(int)
    shadow_total = empty_shadow_counts()
    shadow_by_seat = {str(seat): empty_shadow_counts() for seat in range(2)}
    shadow_by_map: dict[int, dict[str, int]] = defaultdict(empty_shadow_counts)
    with torch.inference_mode():
        for position in replay_slate_positions(dataset):
            record = position.record
            plans = cast(list[list[int]], record["candidate_action_indices"])
            teacher = plans[cast(int, record["selected_index"])][0]
            observation = position.root.observe()
            legal = int(np.diff(observation["action_offsets"])[0])
            if teacher < 0 or teacher >= legal:
                raise ValueError("selected teacher action is not locally legal")
            rules = encode_rules(position.root.rules_json(), torch.device("cpu"))
            if shadow_disagreement:
                source_logits, _, source_features = source.forward_with_action_features(
                    observation, rules
                )
                source_action = action_from_logits(source_logits, observation)
                shadow_action = action_from_logits(
                    student.score_actions(source_features), observation
                )
            else:
                source_action = model_action(source, observation, rules)
            student_action = model_action(student, observation, rules)
            if shadow_disagreement:
                for counts in (
                    shadow_total,
                    shadow_by_seat[str(record["seat"])],
                    shadow_by_map[position.seed],
                ):
                    record_shadow_counts(
                        counts, source_action, student_action, shadow_action
                    )
            kind = ACTION_KIND_NAMES[int(observation["action_kinds"][teacher])]
            for counts in (
                total,
                by_seat[str(record["seat"])],
                by_kind[kind],
                by_legal_count[legal_count_bin(legal)],
            ):
                record_counts(counts, source_action, student_action, teacher)
            by_map[position.seed] += int(student_action == teacher) - int(
                source_action == teacher
            )
    if len(by_map) != FIT_MAPS:
        raise ValueError("root fidelity audit did not cover every fit map")
    result: dict[str, object] = {
        "kind": "rejected_student_root_fidelity_read_only",
        "hashes": hashes,
        "source_expert": source_config["selected_expert"],
        "student_expert": student_config["selected_expert"],
        "independent_maps": len(by_map),
        "total": total,
        "by_seat": by_seat,
        "by_teacher_action_kind": by_kind,
        "by_legal_count": by_legal_count,
        "map_net_agreement": {
            "student_higher": sum(value > 0 for value in by_map.values()),
            "source_higher": sum(value < 0 for value in by_map.values()),
            "same": sum(value == 0 for value in by_map.values()),
        },
        "map_net_agreement_by_seed": {
            str(seed): value for seed, value in sorted(by_map.items())
        },
        "qualification": "Post hoc same-root teacher-action agreement on previously inspected data; not game strength, causal terminal credit, Elo or student promotion",
    }
    if shadow_disagreement:
        result["shadow_disagreement"] = {
            "total": shadow_total,
            "by_seat": shadow_by_seat,
            "by_map": {str(seed): counts for seed, counts in sorted(shadow_by_map.items())},
            "qualification": "Frozen student head on frozen source features, not the original student policy or game strength",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("student", type=Path)
    parser.add_argument("--student-sha256", default=STUDENT_SHA256)
    parser.add_argument("--shadow-disagreement", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.fit,
                arguments.source,
                arguments.student,
                arguments.student_sha256,
                arguments.shadow_disagreement,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
