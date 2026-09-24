from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import cast

from .audit_duel_source_reply_ranking import CandidateRecord, audit
from .build_bundle import digest
from .evaluate import paired_comparison_summary


def audit_on_policy(evaluation: dict[str, object]) -> dict[str, object]:
    if evaluation["model_agent"] != "model_reply_search":
        raise ValueError("on-policy reply audit requires the hybrid agent")
    records = cast(list[dict[str, object]], evaluation["model_reply_native_audit"])
    normalized = []
    map_seat_regrets: dict[tuple[int, int], list[float]] = defaultdict(list)
    for record in records:
        native = cast(list[int], record["native_reply_scores"])
        predicted = cast(list[int], record["autonomous_reply_scores"])
        static = cast(list[int], record["static_scores"])
        teacher = cast(int, record["native_selected_index"])
        selected = cast(int, record["autonomous_selected_index"])
        expected_selected = max(
            range(len(predicted)),
            key=lambda index: (predicted[index], static[index], -index),
        )
        if selected != expected_selected:
            raise ValueError("audited hybrid choice disagrees with its reply scores")
        seed = cast(int, record["seed"])
        seat = cast(int, record["root_seat"])
        normalized.append(
            cast(
                CandidateRecord,
                {
                    "seed": seed,
                    "root_seat": seat,
                    "round": record["round"],
                    "native_selected_index": teacher,
                    "native_reply_scores": native,
                    "static_scores": static,
                    "autonomous_selected_indices": {
                        "source": selected,
                        "student": selected,
                    },
                    "autonomous_reply_scores": {
                        "source": predicted,
                        "student": predicted,
                    },
                },
            )
        )
        map_seat_regrets[(seed, seat)].append(
            math.asinh(native[teacher] / 1000) - math.asinh(native[selected] / 1000)
        )
    ranking = audit(normalized)
    seeds = sorted({seed for seed, _ in map_seat_regrets})
    paired = [
        (
            seed,
            sum(map_seat_regrets[(seed, 0)]) / len(map_seat_regrets[(seed, 0)]),
            sum(map_seat_regrets[(seed, 1)]) / len(map_seat_regrets[(seed, 1)]),
        )
        for seed in seeds
        if (seed, 0) in map_seat_regrets and (seed, 1) in map_seat_regrets
    ]
    seat_one_higher = sum(one > zero for _, zero, one in paired)
    seat_zero_higher = sum(one < zero for _, zero, one in paired)
    same = len(paired) - seat_one_higher - seat_zero_higher
    regret_games = {
        (entry["seed"], entry["root_seat"])
        for entry in cast(list[dict[str, object]], ranking["strict_regret_position_ledger"])
    }
    loss_overlap = {
        seat: {
            "wins_with_regret": 0,
            "wins_without_regret": 0,
            "losses_with_regret": 0,
            "losses_without_regret": 0,
        }
        for seat in (0, 1)
    }
    for seed, seat, winner in zip(
        cast(list[int], evaluation["game_seeds"]),
        cast(list[int], evaluation["model_seats"]),
        cast(list[int], evaluation["winners"]),
        strict=True,
    ):
        if winner == 255:
            continue
        outcome = "wins" if winner == seat else "losses"
        regret = "with_regret" if (seed, seat) in regret_games else "without_regret"
        loss_overlap[seat][f"{outcome}_{regret}"] += 1
    return {
        "kind": "post_rejection_hybrid_on_policy_reply_ranking_diagnostic",
        "seed": evaluation["seed"],
        "games": len(cast(list[int], evaluation["winners"])),
        "winners": evaluation["winners"],
        "seats": evaluation["seats"],
        "sampled_candidate_states": sum(len(record["native_reply_scores"]) for record in records),
        "ranking": ranking,
        "strict_regret_game_outcome_overlap_by_model_seat": loss_overlap,
        "paired_map_seat_mean_transformed_regret": {
            "paired_maps": len(paired),
            "seat_one_higher": seat_one_higher,
            "seat_zero_higher": seat_zero_higher,
            "same": same,
            "exact_two_sided_sign_test_p": paired_comparison_summary(
                seat_one_higher, seat_zero_higher, same
            )["exact_two_sided_sign_test_p"],
        },
        "qualification": "Conditional native one-turn reply-score audit on failed-hybrid game states; not a new policy gate, terminal outcome label, or Elo estimate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation", type=Path)
    arguments = parser.parse_args()
    evaluation = json.loads(arguments.evaluation.read_text(encoding="utf-8"))
    report = audit_on_policy(evaluation)
    report["evaluation_sha256"] = digest(arguments.evaluation)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
