from __future__ import annotations

import argparse
import json
from typing import cast

from .audit_duel_first_regret import ACTION_LIMIT, BranchOutcome
from .audit_duel_replan_head_to_head import play_match, summarize
from .audit_duel_teacher_coverage import finite_score
from .evaluate import paired_comparison_summary


PROTOCOL = "benchmarks/protocols/2026-09-24-duel-replan-profiles-v1.json"
WINDOWS = {
    "online_default_v1": 6495000,
    "classic_slay_2022": 6496000,
}
MAPS = 64


def worst_case_comparison(records: list[dict[str, object]]) -> dict[str, float | int]:
    indexed = {
        (cast(int, record["seed"]), cast(int, record["replanned_seat"])): record
        for record in records
    }
    scores = []
    for seed in sorted({cast(int, record["seed"]) for record in records}):
        score = 0.0
        for seat in (0, 1):
            branch = cast(BranchOutcome, indexed[seed, seat]["outcome"])
            score += 0.0 if branch["truncated"] else finite_score(branch, seat)
        scores.append(score)
    better = sum(score > 1 for score in scores)
    worse = sum(score < 1 for score in scores)
    return paired_comparison_summary(better, worse, len(scores) - better - worse)


def audit(profile: str) -> dict[str, object]:
    seed_first = WINDOWS[profile]
    records = [
        play_match(seed, seat, profile)
        for seed in range(seed_first, seed_first + MAPS)
        for seat in (0, 1)
    ]
    summary = summarize(records)
    worst_case = worst_case_comparison(records)
    paired = cast(dict[str, object], summary["replanned_vs_cached_paired_maps"])
    replanned_wins = cast(list[int], summary["replanned_wins_by_seat"])
    cached_wins = cast(list[int], summary["cached_wins_by_seat"])
    timing = cast(dict[str, object], summary["turn_timing"])
    replanned_timing = cast(dict[str, float | int], timing["replanned"])
    gate = {
        "both_seats_nonnegative": all(
            replanned >= cached
            for replanned, cached in zip(replanned_wins, cached_wins, strict=True)
        ),
        "independent_map_sign": (
            cast(int, paired["candidate_better"]) > cast(int, paired["baseline_better"])
            and cast(float, paired["exact_two_sided_sign_test_p"]) < 0.05
        ),
        "median_native_turn_below_250ms": replanned_timing["median_milliseconds"] < 250,
        "p95_native_turn_below_1000ms": replanned_timing["p95_milliseconds"] < 1000,
    }
    if profile == "online_default_v1":
        gate["zero_action_limit_adjudications"] = (
            summary["action_limit_adjudications"] == 0
        )
    else:
        gate["adjudication_rate_below_50_percent"] = (
            cast(int, summary["action_limit_adjudications"]) < MAPS
        )
        gate["worst_case_timeout_sign"] = (
            worst_case["candidate_better"] > worst_case["baseline_better"]
            and worst_case["exact_two_sided_sign_test_p"] < 0.05
        )
    return {
        "kind": "replanned_vs_cached_three_turn_rules_profile",
        "protocol": PROTOCOL,
        "profile": profile,
        "seed_first": seed_first,
        "maps": MAPS,
        "action_limit": ACTION_LIMIT,
        "records": records,
        "summary": summary,
        "worst_case_replanned_timeout_paired_maps": worst_case,
        "gate": gate,
        "profile_gate_passed": all(gate.values()),
        "qualification": "Classic Slay action-limit outcomes remain finite-horizon material adjudications, not terminal victories",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(WINDOWS), required=True)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.profile), sort_keys=True))


if __name__ == "__main__":
    main()
