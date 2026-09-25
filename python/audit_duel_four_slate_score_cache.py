from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .audit_duel_exact_score_cache import replay, summarize
from .audit_duel_first_regret import ACTION_LIMIT


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-four-slate-exact-score-cache-v1.json"
FIRST_SEED = 6620000
MAPS = 16


def summarize_four_slate(records: list[dict[str, object]]) -> dict[str, object]:
    summary = summarize(records)
    gate = cast(dict[str, bool], summary["gate"])
    del gate["median_map_cpu_improvement_at_least_ten_percent"]
    gate["median_map_cpu_improvement_at_least_three_percent"] = (
        cast(float, summary["median_map_cpu_improvement"]) >= 0.03
    )
    recent_hits = sum(cast(int, record["plain_cache_hits"]) for record in records)
    history_hits = sum(cast(int, record["cache_hits"]) for record in records)
    gate["four_slates_reused_additional_exact_scores"] = history_hits > recent_hits
    summary["one_slate_score_reuses"] = recent_hits
    summary["four_slate_score_reuses"] = history_hits
    summary["additional_exact_score_reuses"] = history_hits - recent_hits
    summary["advance_gate_passed"] = all(gate.values())
    return summary


def audit() -> dict[str, object]:
    records = [
        replay(seed, baseline_score_cache=True, candidate_four_slate=True)
        for seed in range(FIRST_SEED, FIRST_SEED + MAPS)
    ]
    return {
        "kind": "native_replanned_four_slate_exact_score_cache_cpu_and_equivalence",
        "protocol": PROTOCOL,
        "seed_first": FIRST_SEED,
        "maps": MAPS,
        "action_limit": ACTION_LIMIT,
        "records": records,
        "summary": summarize_four_slate(records),
        "qualification": "One-core native search CPU and exact semantic equivalence only; no new strength, Elo, browser latency or policy promotion",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit()
    if arguments.output is None:
        print(json.dumps(report, sort_keys=True))
    else:
        arguments.output.write_text(json.dumps(report, sort_keys=True) + "\n")
        print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
