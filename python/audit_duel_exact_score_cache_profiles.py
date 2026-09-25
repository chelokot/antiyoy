from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .audit_duel_exact_score_cache import replay, summarize
from .audit_duel_first_regret import ACTION_LIMIT


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-exact-score-cache-profile-v1.json"
WINDOWS = {"online_default_v1": 6581000, "classic_slay_2022": 6582000}
MAPS = 16


def summarize_profile(records: list[dict[str, object]], profile: str) -> dict[str, object]:
    summary = summarize(records)
    baseline_gate = cast(dict[str, bool], summary["gate"])
    truncated = sum(
        cast(dict[str, object], record["outcome"])["truncated"] is True
        for record in records
    )
    gate = {
        name: passed
        for name, passed in baseline_gate.items()
        if name != "all_games_terminal"
    }
    gate["profile_censor_policy"] = (
        truncated == 0 if profile == "online_default_v1" else truncated < MAPS // 2
    )
    return {
        **summary,
        "terminal_games": len(records) - truncated,
        "action_limit_censored_games": truncated,
        "profile_gate": gate,
        "profile_gate_passed": all(gate.values()),
    }


def audit(profile: str) -> dict[str, object]:
    seed_first = WINDOWS[profile]
    records = [replay(seed, profile) for seed in range(seed_first, seed_first + MAPS)]
    return {
        "kind": "native_replanned_exact_score_cache_cross_profile_cpu_and_equivalence",
        "protocol": PROTOCOL,
        "profile": profile,
        "seed_first": seed_first,
        "maps": MAPS,
        "action_limit": ACTION_LIMIT,
        "records": records,
        "summary": summarize_profile(records, profile),
        "qualification": "Exact semantic equivalence and one-core native CPU only; Slay action-limit outcomes remain nonterminal, and neither profile result is browser latency or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(WINDOWS), required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = audit(arguments.profile)
    if arguments.output is None:
        print(json.dumps(report, sort_keys=True))
    else:
        arguments.output.write_text(json.dumps(report, sort_keys=True) + "\n")
        print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
