from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .audit_three_turn_direct_model import DirectReport, audit as audit_direct


PROTOCOL = "benchmarks/protocols/2026-09-24-duel-replan-direct-model-v1.json"
SEED_FIRST = 6494000
MAPS = 256


class ReplannedDirectReport(DirectReport):
    replan_reply_search: bool


def audit(raw: ReplannedDirectReport) -> dict[str, object]:
    if raw["replan_reply_search"] is not True:
        raise ValueError("direct-model report did not use replanned reply search")
    result = audit_direct(raw, SEED_FIRST, MAPS)
    paired = cast(dict[str, object], result["paired_independent_maps"])
    native_wins = cast(list[int], result["native_wins_by_model_seat"])
    model_wins = cast(list[int], result["direct_model_wins_by_model_seat"])
    bootstrap = cast(list[float], result["map_bootstrap_95_head_to_head_elo"])
    gate = {
        "all_512_games_terminal": result["nonterminal_games"] == 0,
        "replanned_better_in_both_seats": all(
            native > model
            for native, model in zip(native_wins, model_wins, strict=True)
        ),
        "independent_map_sign": (
            cast(int, paired["candidate_better"]) > cast(int, paired["baseline_better"])
            and cast(float, paired["exact_two_sided_sign_test_p"]) < 0.05
        ),
        "bootstrap_lower_elo_positive": bootstrap[0] > 0,
    }
    return {
        **result,
        "kind": "replanned_three_turn_vs_frozen_direct_model_fresh_matched_audit",
        "protocol": PROTOCOL,
        "gate": gate,
        "direct_model_gate_passed": all(gate.values()),
        "qualification": "Actual fixed-pool head-to-head Elo, not global Elo or browser speed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    arguments = parser.parse_args()
    with arguments.input.open(encoding="utf-8") as stream:
        raw = cast(ReplannedDirectReport, json.load(stream))
    print(json.dumps(audit(raw), sort_keys=True))


if __name__ == "__main__":
    main()
