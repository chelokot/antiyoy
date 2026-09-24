from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import cast

import torch

from .audit_duel_first_regret import ACTION_LIMIT, CHECKPOINT_SHA256, load_routed_policy
from .audit_duel_teacher_blocks import paired_outcomes, scheduled_teacher, summarize
from .audit_duel_teacher_coverage import play_game
from .build_bundle import digest


PROTOCOL = "benchmarks/protocols/2026-09-24-duel-teacher-block-confirmation-v1.json"
SEED_FIRST = 6480000
MAPS = 128
ARMS = ("direct", "first_1", "first_4", "spaced_4", "distributed_25")
SPACED_TURNS = (0, 2, 4, 6)


def confirmation_teacher(arm: str, seed: int, seat: int, turn: int) -> bool:
    if arm == "spaced_4":
        return turn in SPACED_TURNS
    return scheduled_teacher(arm, seed, seat, turn)


def audit(checkpoint_path: Path) -> dict[str, object]:
    checkpoint_sha256 = digest(checkpoint_path)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("teacher block confirmation requires the frozen routed-v6 checkpoint")
    torch.set_num_threads(1)
    policy, experts = load_routed_policy(checkpoint_path)
    records = []
    for seed in range(SEED_FIRST, SEED_FIRST + MAPS):
        for seat in (0, 1):
            for arm in ARMS:
                record = play_game(
                    seed,
                    seat,
                    0,
                    policy,
                    select_teacher=partial(confirmation_teacher, arm, seed, seat),
                )
                record["arm"] = arm
                records.append(record)
    indexed = {
        (cast(int, row["seed"]), cast(int, row["root_seat"]), cast(str, row["arm"])): row
        for row in records
    }
    maps = list(range(SEED_FIRST, SEED_FIRST + MAPS))
    summary = summarize(records, ARMS)
    summary["predeclared_contrasts"] = {
        "first_4_vs_spaced_4": paired_outcomes(indexed, maps, "first_4", "spaced_4"),
        "first_4_vs_distributed_25": paired_outcomes(
            indexed, maps, "first_4", "distributed_25"
        ),
    }
    return {
        "kind": "exact_teacher_four_turn_block_independent_confirmation",
        "protocol": PROTOCOL,
        "seed_first": SEED_FIRST,
        "maps": MAPS,
        "action_limit": ACTION_LIMIT,
        "checkpoint_sha256": checkpoint_sha256,
        "selected_experts": experts,
        "records": records,
        "summary": summary,
        "qualification": "Fresh complete-game mechanism confirmation with explicit censoring, not a trained student, promotion gate or global Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
