from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_duel_teacher_plan_cache import audit_window


PROTOCOL = (
    "benchmarks/protocols/2026-09-24-duel-teacher-plan-cache-confirmation-v1.json"
)
SEED_FIRST = 6491000
MAPS = 64


def audit(checkpoint_path: Path) -> dict[str, object]:
    return audit_window(
        checkpoint_path,
        SEED_FIRST,
        MAPS,
        PROTOCOL,
        "native_three_turn_cached_vs_replanned_teacher_independent_confirmation",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
