from __future__ import annotations

import argparse
import json
from pathlib import Path

from .collect_duel_structured_process import collect


PROTOCOL = (
    "benchmarks/protocols/2026-09-25-duel-structured-response-censor-aware-v2.json"
)
FIRST_SEED = 6612000
MAPS = 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = collect(
        arguments.source,
        arguments.output,
        first_seed=FIRST_SEED,
        maps=MAPS,
        protocol=PROTOCOL,
        allow_rollin_censor=True,
        branch_action_limit=2**32 - 1,
        dataset_role="offline_validation",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
