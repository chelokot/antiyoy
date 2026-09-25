from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_duel_structured_process_calibration import audit


PROTOCOL = (
    "benchmarks/protocols/2026-09-25-duel-structured-response-censor-aware-v2.json"
)
FIRST_SEED = 6611000
MAPS = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = json.dumps(
        audit(
            arguments.source,
            first_seed=FIRST_SEED,
            maps=MAPS,
            protocol=PROTOCOL,
            allow_rollin_censor=True,
            branch_action_limit=2**32 - 1,
        ),
        sort_keys=True,
    )
    if arguments.output:
        arguments.output.write_text(f"{report}\n")
        print(json.dumps(json.loads(report)["summary"], sort_keys=True))
    else:
        print(report)


if __name__ == "__main__":
    main()
