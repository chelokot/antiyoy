from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .audit_duel_first_regret import CHECKPOINT_SHA256, load_routed_policy
from .build_bundle import digest
from .collect_duel_episode_preferences import play, summarize, verify_trace


PROTOCOL = (
    "benchmarks/protocols/2026-09-25-duel-terminal-episode-preference-distillation-v1.json"
)
WINDOWS = {
    "fit": (6561000, 128, 64, 24),
    "resource_calibration": (6562000, 8, 0, 0),
    "offline_validation": (6563000, 64, 24, 0),
}


def collect(source_path: Path, window: str) -> dict[str, object]:
    if digest(source_path) != CHECKPOINT_SHA256:
        raise ValueError("frozen source checkpoint disagrees with the protocol")
    first_seed, maps, minimum_informative, minimum_per_seat = WINDOWS[window]
    torch.set_num_threads(1)
    source, experts = load_routed_policy(source_path)
    started = time.perf_counter()
    records = []
    with torch.inference_mode():
        for seed in range(first_seed, first_seed + maps):
            for teacher_seat in (None, 0, 1):
                record = play(seed, teacher_seat, source)
                verify_trace(record)
                records.append(record)
    elapsed = time.perf_counter() - started
    summary = summarize(
        records,
        first_seed,
        maps,
        minimum_informative,
        minimum_per_seat,
    )
    summary["collection_elapsed_seconds"] = elapsed
    if window == "resource_calibration":
        summary = {
            "maps": maps,
            "games": summary["games"],
            "terminal_games": summary["terminal_games"],
            "censored_pairs": summary["censored_pairs"],
            "collection_elapsed_seconds": elapsed,
            "integrity_gate_passed": summary["data_gate_passed"],
        }
    return {
        "kind": "terminal_whole_episode_preference_window",
        "protocol": PROTOCOL,
        "window": window,
        "first_seed": first_seed,
        "maps": maps,
        "source_sha256": CHECKPOINT_SHA256,
        "source_experts": experts,
        "records": records,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--window", choices=tuple(WINDOWS), required=True)
    arguments = parser.parse_args()
    print(json.dumps(collect(arguments.source, arguments.window), sort_keys=True))


if __name__ == "__main__":
    main()
