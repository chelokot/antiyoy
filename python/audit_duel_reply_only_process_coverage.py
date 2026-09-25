from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from antiyoy_rl import SCORE_COMPONENT_WEIGHTS


FIT_SHA256 = "f7a3e51c3659b4183dd87c759ff38175c630a6a345b366eda61b825d949b6da8"
WIN_SCORE = 10**12
COMPONENT_FIELDS = ("post_components", "reply_components", "followup_components")


def stage_score(candidate: dict, stage: int) -> int:
    terminal_class = candidate["terminal_classes"][stage]
    if terminal_class != 1:
        return WIN_SCORE if terminal_class == 2 else -WIN_SCORE
    return sum(
        value * weight
        for value, weight in zip(
            candidate[COMPONENT_FIELDS[stage]], SCORE_COMPONENT_WEIGHTS, strict=True
        )
    )


def plan_lengths(lengths: list[int]) -> list[int]:
    ordered = sorted(lengths)
    return [
        ordered[len(ordered) // 2],
        ordered[int(0.9 * (len(ordered) - 1))],
        ordered[-1],
    ]


def summarize(samples: list[dict]) -> dict:
    index_counts = Counter()
    action_counts = Counter()
    seat_counts = {0: Counter(), 1: Counter()}
    root_lengths = []
    reply_lengths = []
    chosen_reply_lengths = []
    followup_lengths = []
    terminal_root_replies = 0
    candidate_count = 0
    for sample in samples:
        candidates = sample["candidates"]
        teacher = sample["teacher_selected_index"]
        if sample["rollin_censored"] or not 0 <= teacher < len(candidates):
            raise ValueError("invalid fit sample")
        for candidate in candidates:
            if stage_score(candidate, 0) != candidate["static_score"]:
                raise ValueError("root score disagrees with its components")
            if stage_score(candidate, 2) != candidate["response_score"]:
                raise ValueError("three-turn score disagrees with its components")
            root_lengths.append(len(candidate["plan"]))
            reply_lengths.append(len(candidate["reply_plan"]))
            followup_lengths.append(len(candidate["followup_plan"]))
            terminal_root_replies += not candidate["reply_plan"]
            candidate_count += 1
        chosen_reply_lengths.append(len(candidates[teacher]["reply_plan"]))
        full = max(
            range(len(candidates)),
            key=lambda index: (
                candidates[index]["response_score"],
                candidates[index]["static_score"],
                -index,
            ),
        )
        if full != teacher:
            raise ValueError("recorded teacher choice disagrees with exact ranking")
        static = max(
            range(len(candidates)),
            key=lambda index: (candidates[index]["static_score"], -index),
        )
        reply = max(
            range(len(candidates)),
            key=lambda index: (
                stage_score(candidates[index], 1),
                candidates[index]["static_score"],
                -index,
            ),
        )
        static_index_correct = static == teacher
        reply_index_correct = reply == teacher
        static_action_correct = (
            candidates[static]["plan"][0] == candidates[teacher]["plan"][0]
        )
        reply_action_correct = (
            candidates[reply]["plan"][0] == candidates[teacher]["plan"][0]
        )
        seat = sample["root_seat"]
        for counts, baseline, candidate in (
            (index_counts, static_index_correct, reply_index_correct),
            (action_counts, static_action_correct, reply_action_correct),
        ):
            counts["static_match"] += baseline
            counts["reply_match"] += candidate
            counts["better"] += candidate and not baseline
            counts["worse"] += baseline and not candidate
            counts["same"] += candidate == baseline
        seat_counts[seat]["samples"] += 1
        seat_counts[seat]["static_index_match"] += static_index_correct
        seat_counts[seat]["reply_index_match"] += reply_index_correct
        seat_counts[seat]["static_action_match"] += static_action_correct
        seat_counts[seat]["reply_action_match"] += reply_action_correct
    return {
        "samples": len(samples),
        "candidates": candidate_count,
        "candidate_indices": dict(index_counts),
        "first_actions": dict(action_counts),
        "by_root_seat": {
            str(seat): dict(counts) for seat, counts in seat_counts.items()
        },
        "root_plan_lengths_median_p90_max": plan_lengths(root_lengths),
        "reply_plan_lengths_median_p90_max": plan_lengths(reply_lengths),
        "selected_reply_plan_lengths_median_p90_max": plan_lengths(
            chosen_reply_lengths
        ),
        "followup_plan_lengths_median_p90_max": plan_lengths(followup_lengths),
        "zero_action_replies": terminal_root_replies,
    }


def audit(path: Path) -> dict:
    with path.open("rb") as raw:
        digest = hashlib.file_digest(raw, "sha256").hexdigest()
    if digest != FIT_SHA256:
        raise ValueError("fit dataset hash disagrees with the frozen audit input")
    with gzip.open(path, "rt", encoding="utf-8") as raw:
        header = json.loads(next(raw))
        if (
            header["kind"] != "structured_multi_turn_response_process_fit"
            or header["first_seed"] != 6610000
            or header["maps"] != 128
        ):
            raise ValueError("fit dataset header disagrees with the frozen audit input")
        samples = []
        games = 0
        for line in raw:
            record = json.loads(line)
            if record["type"] == "sample":
                samples.append(record)
            elif record["type"] == "game":
                if not record["terminal"] or record["truncated"]:
                    raise ValueError("fit game was censored")
                games += 1
            else:
                raise ValueError("unknown fit record")
    if games != 512:
        raise ValueError("fit ledger has a missing game")
    return {"fit_sha256": digest, "games": games, **summarize(samples)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.input), sort_keys=True))


if __name__ == "__main__":
    main()
