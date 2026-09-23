from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import cast

from .build_bundle import digest


INVALID_PROVINCE = 65535
UNIT_SCORES = (0, 28, 82, 210, 430)
OBJECT_SCORES = (0, 42, 58, 46, 96, -10, -10, -16)


def observed_position_score(
    observation: dict[str, object], state: int, root_seat: int
) -> int:
    cell_offsets = cast(list[int], observation["cell_offsets"])
    province_offsets = cast(list[int], observation["province_offsets"])
    cell_start, cell_end = cell_offsets[state : state + 2]
    province_start, province_end = province_offsets[state : state + 2]
    owners = cast(list[int], observation["owners"])
    objects = cast(list[int], observation["objects"])
    strengths = cast(list[int], observation["unit_strengths"])
    defenses = cast(list[int], observation["defenses"])
    province_ids = cast(list[int], observation["province_ids"])
    province_owners = cast(list[int], observation["province_owners"])
    province_money = cast(list[int], observation["province_money"])
    province_profit = cast(list[int], observation["province_profit"])
    rules = cast(list[dict[str, object]], observation["rules"])
    economy = cast(dict[str, object], rules[state]["economy"])
    unit_upkeep = cast(list[int], economy["unit_upkeep"])
    incomes = [0] * (province_end - province_start)
    upkeeps = [0] * (province_end - province_start)
    score = 0
    for cell in range(cell_start, cell_end):
        owner = owners[cell]
        if owner == 255:
            continue
        direction = 1 if owner == root_seat else -1
        strength = strengths[cell]
        obj = objects[cell]
        score += direction * (
            128 + UNIT_SCORES[strength] + OBJECT_SCORES[obj] + defenses[cell] * 5
        )
        province = province_ids[cell]
        if province == INVALID_PROVINCE:
            continue
        incomes[province] += (
            0
            if obj in (5, 6)
            else int(economy["farm_hex_income"])
            if obj == 2
            else int(economy["clear_hex_income"])
        )
        upkeeps[province] += (
            unit_upkeep[strength]
            if strength > 0
            else int(economy["tower_upkeep"])
            if obj == 3
            else int(economy["strong_tower_upkeep"])
            if obj == 4
            else 0
        )
    for local, province in enumerate(range(province_start, province_end)):
        if incomes[local] - upkeeps[local] != province_profit[province]:
            raise ValueError("observed province profit disagrees with its cells")
        direction = 1 if province_owners[province] == root_seat else -1
        score += direction * (
            province_money[province]
            + incomes[local] * 11
            - upkeeps[local] * 15
            - 14
        )
    return score


def audit(paths: list[Path]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            dataset = cast(dict[str, object], json.load(source))
        records = cast(list[dict[str, object]], dataset["records"])
        stage_counts: dict[str, dict[str, int]] = {}
        for stage, score_name in (
            ("post_turn", "static_scores"),
            ("post_reply", "reply_scores"),
        ):
            compared = terminal = mismatches = maximum_error = 0
            for record in records:
                observation = cast(dict[str, object], record[stage])
                targets = cast(list[int], record[score_name])
                for state, target in enumerate(targets):
                    if abs(target) >= 1_000_000_000:
                        terminal += 1
                        continue
                    predicted = observed_position_score(
                        observation, state, int(record["seat"])
                    )
                    compared += 1
                    error = abs(predicted - target)
                    mismatches += error != 0
                    maximum_error = max(maximum_error, error)
            stage_counts[stage] = {
                "nonterminal_states_compared": compared,
                "terminal_score_states_excluded": terminal,
                "mismatches": mismatches,
                "maximum_absolute_error": maximum_error,
            }
        summary[path.name] = {
            "sha256": digest(path),
            "maps": dataset["maps"],
            "positions": len(records),
            "stages": stage_counts,
        }
    return {
        "kind": "procedural_duel_raw_observation_native_score_reconstruction",
        "files": summary,
        "qualification": "Read-only deterministic reconstruction on already inspected data; terminal winner is not in BatchObservation and terminal scores are excluded",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="+")
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.dataset), sort_keys=True))


if __name__ == "__main__":
    main()
