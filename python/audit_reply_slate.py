from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import cast

from antiyoy_rl.turn_credit import outcome_score

from .build_bundle import digest
from .evaluate import paired_comparison_summary

COUNT_FIELDS = (
    "candidate_states",
    "positions_with_distinct_first_actions",
    "positions_with_outcome_diversity",
    "positions_missing_reply_score",
    "reply_choice_differs_from_static",
    "reply_choice_changes_first_action",
    "positions_with_oracle_opportunity",
    "selected_outcome_censored",
    "selected_outcome_better",
    "selected_outcome_worse",
    "selected_outcome_same",
    "informative_candidate_pairs",
    "pairwise_concordant",
    "pairwise_discordant",
    "pairwise_tied_score",
)


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("reply slate audit requires sampled positions")
    counts: Counter[str] = Counter()
    seat_counts: dict[int, Counter[str]] = {}
    selected_ranks: Counter[int] = Counter()
    map_deltas: dict[int, int] = {}
    censored_maps: set[int] = set()
    reply_spans = []
    for record in records:
        seed = cast(int, record["seed"])
        seat = cast(int, record["seat"])
        seat_counts.setdefault(seat, Counter())["positions"] += 1
        search = cast(dict[str, object], record["search"])
        beam = cast(list[dict[str, object]], record["beam_candidates"])
        branches = [search] + [cast(dict[str, object], item["branch"]) for item in beam]
        continuations = cast(
            list[dict[str, object]], record["opponent_search_continuations"]
        )
        state_indices = [cast(int, branch["state_index"]) for branch in branches]
        if any(index >= len(continuations) for index in state_indices):
            raise ValueError("candidate state index exceeds continuation count")
        replies = [continuations[index]["reply_score"] for index in state_indices]
        outcomes = [
            outcome_score(continuations[index], seat) for index in state_indices
        ]
        first_actions = [
            json.dumps(cast(list[object], branch["actions"])[0], sort_keys=True)
            for branch in branches
        ]
        counts["candidate_states"] += len(branches)
        counts["positions_with_distinct_first_actions"] += len(set(first_actions)) > 1
        counts["positions_with_outcome_diversity"] += (
            len({outcome for outcome in outcomes if outcome >= 0}) > 1
        )
        if any(reply is None for reply in replies):
            counts["positions_missing_reply_score"] += 1
            censored_maps.add(seed)
            continue
        scores = [cast(int, reply) for reply in replies]
        reply_spans.append(max(scores) - min(scores))
        selected = max(
            range(len(branches)),
            key=lambda index: (
                scores[index],
                cast(int, branches[index]["static_score"]),
                -index,
            ),
        )
        selected_ranks[selected] += 1
        counts["reply_choice_differs_from_static"] += selected != 0
        counts["reply_choice_changes_first_action"] += (
            first_actions[selected] != first_actions[0]
        )
        counts["positions_with_oracle_opportunity"] += outcomes[0] >= 0 and any(
            outcome > outcomes[0] for outcome in outcomes
        )
        for left in range(len(branches)):
            for right in range(left + 1, len(branches)):
                if outcomes[left] < 0 or outcomes[right] < 0:
                    continue
                outcome_preference = (outcomes[left] > outcomes[right]) - (
                    outcomes[left] < outcomes[right]
                )
                if outcome_preference == 0:
                    continue
                reply_preference = (scores[left] > scores[right]) - (
                    scores[left] < scores[right]
                )
                counts["informative_candidate_pairs"] += 1
                counts["pairwise_concordant"] += reply_preference == outcome_preference
                counts["pairwise_discordant"] += reply_preference == -outcome_preference
                counts["pairwise_tied_score"] += reply_preference == 0
        if outcomes[0] < 0 or outcomes[selected] < 0:
            counts["selected_outcome_censored"] += 1
            censored_maps.add(seed)
            continue
        difference = outcomes[selected] - outcomes[0]
        map_deltas[seed] = map_deltas.get(seed, 0) + difference
        counts["selected_outcome_better"] += difference > 0
        counts["selected_outcome_worse"] += difference < 0
        counts["selected_outcome_same"] += difference == 0
        seat_counts[seat]["better"] += difference > 0
        seat_counts[seat]["worse"] += difference < 0
        seat_counts[seat]["same"] += difference == 0
    complete_deltas = [
        delta for seed, delta in map_deltas.items() if seed not in censored_maps
    ]
    better_maps = sum(delta > 0 for delta in complete_deltas)
    worse_maps = sum(delta < 0 for delta in complete_deltas)
    same_maps = sum(delta == 0 for delta in complete_deltas)
    paired_maps = paired_comparison_summary(better_maps, worse_maps, same_maps)
    return {
        "positions": len(records),
        "independent_maps": len({cast(int, record["seed"]) for record in records}),
        **{field: counts[field] for field in COUNT_FIELDS},
        "median_reply_score_span": median(reply_spans) if reply_spans else None,
        "selected_rank_counts": {
            str(rank): count for rank, count in sorted(selected_ranks.items())
        },
        "by_seat": {
            str(seat): {
                field: seat_counts[seat][field]
                for field in ("positions", "better", "worse", "same")
            }
            for seat in sorted(seat_counts)
        },
        "independent_maps_selected_better": better_maps,
        "independent_maps_selected_worse": worse_maps,
        "independent_maps_selected_same": same_maps,
        "independent_maps_censored": len(censored_maps),
        "independent_map_sign_test_p": paired_maps["exact_two_sided_sign_test_p"],
    }


def audit(paths: list[Path]) -> dict[str, object]:
    reports = []
    for path in paths:
        source = (
            gzip.open(path, "rt", encoding="utf-8")
            if path.suffix == ".gz"
            else path.open(encoding="utf-8")
        )
        with source:
            reports.append(cast(dict[str, object], json.load(source)))
    for report in reports:
        generator = cast(dict[str, object], report["generator"])
        if generator["players"] != 2 or report["opponent_search_nodes"] is None:
            raise ValueError(
                "reply slate audit requires searched two-player continuations"
            )
        if (
            report["include_observations"] is not True
            or cast(int, report["beam_slate_size"]) < 2
        ):
            raise ValueError(
                "reply slate audit requires observed multi-turn candidate slates"
            )
    if len({report["opponent_search_nodes"] for report in reports}) != 1:
        raise ValueError("reply slate audit requires a shared opponent search budget")
    rollins = {report.get("reply_rollin", False) for report in reports}
    if len(rollins) != 1:
        raise ValueError("reply slate audit requires a shared roll-in policy")
    return {
        "kind": "procedural_duel_reply_slate_audit",
        "source_files": [{"name": path.name, "sha256": digest(path)} for path in paths],
        "opponent_search_nodes": reports[0]["opponent_search_nodes"],
        "rollin": "reply_search" if rollins.pop() else "greedy",
        "qualification": "conditional continuation labels under greedy root and searched opponents; candidate pairs within maps are correlated",
        **summarize(
            [
                cast(dict[str, object], record)
                for report in reports
                for record in cast(list[object], report["records"])
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, action="append", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.input), sort_keys=True))


if __name__ == "__main__":
    main()
