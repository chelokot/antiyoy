from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Literal, cast

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

OutcomeProbe = Literal["opponent", "teacher"]


def compare_probes(records: list[dict[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    changed_maps: set[int] = set()
    reversed_maps: set[int] = set()
    for record in records:
        seed = cast(int, record["seed"])
        seat = cast(int, record["seat"])
        opponent = cast(
            list[dict[str, object]], record["opponent_search_continuations"]
        )
        teacher = cast(list[dict[str, object]], record["teacher_continuations"])
        if len(opponent) != len(teacher):
            raise ValueError("continuation probes have different state counts")
        opponent_scores = [outcome_score(result, seat) for result in opponent]
        teacher_scores = [outcome_score(result, seat) for result in teacher]
        changed = False
        reversed_preference = False
        for index, (opponent_score, teacher_score) in enumerate(
            zip(opponent_scores, teacher_scores, strict=True)
        ):
            if min(opponent_score, teacher_score) < 0:
                counts["censored_state_pairs"] += 1
                continue
            counts["paired_complete_states"] += 1
            if opponent_score != teacher_score:
                counts["changed_state_outcomes"] += 1
                changed = True
            for other in range(index + 1, len(opponent_scores)):
                if min(opponent_scores[other], teacher_scores[other]) < 0:
                    continue
                opponent_preference = opponent_score - opponent_scores[other]
                teacher_preference = teacher_score - teacher_scores[other]
                if opponent_preference == 0 or teacher_preference == 0:
                    continue
                counts["strictly_informative_candidate_pairs_under_both"] += 1
                if opponent_preference * teacher_preference < 0:
                    counts["strict_pairwise_preference_reversals"] += 1
                    reversed_preference = True
        if changed:
            counts["positions_with_changed_state_outcome"] += 1
            changed_maps.add(seed)
        if reversed_preference:
            counts["positions_with_strict_preference_reversal"] += 1
            reversed_maps.add(seed)
    return {
        field: counts[field]
        for field in (
            "paired_complete_states",
            "censored_state_pairs",
            "changed_state_outcomes",
            "positions_with_changed_state_outcome",
            "strictly_informative_candidate_pairs_under_both",
            "strict_pairwise_preference_reversals",
            "positions_with_strict_preference_reversal",
        )
    } | {
        "independent_maps_with_changed_state_outcome": len(changed_maps),
        "independent_maps_with_strict_preference_reversal": len(reversed_maps),
    }


def summarize(
    records: list[dict[str, object]], outcome_probe: OutcomeProbe = "opponent"
) -> dict[str, object]:
    if not records:
        raise ValueError("reply slate audit requires sampled positions")
    counts: Counter[str] = Counter()
    seat_counts: dict[int, Counter[str]] = {}
    selected_ranks: Counter[int] = Counter()
    map_deltas: dict[int, int] = {}
    map_pair_deltas: dict[int, int] = {}
    censored_maps: set[int] = set()
    reply_spans = []
    for record in records:
        seed = cast(int, record["seed"])
        seat = cast(int, record["seat"])
        seat_counts.setdefault(seat, Counter())["positions"] += 1
        search = cast(dict[str, object], record["search"])
        beam = cast(list[dict[str, object]], record["beam_candidates"])
        branches = [search] + [cast(dict[str, object], item["branch"]) for item in beam]
        reply_continuations = cast(
            list[dict[str, object]], record["opponent_search_continuations"]
        )
        outcome_continuations = cast(
            list[dict[str, object]],
            record[
                "teacher_continuations"
                if outcome_probe == "teacher"
                else "opponent_search_continuations"
            ],
        )
        state_indices = [cast(int, branch["state_index"]) for branch in branches]
        if any(
            index >= min(len(reply_continuations), len(outcome_continuations))
            for index in state_indices
        ):
            raise ValueError("candidate state index exceeds continuation count")
        replies = [reply_continuations[index]["reply_score"] for index in state_indices]
        outcomes = [
            outcome_score(outcome_continuations[index], seat) for index in state_indices
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
                map_pair_deltas[seed] = map_pair_deltas.get(seed, 0) + (
                    int(reply_preference == outcome_preference)
                    - int(reply_preference == -outcome_preference)
                )
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
    paired_preference_maps = paired_comparison_summary(
        sum(delta > 0 for delta in map_pair_deltas.values()),
        sum(delta < 0 for delta in map_pair_deltas.values()),
        sum(delta == 0 for delta in map_pair_deltas.values()),
    )
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
        "independent_map_pairwise_alignment": paired_preference_maps,
    }


def audit(
    paths: list[Path], outcome_probe: OutcomeProbe = "opponent"
) -> dict[str, object]:
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
        if (
            outcome_probe == "teacher"
            and report.get("teacher_continuations") is not True
        ):
            raise ValueError("teacher outcome audit requires teacher continuations")
    if len({report["opponent_search_nodes"] for report in reports}) != 1:
        raise ValueError("reply slate audit requires a shared opponent search budget")
    rollins = {report.get("reply_rollin", False) for report in reports}
    if len(rollins) != 1:
        raise ValueError("reply slate audit requires a shared roll-in policy")
    records = [
        cast(dict[str, object], record)
        for report in reports
        for record in cast(list[object], report["records"])
    ]
    return {
        "kind": "procedural_duel_reply_slate_audit",
        "source_files": [{"name": path.name, "sha256": digest(path)} for path in paths],
        "opponent_search_nodes": reports[0]["opponent_search_nodes"],
        "rollin": "reply_search" if rollins.pop() else "greedy",
        "outcome_probe": outcome_probe,
        "qualification": (
            "conditional terminal labels under persistent reply-search on both seats"
            if outcome_probe == "teacher"
            else "conditional terminal labels under greedy root and searched opponents"
        )
        + "; candidate pairs within maps are correlated",
        **summarize(records, outcome_probe),
        **(
            {"probe_disagreement": compare_probes(records)}
            if outcome_probe == "teacher"
            else {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, action="append", type=Path)
    parser.add_argument(
        "--outcome-probe", choices=("opponent", "teacher"), default="opponent"
    )
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.input, arguments.outcome_probe), sort_keys=True))


if __name__ == "__main__":
    main()
