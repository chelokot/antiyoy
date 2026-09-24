from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from antiyoy_rl.model import ACTION_KIND_NAMES
from antiyoy_rl.slate_dataset import replay_slate_positions

from .audit_three_turn_plan_feasibility import candidate_decisions
from .build_bundle import digest
from .train_three_turn_plan import FIT_MAPS, FIT_SEED, checked_dataset


@dataclass(frozen=True)
class EndTurnContext:
    active_player: int
    round: int
    other_legal_actions: int
    other_legal_action_kinds: tuple[int, ...]
    ready_owned_units: int
    owned_province_money: int


def end_turn_context(
    observation: dict[str, np.ndarray], decision: int, action_index: int
) -> EndTurnContext:
    action_start, action_end = observation["action_offsets"][decision : decision + 2]
    kinds = observation["action_kinds"][action_start:action_end]
    if int(kinds[action_index]) != 0 or int(np.count_nonzero(kinds == 0)) != 1:
        raise ValueError("completed plan does not select its unique legal EndTurn")
    active = observation["active_players"][decision]
    cell_start, cell_end = observation["cell_offsets"][decision : decision + 2]
    owners = observation["owners"][cell_start:cell_end]
    strengths = observation["unit_strengths"][cell_start:cell_end]
    ready = observation["ready"][cell_start:cell_end]
    province_start, province_end = observation["province_offsets"][
        decision : decision + 2
    ]
    province_owners = observation["province_owners"][province_start:province_end]
    province_money = observation["province_money"][province_start:province_end]
    return EndTurnContext(
        active_player=int(active),
        round=int(observation["rounds"][decision]),
        other_legal_actions=len(kinds) - 1,
        other_legal_action_kinds=tuple(
            int(np.count_nonzero(kinds == code))
            for code in range(len(ACTION_KIND_NAMES))
        ),
        ready_owned_units=int(
            np.count_nonzero((owners == active) & (strengths > 0) & (ready > 0))
        ),
        owned_province_money=int(np.sum(province_money[province_owners == active])),
    )


def summarize(contexts: list[EndTurnContext]) -> dict[str, object]:
    other_counts = np.asarray([row.other_legal_actions for row in contexts])
    return {
        "positions": len(contexts),
        "forced_endturn": int(np.count_nonzero(other_counts == 0)),
        "optional_endturn": int(np.count_nonzero(other_counts > 0)),
        "other_legal_actions_quantiles_10_50_90": np.quantile(
            other_counts, [0.1, 0.5, 0.9]
        ).tolist(),
        "ready_owned_units_median": float(
            np.median([row.ready_owned_units for row in contexts])
        ),
        "owned_province_money_median": float(
            np.median([row.owned_province_money for row in contexts])
        ),
        "positions_with_action_kind": {
            name: sum(row.other_legal_action_kinds[code] > 0 for row in contexts)
            for code, name in enumerate(ACTION_KIND_NAMES)
            if code > 0
        },
    }


def summarize_score_gaps(gaps: list[tuple[int, int]]) -> dict[str, object]:
    immediate = np.asarray([gap[0] for gap in gaps])
    followup = np.asarray([gap[1] for gap in gaps])
    return {
        "immediate_teacher_minus_static_quantiles_10_50_90": np.quantile(
            immediate, [0.1, 0.5, 0.9]
        ).tolist(),
        "immediate_teacher_better_equal_worse": [
            int(np.count_nonzero(immediate > 0)),
            int(np.count_nonzero(immediate == 0)),
            int(np.count_nonzero(immediate < 0)),
        ],
        "followup_teacher_minus_static_quantiles_10_50_90": np.quantile(
            followup, [0.1, 0.5, 0.9]
        ).tolist(),
        "followup_teacher_better_equal_worse": [
            int(np.count_nonzero(followup > 0)),
            int(np.count_nonzero(followup == 0)),
            int(np.count_nonzero(followup < 0)),
        ],
    }


def audit(fit_path: Path) -> dict[str, object]:
    dataset = checked_dataset(fit_path, FIT_SEED, FIT_MAPS)
    contexts: dict[str, list[EndTurnContext]] = {
        "teacher": [],
        "static": [],
    }
    per_seat: dict[int, dict[str, list[EndTurnContext]]] = {
        seat: {"teacher": [], "static": []} for seat in (0, 1)
    }
    sampled_positions = 0
    map_counts: Counter[int] = Counter()
    score_gaps: list[tuple[int, int]] = []
    for position in replay_slate_positions(dataset):
        sampled_positions += 1
        record = position.record
        teacher = cast(int, record["selected_index"])
        if teacher == 0:
            continue
        observation, indices, offsets = candidate_decisions(position)
        seat = cast(int, record["seat"])
        map_counts[position.seed] += 1
        static_scores = cast(list[int], record["static_scores"])
        followup_scores = cast(list[int], record["followup_scores"])
        score_gaps.append(
            (
                static_scores[teacher] - static_scores[0],
                followup_scores[teacher] - followup_scores[0],
            )
        )
        for role, candidate in (("teacher", teacher), ("static", 0)):
            decision = int(offsets[candidate + 1] - 1)
            context = end_turn_context(observation, decision, int(indices[decision]))
            if context.active_player != seat or context.round != record["round"]:
                raise ValueError("EndTurn state disagrees with sampled root turn")
            contexts[role].append(context)
            per_seat[seat][role].append(context)
    return {
        "kind": "three_turn_fit_only_endturn_context_audit",
        "fit_sha256": digest(fit_path),
        "sampled_positions": sampled_positions,
        "override_positions": len(contexts["teacher"]),
        "independent_maps_with_overrides": len(map_counts),
        "contexts": {role: summarize(rows) for role, rows in contexts.items()},
        "score_gaps": summarize_score_gaps(score_gaps),
        "by_seat": {
            seat: {role: summarize(rows) for role, rows in roles.items()}
            for seat, roles in per_seat.items()
        },
        "qualification": "Read-only fit-map legality audit; no held-out strength or causal claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.fit), sort_keys=True))


if __name__ == "__main__":
    main()
