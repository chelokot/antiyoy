from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from antiyoy_rl.model import select_environments
from antiyoy_rl.slate_dataset import replay_slate_positions

from .audit_three_turn_plan_feasibility import (
    action_log_probabilities,
    candidate_decisions,
    mean_plan_log_probabilities,
    selected_plan,
)
from .audit_three_turn_endturn_context import end_turn_context
from .build_bundle import digest
from .evaluate import load_policy
from .train_three_turn_plan import FIT_MAPS, FIT_SEED, SOURCE_SHA256, checked_dataset


@dataclass(frozen=True)
class OverrideMargin:
    seed: int
    seat: int
    source_choice: int
    student_choice: int
    teacher_choice: int
    source_margin: float
    student_margin: float
    selected_length: int
    static_length: int
    same_first_action: bool
    source_teacher_end_log_prob: float
    source_static_end_log_prob: float
    student_teacher_end_log_prob: float
    student_static_end_log_prob: float
    source_first_action_gap: float
    student_first_action_gap: float


@dataclass(frozen=True)
class SharedStateStop:
    seed: int
    seat: int
    next_action_kind: int
    legal_non_end_actions: int
    legal_moves: int
    ready_units: int
    source_stop_advantage: float
    student_stop_advantage: float
    teacher_score_advantage: int


def shared_state_stop_indices(
    plans: list[list[int]], teacher: int, offsets: np.ndarray
) -> tuple[int, int] | None:
    selected = plans[teacher]
    static = plans[0]
    prefix_length = len(selected) - 1
    if len(selected) >= len(static) or selected[:-1] != static[:prefix_length]:
        return None
    return int(offsets[teacher + 1] - 1), int(offsets[0] + prefix_length)


def summarize_shared_stops(rows: list[SharedStateStop]) -> dict[str, object]:
    if not rows:
        return {"positions": 0, "independent_maps": 0}
    return {
        "positions": len(rows),
        "independent_maps": len({row.seed for row in rows}),
        "next_action_kinds": dict(
            sorted(Counter(row.next_action_kind for row in rows).items())
        ),
        "no_legal_non_end_actions": sum(row.legal_non_end_actions == 0 for row in rows),
        "legal_non_end_actions_median": float(
            np.median([row.legal_non_end_actions for row in rows])
        ),
        "legal_moves_median": float(np.median([row.legal_moves for row in rows])),
        "ready_units_median": float(np.median([row.ready_units for row in rows])),
        "source_stop_advantage_median": float(
            np.median([row.source_stop_advantage for row in rows])
        ),
        "student_stop_advantage_median": float(
            np.median([row.student_stop_advantage for row in rows])
        ),
        "source_prefers_stop": sum(row.source_stop_advantage > 0 for row in rows),
        "student_prefers_stop": sum(row.student_stop_advantage > 0 for row in rows),
        "teacher_score_advantage_median": float(
            np.median([row.teacher_score_advantage for row in rows])
        ),
        "teacher_score_advantage_positive": sum(
            row.teacher_score_advantage > 0 for row in rows
        ),
    }


def length_relation(row: OverrideMargin) -> str:
    if row.selected_length > row.static_length:
        return "longer"
    if row.selected_length < row.static_length:
        return "shorter"
    return "equal"


def summarize_overrides(rows: list[OverrideMargin]) -> dict[str, object]:
    source_margins = np.asarray([row.source_margin for row in rows])
    student_margins = np.asarray([row.student_margin for row in rows])
    length_deltas = np.asarray(
        [row.selected_length - row.static_length for row in rows]
    )
    return {
        "positions": len(rows),
        "independent_maps": len({row.seed for row in rows}),
        "source_selects_static": sum(row.source_choice == 0 for row in rows),
        "student_selects_static": sum(row.student_choice == 0 for row in rows),
        "source_selects_teacher": sum(
            row.source_choice == row.teacher_choice for row in rows
        ),
        "student_selects_teacher": sum(
            row.student_choice == row.teacher_choice for row in rows
        ),
        "source_teacher_margin_quantiles": np.quantile(
            source_margins, [0.1, 0.5, 0.9]
        ).tolist(),
        "student_teacher_margin_quantiles": np.quantile(
            student_margins, [0.1, 0.5, 0.9]
        ).tolist(),
        "margin_increased": int(np.sum(student_margins > source_margins)),
        "margin_decreased": int(np.sum(student_margins < source_margins)),
        "margin_same": int(np.sum(student_margins == source_margins)),
        "student_positive_margin": int(np.sum(student_margins > 0)),
        "source_positive_margin": int(np.sum(source_margins > 0)),
        "selected_longer": int(np.sum(length_deltas > 0)),
        "selected_same_length": int(np.sum(length_deltas == 0)),
        "selected_shorter": int(np.sum(length_deltas < 0)),
        "mean_selected_actions": float(np.mean([row.selected_length for row in rows])),
        "mean_static_actions": float(np.mean([row.static_length for row in rows])),
        "selected_action_counts": dict(
            sorted(Counter(row.selected_length for row in rows).items())
        ),
        "static_action_counts": dict(
            sorted(Counter(row.static_length for row in rows).items())
        ),
        "source_teacher_end_logprob_median": float(
            np.median([row.source_teacher_end_log_prob for row in rows])
        ),
        "source_static_end_logprob_median": float(
            np.median([row.source_static_end_log_prob for row in rows])
        ),
        "student_teacher_end_logprob_median": float(
            np.median([row.student_teacher_end_log_prob for row in rows])
        ),
        "student_static_end_logprob_median": float(
            np.median([row.student_static_end_log_prob for row in rows])
        ),
        "source_early_end_gap_median": float(
            np.median(
                [
                    row.source_teacher_end_log_prob - row.source_static_end_log_prob
                    for row in rows
                ]
            )
        ),
        "student_early_end_gap_median": float(
            np.median(
                [
                    row.student_teacher_end_log_prob - row.student_static_end_log_prob
                    for row in rows
                ]
            )
        ),
        "source_first_action_gap_median": float(
            np.median([row.source_first_action_gap for row in rows])
        ),
        "student_first_action_gap_median": float(
            np.median([row.student_first_action_gap for row in rows])
        ),
    }


def audit(
    fit_path: Path, source_path: Path, checkpoint_path: Path
) -> dict[str, object]:
    if digest(source_path) != SOURCE_SHA256:
        raise ValueError("frozen source checkpoint disagrees with the fixed protocol")
    dataset = checked_dataset(fit_path, FIT_SEED, FIT_MAPS)
    torch.set_num_threads(1)
    source, config = load_policy(
        source_path,
        torch.device("cpu"),
        profile="classic_generic_2022",
        generator="procedural_v1",
        players=2,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        state["kind"] != "three_turn_whole_plan_action_head"
        or state["source_sha256"] != SOURCE_SHA256
        or state["selected_expert"] != config["selected_expert"]
    ):
        raise ValueError("student checkpoint disagrees with the fixed protocol")
    student = copy.deepcopy(source)
    student.action_head.load_state_dict(state["action_head"])
    if student.action_residual is not None:
        student.action_residual.load_state_dict(state["action_residual"])
    student.eval()
    rows = []
    shared_stops = []
    sampled_positions = 0
    for position in replay_slate_positions(dataset):
        sampled_positions += 1
        record = position.record
        teacher = cast(int, record["selected_index"])
        if teacher == 0:
            continue
        observations, actions, offsets = candidate_decisions(position)
        rules = position.root.rules_json()
        source_actions = action_log_probabilities(source, observations, actions, rules)
        student_actions = action_log_probabilities(
            student, observations, actions, rules
        )
        source_scores = mean_plan_log_probabilities(source_actions, offsets)
        student_scores = mean_plan_log_probabilities(student_actions, offsets)
        static_scores = cast(list[int], record["static_scores"])
        plans = cast(list[list[int]], record["candidate_action_indices"])
        serialized = cast(list[list[object]], record["actions"])
        if serialized[0][-1] != "EndTurn" or serialized[teacher][-1] != "EndTurn":
            raise ValueError("completed turn plan lacks EndTurn")
        static_source_actions = source_actions[offsets[0] : offsets[1]]
        teacher_source_actions = source_actions[offsets[teacher] : offsets[teacher + 1]]
        static_student_actions = student_actions[offsets[0] : offsets[1]]
        teacher_student_actions = student_actions[
            offsets[teacher] : offsets[teacher + 1]
        ]
        stop_indices = shared_state_stop_indices(plans, teacher, offsets)
        if stop_indices is not None:
            stop_decision, continue_decision = stop_indices
            stop_state = select_environments(observations, [stop_decision])
            continue_state = select_environments(observations, [continue_decision])
            if any(
                not np.array_equal(stop_state[key], value)
                for key, value in continue_state.items()
            ):
                raise ValueError("identical action prefix produced different states")
            action_start = int(observations["action_offsets"][stop_decision])
            action_end = int(observations["action_offsets"][stop_decision + 1])
            kinds = observations["action_kinds"][action_start:action_end]
            context = end_turn_context(
                observations, stop_decision, int(actions[stop_decision])
            )
            next_kind = int(kinds[actions[continue_decision]])
            if next_kind == 0 or context.active_player != record["seat"]:
                raise ValueError("shared-prefix stop and continuation labels disagree")
            followup_scores = cast(list[int], record["followup_scores"])
            shared_stops.append(
                SharedStateStop(
                    seed=position.seed,
                    seat=cast(int, record["seat"]),
                    next_action_kind=next_kind,
                    legal_non_end_actions=context.other_legal_actions,
                    legal_moves=context.other_legal_action_kinds[1],
                    ready_units=context.ready_owned_units,
                    source_stop_advantage=float(
                        source_actions[stop_decision]
                        - source_actions[continue_decision]
                    ),
                    student_stop_advantage=float(
                        student_actions[stop_decision]
                        - student_actions[continue_decision]
                    ),
                    teacher_score_advantage=followup_scores[teacher]
                    - followup_scores[0],
                )
            )
        rows.append(
            OverrideMargin(
                seed=position.seed,
                seat=cast(int, record["seat"]),
                source_choice=selected_plan(source_scores, static_scores),
                student_choice=selected_plan(student_scores, static_scores),
                teacher_choice=teacher,
                source_margin=float(source_scores[teacher] - source_scores[0]),
                student_margin=float(student_scores[teacher] - student_scores[0]),
                selected_length=len(plans[teacher]),
                static_length=len(plans[0]),
                same_first_action=plans[teacher][0] == plans[0][0],
                source_teacher_end_log_prob=float(teacher_source_actions[-1]),
                source_static_end_log_prob=float(static_source_actions[-1]),
                student_teacher_end_log_prob=float(teacher_student_actions[-1]),
                student_static_end_log_prob=float(static_student_actions[-1]),
                source_first_action_gap=float(
                    teacher_source_actions[0] - static_source_actions[0]
                ),
                student_first_action_gap=float(
                    teacher_student_actions[0] - static_student_actions[0]
                ),
            )
        )
    return {
        "kind": "three_turn_whole_plan_fit_only_margin_audit",
        "fit_sha256": digest(fit_path),
        "source_sha256": digest(source_path),
        "student_sha256": digest(checkpoint_path),
        "sampled_positions": sampled_positions,
        "teacher_override_positions": summarize_overrides(rows),
        "by_seat": {
            seat: summarize_overrides([row for row in rows if row.seat == seat])
            for seat in (0, 1)
        },
        "by_first_action_relation": {
            relation: summarize_overrides(
                [row for row in rows if row.same_first_action == same]
            )
            for relation, same in (("same", True), ("different", False))
        },
        "by_plan_length_relation": {
            relation: summarize_overrides(
                [row for row in rows if length_relation(row) == relation]
            )
            for relation in ("longer", "equal", "shorter")
            if any(length_relation(row) == relation for row in rows)
        },
        "identical_state_stop_vs_continue": summarize_shared_stops(shared_stops),
        "identical_state_stop_vs_continue_by_seat": {
            seat: summarize_shared_stops(
                [row for row in shared_stops if row.seat == seat]
            )
            for seat in (0, 1)
        },
        "qualification": "Read-only fit-map diagnosis; no fresh validation, game-strength, or Elo claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fit", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(arguments.fit, arguments.source, arguments.checkpoint), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
