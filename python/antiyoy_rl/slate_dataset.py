from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from .turn_credit import model_observation


@dataclass(frozen=True)
class OpponentDecisions:
    observation: dict[str, np.ndarray]
    rules_json: tuple[str, ...]
    candidate_offsets: np.ndarray
    action_indices: np.ndarray


@dataclass(frozen=True)
class TeacherSlatePosition:
    seed: int
    seat: int
    round: int
    post_turn: dict[str, np.ndarray]
    post_turn_rules_json: tuple[str, ...]
    static_scores: np.ndarray
    outcome_scores: np.ndarray
    opponent_reply_scores: np.ndarray
    slate_indices: tuple[int, ...]
    slate_first_actions: tuple[str, ...]
    opponent_actions: tuple[tuple[str, ...], ...] | None
    post_reply: dict[str, np.ndarray] | None
    opponent_decisions: OpponentDecisions | None = None


def load_teacher_slates(path: Path) -> list[TeacherSlatePosition]:
    source = (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(encoding="utf-8")
    )
    with source:
        report = cast(dict[str, object], json.load(source))
    generator = cast(dict[str, object], report["generator"])
    if report["schema_version"] != 1 or generator["players"] != 2:
        raise ValueError("teacher slate loader requires version-one two-player data")
    positions = []
    for record in cast(list[dict[str, object]], report["records"]):
        observation, rules = model_observation(
            cast(dict[str, object], record["post_turn"])
        )
        static = np.asarray(record["static_scores"], dtype=np.int64)
        reply = np.asarray(record["reply_scores"], dtype=np.float64)
        actions = cast(list[list[object]], record["actions"])
        count = len(static)
        if not (count == len(reply) == len(actions) == len(observation["widths"])):
            raise ValueError("teacher slate candidate and observation counts differ")
        opponent_actions = cast(
            list[list[object]] | None, record.get("opponent_actions")
        )
        if opponent_actions is not None and len(opponent_actions) != count:
            raise ValueError("teacher slate opponent action counts differ")
        post_reply = cast(dict[str, object] | None, record.get("post_reply"))
        reply_observation = None
        if post_reply is not None:
            reply_observation, reply_rules = model_observation(post_reply)
            if len(reply_observation["widths"]) != count:
                raise ValueError("teacher slate opponent observation counts differ")
            if reply_rules != rules:
                raise ValueError("teacher slate opponent observation rules differ")
        exported_decisions = cast(
            dict[str, object] | None, record.get("opponent_decisions")
        )
        decisions = None
        if exported_decisions is not None:
            decision_observation, decision_rules = model_observation(
                cast(dict[str, object], exported_decisions["observation"])
            )
            offsets = np.asarray(
                exported_decisions["candidate_offsets"], dtype=np.int64
            )
            indices = np.asarray(exported_decisions["action_indices"], dtype=np.int64)
            decision_count = len(decision_observation["widths"])
            if (
                len(offsets) != count + 1
                or offsets[0] != 0
                or offsets[-1] != decision_count
                or np.any(np.diff(offsets) < 0)
                or len(indices) != decision_count
                or len(decision_rules) != decision_count
                or len(decision_observation["action_offsets"]) != decision_count + 1
            ):
                raise ValueError("teacher slate opponent decision counts differ")
            action_counts = np.diff(decision_observation["action_offsets"])
            if np.any(indices < 0) or np.any(indices >= action_counts):
                raise ValueError("teacher slate opponent decision action is not legal")
            if opponent_actions is not None and any(
                offsets[index + 1] - offsets[index] != len(plan)
                for index, plan in enumerate(opponent_actions)
            ):
                raise ValueError("teacher slate opponent plan and decisions differ")
            decisions = OpponentDecisions(
                observation=decision_observation,
                rules_json=decision_rules,
                candidate_offsets=offsets,
                action_indices=indices,
            )
        chosen = max(
            range(count),
            key=lambda index: (reply[index], static[index], -index),
        )
        if chosen != record["selected_index"]:
            raise ValueError("teacher slate selection disagrees with reply scores")
        positions.append(
            TeacherSlatePosition(
                seed=cast(int, record["seed"]),
                seat=cast(int, record["seat"]),
                round=cast(int, record["round"]),
                post_turn=observation,
                post_turn_rules_json=rules,
                static_scores=static,
                outcome_scores=np.full(count, -1, dtype=np.int8),
                opponent_reply_scores=reply,
                slate_indices=tuple(range(count)),
                slate_first_actions=tuple(
                    json.dumps(candidate[0], sort_keys=True) for candidate in actions
                ),
                opponent_actions=(
                    tuple(
                        tuple(
                            json.dumps(action, sort_keys=True) for action in candidate
                        )
                        for candidate in opponent_actions
                    )
                    if opponent_actions is not None
                    else None
                ),
                post_reply=reply_observation,
                opponent_decisions=decisions,
            )
        )
    if len(positions) != report["positions"]:
        raise ValueError("teacher slate position count disagrees with summary")
    return positions
