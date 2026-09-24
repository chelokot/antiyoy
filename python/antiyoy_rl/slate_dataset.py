from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, cast

import numpy as np

from . import ProceduralConfig, VectorEnv
from .model import select_environments
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
    followup_scores: np.ndarray | None = None


@dataclass(frozen=True)
class ReplayedSlatePosition:
    seed: int
    record: dict[str, object]
    root: VectorEnv
    post_turn: dict[str, np.ndarray]


@dataclass(frozen=True)
class CorrectiveSlatePosition:
    seed: int
    seat: int
    round: int
    decisions: OpponentDecisions
    student_action_indices: np.ndarray


def step_action_index(environment: VectorEnv, index: int) -> bool:
    result = environment.step(np.asarray([index], dtype=np.uint64))
    return bool(result["truncated"][0])


def replay_slate_positions(
    dataset: dict[str, object],
) -> Iterator[ReplayedSlatePosition]:
    generator = cast(dict[str, int], dataset["generator"])
    if dataset["schema_version"] != 1 or generator["players"] != 2:
        raise ValueError("slate replay requires version-one two-player data")
    by_seed: dict[int, list[dict[str, object]]] = {}
    for record in cast(list[dict[str, object]], dataset["records"]):
        by_seed.setdefault(cast(int, record["seed"]), []).append(record)
    for seed, records in sorted(by_seed.items()):
        map_config = ProceduralConfig(
            width=generator["width"],
            height=generator["height"],
            players=2,
            seed=seed,
            land_density_per_million=generator["land_density_per_million"],
            starting_province_size=generator["starting_province_size"],
            starting_money=generator["starting_money"],
            tree_density_per_million=generator["tree_density_per_million"],
            neutral_tower_density_per_million=generator[
                "neutral_tower_density_per_million"
            ],
            neutral_capital_density_per_million=generator[
                "neutral_capital_density_per_million"
            ],
            grave_density_per_million=generator["grave_density_per_million"],
            schema_version=generator["schema_version"],
        )
        environment = VectorEnv.procedural(
            1,
            map_config,
            action_limit=cast(int, dataset["action_limit"]),
            profile=cast(str, dataset["rules"]),
        )
        applied = 0
        for record in records:
            rollin = cast(list[int], record["rollin_indices"])
            for index in rollin[applied:]:
                if step_action_index(environment, index):
                    raise ValueError("searched roll-in was action-limit adjudicated")
            applied = len(rollin)
            observed = environment.observe()
            if (
                int(observed["active_players"][0]) != record["seat"]
                or int(observed["rounds"][0]) != record["round"]
            ):
                raise ValueError("sampled root state disagrees with replayed roll-in")
            expected, _ = model_observation(
                cast(dict[str, object], record["post_turn"])
            )
            yield ReplayedSlatePosition(seed, record, environment, expected)


def replay_slate_candidate(
    position: ReplayedSlatePosition, candidate: int
) -> VectorEnv:
    branch = position.root.fork(np.asarray([0], dtype=np.uint64))
    indices = cast(list[list[int]], position.record["candidate_action_indices"])
    for index in indices[candidate]:
        if step_action_index(branch, index):
            raise ValueError("searched root candidate was adjudicated")
    actual = branch.observe()
    expected = select_environments(position.post_turn, [candidate])
    if any(not np.array_equal(actual[key], values) for key, values in expected.items()):
        raise ValueError("searched candidate state disagrees with replay")
    return branch


def load_corrective_positions(path: Path) -> list[CorrectiveSlatePosition]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        report = cast(dict[str, object], json.load(source))
    if (
        report["schema_version"] != 1
        or report["kind"] != "procedural_duel_corrective_opponent_decisions"
    ):
        raise ValueError("corrective slate dataset has an unsupported schema")
    positions = []
    for record in cast(list[dict[str, object]], report["records"]):
        raw = cast(dict[str, list[int]] | None, record["observation"])
        observation = (
            {key: np.asarray(values, dtype=np.int64) for key, values in raw.items()}
            if raw is not None
            else None
        )
        labels = np.asarray(record["teacher_action_indices"], dtype=np.int64)
        chosen = np.asarray(record["student_action_indices"], dtype=np.int64)
        offsets = np.asarray(record["candidate_offsets"], dtype=np.int64)
        rules = tuple(cast(list[str], record["rules_json"]))
        count = len(labels)
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != count
            or np.any(np.diff(offsets) < 0)
            or len(chosen) != count
            or len(rules) != count
            or (observation is None) != (count == 0)
        ):
            raise ValueError("corrective candidate and decision counts differ")
        if observation is None:
            continue
        action_counts = np.diff(observation["action_offsets"])
        if (
            len(observation["widths"]) != count
            or len(observation["action_offsets"]) != count + 1
            or len(action_counts) != count
            or np.any(labels < 0)
            or np.any(labels >= action_counts)
            or np.any(chosen < 0)
            or np.any(chosen >= action_counts)
        ):
            raise ValueError("corrective action index is not locally legal")
        positions.append(
            CorrectiveSlatePosition(
                seed=cast(int, record["seed"]),
                seat=cast(int, record["seat"]),
                round=cast(int, record["round"]),
                decisions=OpponentDecisions(observation, rules, offsets, labels),
                student_action_indices=chosen,
            )
        )
    if len(cast(list[object], report["records"])) != report["positions"]:
        raise ValueError("corrective dataset position count disagrees with summary")
    return positions


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
    followup_nodes = cast(int | None, report.get("followup_search_nodes"))
    if followup_nodes is not None and followup_nodes <= 0:
        raise ValueError("teacher slate followup search node count must be positive")
    positions = []
    for record in cast(list[dict[str, object]], report["records"]):
        observation, rules = model_observation(
            cast(dict[str, object], record["post_turn"])
        )
        static = np.asarray(record["static_scores"], dtype=np.int64)
        reply = np.asarray(record["reply_scores"], dtype=np.float64)
        raw_followup = cast(list[int] | None, record.get("followup_scores"))
        if (followup_nodes is None) != (raw_followup is None):
            raise ValueError("teacher slate followup scores disagree with search configuration")
        followup = (
            np.asarray(raw_followup, dtype=np.float64)
            if raw_followup is not None
            else None
        )
        actions = cast(list[list[object]], record["actions"])
        count = len(static)
        if not (count == len(reply) == len(actions) == len(observation["widths"])):
            raise ValueError("teacher slate candidate and observation counts differ")
        if followup is not None and len(followup) != count:
            raise ValueError("teacher slate followup score count differs")
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
        selection_scores = reply if followup is None else followup
        chosen = max(
            range(count),
            key=lambda index: (selection_scores[index], static[index], -index),
        )
        if chosen != record["selected_index"]:
            raise ValueError("teacher slate selection disagrees with search scores")
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
                followup_scores=followup,
            )
        )
    if len(positions) != report["positions"]:
        raise ValueError("teacher slate position count disagrees with summary")
    return positions
