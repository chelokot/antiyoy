from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np


ACTION_KINDS = {
    "EndTurn": 0,
    "Move": 1,
    "Recruit": 2,
    "Build": 3,
    "PlantTree": 4,
    "Diplomacy": 5,
}

OBSERVATION_FIELDS = (
    "cell_offsets",
    "province_offsets",
    "action_offsets",
    "relation_offsets",
    "widths",
    "heights",
    "active_players",
    "player_counts",
    "rounds",
    "playable",
    "visible",
    "owners",
    "objects",
    "unit_strengths",
    "ready",
    "defenses",
    "province_ids",
    "province_owners",
    "province_money",
    "province_profit",
    "province_capitals",
    "province_sizes",
    "relations",
    "proposals",
)


class ExportedAction(TypedDict):
    kind: str
    source: int
    target: int
    parameter: int


@dataclass(frozen=True)
class TurnCreditPosition:
    seed: int
    seat: int
    round: int
    root: dict[str, np.ndarray]
    post_turn: dict[str, np.ndarray]
    root_rules_json: tuple[str, ...]
    post_turn_rules_json: tuple[str, ...]
    outcome_scores: np.ndarray
    static_scores: np.ndarray
    search_index: int
    greedy_index: int
    root_search_scores: np.ndarray | None = None
    root_search_nodes: int | None = None
    opponent_search_scores: np.ndarray | None = None
    opponent_search_nodes: int | None = None

    @property
    def complete(self) -> np.ndarray:
        return self.outcome_scores >= 0


def model_observation(
    exported: dict[str, object],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    actions = cast(list[ExportedAction], exported["actions"])
    rules = cast(list[dict[str, object]], exported["rules"])
    observation = {
        key: np.asarray(exported[key], dtype=np.int64) for key in OBSERVATION_FIELDS
    }
    observation["action_kinds"] = np.asarray(
        [ACTION_KINDS[action["kind"]] for action in actions], dtype=np.int64
    )
    observation["action_sources"] = np.asarray(
        [action["source"] for action in actions], dtype=np.int64
    )
    observation["action_targets"] = np.asarray(
        [action["target"] for action in actions], dtype=np.int64
    )
    observation["action_parameters"] = np.asarray(
        [action["parameter"] for action in actions], dtype=np.int64
    )
    rules_json = tuple(json.dumps(rule, separators=(",", ":")) for rule in rules)
    return observation, rules_json


def outcome_score(continuation: dict[str, object], seat: int) -> int:
    if continuation["truncated"] is True:
        return -1
    winner = continuation["winner"]
    if winner is None:
        return 1
    return 2 if winner == seat else 0


def continuation_probe_scores(
    record: dict[str, object],
    report: dict[str, object],
    seat: int,
    state_count: int,
    nodes_field: str,
    continuations_field: str,
) -> np.ndarray | None:
    continuations = cast(
        list[dict[str, object]] | None,
        record.get(continuations_field),
    )
    if (continuations is None) != (report.get(nodes_field) is None):
        raise ValueError(f"{continuations_field} configuration and labels disagree")
    if continuations is None:
        return None
    if len(continuations) != state_count:
        raise ValueError(f"{continuations_field} probe count differs from end states")
    return np.asarray(
        [outcome_score(continuation, seat) for continuation in continuations],
        dtype=np.int8,
    )


def load_turn_credit_positions(path: Path) -> list[TurnCreditPosition]:
    if path.suffix == ".gz":
        source = gzip.open(path, "rt", encoding="utf-8")
    else:
        source = path.open(encoding="utf-8")
    with source:
        report = cast(dict[str, object], json.load(source))
    if report["include_observations"] is not True:
        raise ValueError("turn-credit report requires --include-observations")
    positions = []
    for record in cast(list[dict[str, object]], report["records"]):
        observations = cast(dict[str, dict[str, object]], record["observations"])
        root, root_rules = model_observation(observations["root"])
        post_turn, post_rules = model_observation(observations["post_turn"])
        state_count = len(post_turn["widths"])
        if state_count != record["distinct_end_states"]:
            raise ValueError(
                "post-turn observation count differs from distinct end states"
            )
        if len(root["widths"]) != 1 or len(post_rules) != state_count:
            raise ValueError("turn-credit observations have inconsistent batch sizes")
        scores = np.full(state_count, -1, dtype=np.int8)
        static_scores = np.zeros(state_count, dtype=np.int64)
        seen = np.zeros(state_count, dtype=np.bool_)
        greedy = cast(dict[str, object], record["greedy"])
        search = cast(dict[str, object], record["search"])
        branches = [greedy, search]
        alternatives = cast(list[dict[str, object]], record["alternatives"])
        candidates = cast(list[dict[str, object]], record["beam_candidates"])
        branches.extend(
            cast(dict[str, object], item["branch"]) for item in alternatives
        )
        branches.extend(cast(dict[str, object], item["branch"]) for item in candidates)
        seat = cast(int, record["seat"])
        for branch in branches:
            index = cast(int, branch["state_index"])
            continuation = cast(dict[str, object], branch["continuation"])
            score = outcome_score(continuation, seat)
            if seen[index] and scores[index] != score:
                raise ValueError(
                    "branches sharing a post-turn state disagree on outcome"
                )
            static_score = cast(int, branch["static_score"])
            if seen[index] and static_scores[index] != static_score:
                raise ValueError(
                    "branches sharing a post-turn state disagree on static score"
                )
            scores[index] = score
            static_scores[index] = static_score
            seen[index] = True
        if not seen.all():
            raise ValueError("a post-turn state has no outcome label")
        root_search_scores = continuation_probe_scores(
            record,
            report,
            seat,
            state_count,
            "root_search_nodes",
            "root_search_continuations",
        )
        opponent_search_scores = continuation_probe_scores(
            record,
            report,
            seat,
            state_count,
            "opponent_search_nodes",
            "opponent_search_continuations",
        )
        positions.append(
            TurnCreditPosition(
                seed=cast(int, record["seed"]),
                seat=seat,
                round=cast(int, record["round"]),
                root=root,
                post_turn=post_turn,
                root_rules_json=root_rules,
                post_turn_rules_json=post_rules,
                outcome_scores=scores,
                static_scores=static_scores,
                search_index=cast(int, search["state_index"]),
                greedy_index=cast(int, greedy["state_index"]),
                root_search_scores=root_search_scores,
                root_search_nodes=cast(int | None, report.get("root_search_nodes")),
                opponent_search_scores=opponent_search_scores,
                opponent_search_nodes=cast(
                    int | None, report.get("opponent_search_nodes")
                ),
            )
        )
    return positions
