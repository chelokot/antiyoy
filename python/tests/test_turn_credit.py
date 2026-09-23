import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from antiyoy_rl.turn_credit import (
    OBSERVATION_FIELDS,
    load_turn_credit_positions,
    model_observation,
)


def observation(states: int) -> dict[str, object]:
    exported: dict[str, object] = {field: [] for field in OBSERVATION_FIELDS}
    for field in ("cell_offsets", "province_offsets", "relation_offsets"):
        exported[field] = [0] * (states + 1)
    exported["action_offsets"] = list(range(states + 1))
    exported["widths"] = [7] * states
    exported["heights"] = [5] * states
    exported["active_players"] = [1] * states
    exported["player_counts"] = [3] * states
    exported["rounds"] = [4] * states
    exported["actions"] = [
        {"kind": "EndTurn", "source": 65535, "target": 65535, "parameter": 0}
        for _ in range(states)
    ]
    exported["rules"] = [{"profile": "ClassicGeneric"}] * states
    return exported


def branch(
    index: int, winner: int | None, truncated: bool = False
) -> dict[str, object]:
    return {
        "state_index": index,
        "static_score": 100 + index,
        "actions": [{"Move": index}, "EndTurn"],
        "continuation": {"winner": winner, "truncated": truncated},
    }


def report() -> dict[str, object]:
    return {
        "include_observations": True,
        "records": [
            {
                "seed": 71,
                "seat": 0,
                "round": 4,
                "distinct_end_states": 3,
                "observations": {"root": observation(1), "post_turn": observation(3)},
                "greedy": branch(0, 1),
                "search": branch(1, 0),
                "alternatives": [{"branch": branch(1, 0)}],
                "beam_candidates": [{"branch": branch(2, None, truncated=True)}],
            }
        ],
    }


def write_report(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_load_turn_credit_preserves_state_labels_and_censoring(tmp_path: Path) -> None:
    path = tmp_path / "credit.json"
    write_report(path, report())
    [position] = load_turn_credit_positions(path)

    assert (position.seed, position.seat, position.round) == (71, 0, 4)
    assert (position.greedy_index, position.search_index) == (0, 1)
    np.testing.assert_array_equal(position.outcome_scores, [0, 2, -1])
    np.testing.assert_array_equal(position.static_scores, [100, 101, 102])
    np.testing.assert_array_equal(position.complete, [True, True, False])
    np.testing.assert_array_equal(position.post_turn["action_kinds"], [0, 0, 0])
    np.testing.assert_array_equal(position.post_turn["action_offsets"], [0, 1, 2, 3])
    assert position.slate_indices == (1, 2)
    assert position.slate_first_actions == ('{"Move": 1}', '{"Move": 2}')
    assert position.post_turn_rules_json == ('{"profile":"ClassicGeneric"}',) * 3


def test_exported_action_kinds_match_native_model_codes() -> None:
    exported = observation(6)
    exported["actions"] = [
        {"kind": kind, "source": 65535, "target": 65535, "parameter": 0}
        for kind in (
            "EndTurn",
            "Move",
            "Recruit",
            "Build",
            "PlantTree",
            "Diplomacy",
        )
    ]
    model_input, _ = model_observation(exported)

    np.testing.assert_array_equal(model_input["action_kinds"], [0, 1, 2, 3, 4, 5])


def test_duplicate_state_must_have_same_continuation(tmp_path: Path) -> None:
    value = report()
    record = value["records"][0]
    record["alternatives"][0]["branch"] = branch(1, 2)
    path = tmp_path / "inconsistent.json"
    write_report(path, value)

    with pytest.raises(ValueError, match="disagree on outcome"):
        load_turn_credit_positions(path)


def test_loader_requires_observations(tmp_path: Path) -> None:
    value = report()
    value["include_observations"] = False
    path = tmp_path / "missing.json"
    write_report(path, value)

    with pytest.raises(ValueError, match="--include-observations"):
        load_turn_credit_positions(path)


def test_loader_accepts_compressed_reports(tmp_path: Path) -> None:
    path = tmp_path / "credit.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as destination:
        json.dump(report(), destination)

    [position] = load_turn_credit_positions(path)

    np.testing.assert_array_equal(position.outcome_scores, [0, 2, -1])


@pytest.mark.parametrize(
    ("nodes_field", "continuations_field", "scores_field"),
    [
        (
            "opponent_search_nodes",
            "opponent_search_continuations",
            "opponent_search_scores",
        ),
        ("root_search_nodes", "root_search_continuations", "root_search_scores"),
    ],
)
def test_loader_aligns_search_probe_labels_with_distinct_states(
    tmp_path: Path,
    nodes_field: str,
    continuations_field: str,
    scores_field: str,
) -> None:
    value = report()
    value[nodes_field] = 32
    value["records"][0][continuations_field] = [
        {"winner": 0, "truncated": False, "reply_score": 100},
        {"winner": 1, "truncated": False, "reply_score": 200},
        {"winner": None, "truncated": True, "reply_score": None},
    ]
    path = tmp_path / "probed.json"
    write_report(path, value)

    [position] = load_turn_credit_positions(path)

    np.testing.assert_array_equal(getattr(position, scores_field), [2, 0, -1])
    assert getattr(position, nodes_field) == 32
    if nodes_field == "opponent_search_nodes":
        np.testing.assert_array_equal(position.opponent_reply_scores[:2], [100, 200])
        assert np.isnan(position.opponent_reply_scores[2])


def test_loader_can_use_persistent_teacher_outcomes(tmp_path: Path) -> None:
    value = report()
    value["teacher_continuations"] = True
    value["records"][0]["teacher_continuations"] = [
        {"winner": 0, "truncated": False},
        {"winner": 1, "truncated": False},
        {"winner": None, "truncated": True},
    ]
    path = tmp_path / "teacher.json"
    write_report(path, value)

    [position] = load_turn_credit_positions(path, "teacher")

    np.testing.assert_array_equal(position.outcome_scores, [2, 0, -1])
    write_report(tmp_path / "missing.json", report())
    with pytest.raises(ValueError, match="teacher outcomes require"):
        load_turn_credit_positions(tmp_path / "missing.json", "teacher")


@pytest.mark.parametrize(
    ("nodes_field", "continuations_field"),
    [
        ("opponent_search_nodes", "opponent_search_continuations"),
        ("root_search_nodes", "root_search_continuations"),
    ],
)
def test_loader_rejects_missing_search_probe_state(
    tmp_path: Path,
    nodes_field: str,
    continuations_field: str,
) -> None:
    value = report()
    value[nodes_field] = 32
    value["records"][0][continuations_field] = [{"winner": 0, "truncated": False}]
    path = tmp_path / "incomplete-probe.json"
    write_report(path, value)

    with pytest.raises(ValueError, match="probe count"):
        load_turn_credit_positions(path)
