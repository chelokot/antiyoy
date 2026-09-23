import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from antiyoy_rl.slate_dataset import load_teacher_slates
from antiyoy_rl.turn_credit import OBSERVATION_FIELDS


def report() -> dict[str, object]:
    observation: dict[str, object] = {field: [] for field in OBSERVATION_FIELDS}
    for field in ("cell_offsets", "province_offsets", "relation_offsets"):
        observation[field] = [0, 0, 0]
    observation["action_offsets"] = [0, 1, 2]
    observation["widths"] = [7, 7]
    observation["heights"] = [5, 5]
    observation["active_players"] = [1, 1]
    observation["player_counts"] = [2, 2]
    observation["rounds"] = [4, 4]
    observation["actions"] = [
        {"kind": "EndTurn", "source": 65535, "target": 65535, "parameter": 0}
    ] * 2
    observation["rules"] = [{"profile": "ClassicGeneric"}] * 2
    return {
        "schema_version": 1,
        "generator": {"players": 2},
        "positions": 1,
        "records": [
            {
                "seed": 71,
                "seat": 0,
                "round": 4,
                "selected_index": 1,
                "static_scores": [200, 100],
                "reply_scores": [10, 20],
                "actions": [
                    [{"Move": {"source": 1, "target": 2}}, "EndTurn"],
                    [{"Recruit": {"target": 3}}, "EndTurn"],
                ],
                "post_turn": observation,
            }
        ],
    }


def test_teacher_slate_loader_aligns_selection_and_observations(tmp_path: Path) -> None:
    path = tmp_path / "slates.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as destination:
        json.dump(report(), destination)

    [position] = load_teacher_slates(path)

    np.testing.assert_array_equal(position.static_scores, [200, 100])
    np.testing.assert_array_equal(position.opponent_reply_scores, [10, 20])
    np.testing.assert_array_equal(position.outcome_scores, [-1, -1])
    assert position.slate_indices == (0, 1)
    assert position.slate_first_actions == (
        '{"Move": {"source": 1, "target": 2}}',
        '{"Recruit": {"target": 3}}',
    )
    assert position.opponent_actions is None
    assert position.post_reply is None
    assert position.opponent_decisions is None


def test_teacher_slate_loader_aligns_opponent_actions(tmp_path: Path) -> None:
    value = report()
    value["records"][0]["opponent_actions"] = [
        [{"Move": {"source": 4, "target": 5}}, "EndTurn"],
        ["EndTurn"],
    ]
    path = tmp_path / "replies.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    [position] = load_teacher_slates(path)

    assert position.opponent_actions == (
        ('{"Move": {"source": 4, "target": 5}}', '"EndTurn"'),
        ('"EndTurn"',),
    )


def test_teacher_slate_loader_aligns_post_reply_observations(tmp_path: Path) -> None:
    value = report()
    value["records"][0]["post_reply"] = value["records"][0]["post_turn"]
    path = tmp_path / "observed-replies.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    [position] = load_teacher_slates(path)

    assert position.post_reply is not None
    np.testing.assert_array_equal(position.post_reply["active_players"], [1, 1])


def test_teacher_slate_loader_aligns_opponent_decisions(tmp_path: Path) -> None:
    value = report()
    value["records"][0]["opponent_actions"] = [["EndTurn"], ["EndTurn"]]
    value["records"][0]["opponent_decisions"] = {
        "observation": value["records"][0]["post_turn"],
        "candidate_offsets": [0, 1, 2],
        "action_indices": [0, 0],
    }
    path = tmp_path / "opponent-decisions.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    [position] = load_teacher_slates(path)

    assert position.opponent_decisions is not None
    np.testing.assert_array_equal(
        position.opponent_decisions.candidate_offsets, [0, 1, 2]
    )
    np.testing.assert_array_equal(position.opponent_decisions.action_indices, [0, 0])
    np.testing.assert_array_equal(
        position.opponent_decisions.observation["widths"], [7, 7]
    )


def test_teacher_slate_loader_rejects_illegal_opponent_decision(
    tmp_path: Path,
) -> None:
    value = report()
    value["records"][0]["opponent_decisions"] = {
        "observation": value["records"][0]["post_turn"],
        "candidate_offsets": [0, 1, 2],
        "action_indices": [0, 1],
    }
    path = tmp_path / "illegal-opponent-decisions.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="action is not legal"):
        load_teacher_slates(path)


def test_teacher_slate_loader_rejects_misaligned_post_reply_observations(
    tmp_path: Path,
) -> None:
    value = report()
    value["records"][0]["post_reply"] = report()["records"][0]["post_turn"]
    value["records"][0]["post_reply"]["widths"] = [7]
    path = tmp_path / "incorrect-observed-replies.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="opponent observation counts differ"):
        load_teacher_slates(path)


def test_teacher_slate_loader_rejects_misaligned_opponent_actions(
    tmp_path: Path,
) -> None:
    value = report()
    value["records"][0]["opponent_actions"] = [["EndTurn"]]
    path = tmp_path / "incorrect-replies.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="opponent action counts differ"):
        load_teacher_slates(path)


def test_teacher_slate_loader_rejects_inconsistent_selection(tmp_path: Path) -> None:
    value = report()
    value["records"][0]["selected_index"] = 0
    path = tmp_path / "incorrect.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="selection disagrees"):
        load_teacher_slates(path)
