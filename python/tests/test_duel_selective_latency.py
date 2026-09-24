from copy import deepcopy
import json
import sys

import pytest

from python import audit_duel_selective_latency as latency


summarize = latency.summarize


def record(seed: int, seat: int) -> dict[str, object]:
    return {
        "seed": seed,
        "root_seat": seat,
        "sampled_turns": 1,
        "sampled_decisions": 2,
        "teacher_queries": 1,
        "final_actions_different_from_source": 1,
        "rollin_truncated": False,
        "action_milliseconds": {
            "source": {"cpu": [1.0, 1.0], "wall": [1.0, 1.0]},
            "selective": {"cpu": [2.0, 3.0], "wall": [2.0, 3.0]},
            "teacher": {"cpu": [4.0, 5.0], "wall": [4.0, 5.0]},
        },
        "source_turn_milliseconds": {
            "source": {"cpu": [2.0], "wall": [2.0]},
            "selective": {"cpu": [5.0], "wall": [5.0]},
            "teacher": {"cpu": [9.0], "wall": [9.0]},
        },
    }


def test_selective_latency_gate_uses_matched_action_cost_and_both_seats() -> None:
    result = summarize([record(1, 0), record(1, 1)])

    assert result["sampled_decisions"] == 4
    assert result["sampled_source_turns"] == 2
    assert result["teacher_query_fraction"] == 0.5
    assert (
        result["matched_timing"]["selective"]["action_milliseconds"]["cpu"]["samples"]
        == 4
    )
    assert (
        result["by_root_seat"]["0"]["selective"]["source_turn_milliseconds"]["cpu"][
            "median_milliseconds"
        ]
        == 5.0
    )
    assert result["runtime_gate_passed"] is True


def test_selective_latency_gate_rejects_censoring_and_slow_queries() -> None:
    slow = record(1, 1)
    slow["rollin_truncated"] = True
    slow["teacher_queries"] = 2
    slow["action_milliseconds"]["selective"]["cpu"] = [300.0, 400.0]

    result = summarize([record(1, 0), slow])

    assert result["censored_rollins"] == 1
    assert result["teacher_query_fraction"] == 0.75
    assert result["gate"]["zero_censored_rollins"] is False
    assert result["gate"]["teacher_query_fraction_at_most_half"] is False
    assert result["gate"]["selective_p95_cpu_below_250ms"] is False
    assert result["runtime_gate_passed"] is False

    missing_seat = deepcopy(slow)
    missing_seat["root_seat"] = 0
    with pytest.raises(ValueError, match="both seats"):
        summarize([record(1, 0), missing_seat])


def test_selective_latency_cli_routes_checkpoint_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "latency",
            "--first-seed",
            "6531900",
            "--maps",
            "2",
            "--source",
            "source.pt",
            "--student",
            "student.pt",
        ],
    )
    monkeypatch.setattr(
        latency,
        "audit",
        lambda **arguments: {
            **arguments,
            "source_path": str(arguments["source_path"]),
            "student_path": str(arguments["student_path"]),
        },
    )

    latency.main()

    assert json.loads(capsys.readouterr().out) == {
        "first_seed": 6531900,
        "maps": 2,
        "turns_per_game": 12,
        "source_path": "source.pt",
        "student_path": "student.pt",
    }
