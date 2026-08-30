from dataclasses import replace

import pytest

from python.benchmark_teacher import BenchmarkConfig, benchmark


def benchmark_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        environments=2,
        transitions=5,
        procedural=False,
        width=7,
        height=5,
        players=2,
        seed=41,
        land_density_per_million=650_000,
        action_limit=100,
        profiles=["classic_generic_2022", "online_duel_v1"],
        search_nodes=64,
        search_beam_width=4,
        search_branch_width=8,
        search_maximum_actions_per_turn=6,
        replan_each_action=False,
    )


def test_teacher_benchmark_rounds_up_complete_vector_steps() -> None:
    result = benchmark(benchmark_config())

    assert result["map"] == "symmetric_duel_v1"
    assert result["requested_transitions"] == 5
    assert result["executed_transitions"] == 6
    assert result["search_plan_mode"] == "cached_whole_turn"
    assert result["teacher_transitions_per_second"] > 0


def test_teacher_benchmark_can_replan_every_action() -> None:
    result = benchmark(replace(benchmark_config(), replan_each_action=True))

    assert result["search_plan_mode"] == "replan_each_action"
    assert result["teacher_transitions_per_second"] > 0


def test_teacher_benchmark_rejects_an_empty_profile_schedule() -> None:
    config = replace(benchmark_config(), profiles=[])

    with pytest.raises(ValueError, match="profiles must not be empty"):
        benchmark(config)
