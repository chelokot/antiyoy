import pytest

from python.benchmark_reply_search_latency import benchmark


def test_reply_search_latency_counts_only_new_whole_turn_plans() -> None:
    result = benchmark(
        first_seed=91_006,
        maps=1,
        width=7,
        height=5,
        action_limit=300,
        node_budget=16,
        reply_nodes=8,
        slate_size=2,
    )

    assert result["root_decisions"] > 0
    assert result["map_ledger"][0]["root_decisions"] == result["root_decisions"]
    assert result["map_ledger"][0]["atomic_actions"] >= result["root_decisions"]
    assert result["median_root_decision_wall_seconds"] > 0


def test_reply_search_latency_rejects_empty_map_window() -> None:
    with pytest.raises(ValueError, match="at least one map"):
        benchmark(first_seed=91_006, maps=0)
