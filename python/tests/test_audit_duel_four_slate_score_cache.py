from python.audit_duel_four_slate_score_cache import summarize_four_slate


def record(history_cpu: float, history_hits: int) -> dict[str, object]:
    return {
        "seed": 1,
        "actions": 2,
        "outcome": {"terminal": True, "truncated": False},
        "plain_cache_hits": 10,
        "cache_hits": history_hits,
        "plain_search_cpu_seconds_by_seat": [1.0, 1.0],
        "cached_search_cpu_seconds_by_seat": [history_cpu, history_cpu],
        "plain_action_wall_seconds": [1.0, 1.0],
        "cached_action_wall_seconds": [1.0, 1.0],
    }


def test_four_slate_gate_requires_additional_reuses_and_cpu_gain() -> None:
    passing = summarize_four_slate([record(0.95, 11)])
    assert passing["advance_gate_passed"] is True
    assert passing["additional_exact_score_reuses"] == 1

    no_new_hits = summarize_four_slate([record(0.95, 10)])
    assert no_new_hits["advance_gate_passed"] is False
    assert no_new_hits["gate"]["four_slates_reused_additional_exact_scores"] is False

    slow = summarize_four_slate([record(0.99, 11)])
    assert slow["advance_gate_passed"] is False
    assert slow["gate"]["median_map_cpu_improvement_at_least_three_percent"] is False
