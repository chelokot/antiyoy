from __future__ import annotations

from python.audit_duel_exact_score_cache import summarize
from python.audit_duel_exact_score_cache_profiles import summarize_profile


def record(seed: int, cached_cpu: list[float], cached_wall: list[float]) -> dict[str, object]:
    return {
        "seed": seed,
        "actions": 2,
        "outcome": {"terminal": True, "truncated": False},
        "cache_hits": 1,
        "plain_search_cpu_seconds_by_seat": [1.0, 1.0],
        "cached_search_cpu_seconds_by_seat": cached_cpu,
        "plain_action_wall_seconds": [1.0, 1.0],
        "cached_action_wall_seconds": cached_wall,
    }


def test_exact_score_cache_gate_requires_both_seats_and_p95() -> None:
    passing = summarize([record(1, [0.8, 0.8], [0.9, 1.0])])
    assert passing["advance_gate_passed"] is True

    slow_seat = summarize([record(1, [0.6, 1.1], [0.9, 1.0])])
    assert slow_seat["advance_gate_passed"] is False
    assert slow_seat["gate"]["both_seats_median_cpu_nonnegative"] is False

    slow_tail = summarize([record(1, [0.8, 0.8], [0.9, 1.2])])
    assert slow_tail["advance_gate_passed"] is False
    assert slow_tail["gate"]["candidate_p95_action_wall_within_ten_percent"] is False


def test_cross_profile_cache_gate_keeps_slay_censoring_nonterminal() -> None:
    records = [record(seed, [0.8, 0.8], [0.9, 1.0]) for seed in range(16)]
    records[0]["outcome"] = {"terminal": False, "truncated": True}

    online = summarize_profile(records, "online_default_v1")
    slay = summarize_profile(records, "classic_slay_2022")
    assert online["profile_gate_passed"] is False
    assert slay["profile_gate_passed"] is True
    assert slay["terminal_games"] == 15
    assert slay["action_limit_censored_games"] == 1
