from __future__ import annotations

import numpy as np

from python.audit_duel_aggregate_score import aggregate_score, omitted_score
from python.audit_native_score_observation import observed_position_score


def economy() -> dict[str, object]:
    return {
        "unit_upkeep": [0, 2, 6, 18, 36],
        "clear_hex_income": 1,
        "farm_hex_income": 5,
        "tower_upkeep": 1,
        "strong_tower_upkeep": 6,
    }


def observation() -> dict[str, np.ndarray]:
    values = {
        "cell_offsets": [0, 2],
        "province_offsets": [0, 1],
        "owners": [0, 1],
        "objects": [2, 7],
        "unit_strengths": [0, 0],
        "ready": [0, 0],
        "defenses": [0, 0],
        "province_ids": [0, 65535],
        "province_owners": [0],
        "province_money": [10],
        "province_profit": [5],
        "province_sizes": [1],
    }
    return {name: np.asarray(value) for name, value in values.items()}


def test_aggregate_omissions_exactly_explain_grave_and_orphan_economy() -> None:
    state = observation()
    coarse = aggregate_score(state, 0, 0, economy())
    grave, orphan, grave_count, orphan_count = omitted_score(state, 0, 0, economy())
    native = observed_position_score({**state, "rules": [{"economy": economy()}]}, 0, 0)

    assert coarse == 98
    assert (grave, orphan, grave_count, orphan_count) == (16, 11, 1, 1)
    assert native == coarse + grave + orphan == 125


def test_aggregate_score_is_exact_without_omitted_terms() -> None:
    state = observation()
    state["objects"][1] = 0
    state["province_ids"][1] = 1
    state["province_offsets"][1] = 2
    state["province_owners"] = np.asarray([0, 1])
    state["province_money"] = np.asarray([10, 0])
    state["province_profit"] = np.asarray([5, 1])
    state["province_sizes"] = np.asarray([1, 1])
    coarse = aggregate_score(state, 0, 0, economy())
    grave, orphan, grave_count, orphan_count = omitted_score(state, 0, 0, economy())
    native = observed_position_score({**state, "rules": [{"economy": economy()}]}, 0, 0)

    assert (grave, orphan, grave_count, orphan_count) == (0, 0, 0, 0)
    assert coarse == native
