from __future__ import annotations

from copy import deepcopy

import pytest

from python.audit_native_score_observation import observed_position_score


def observation() -> dict[str, object]:
    return {
        "cell_offsets": [0, 3],
        "province_offsets": [0, 2],
        "owners": [0, 0, 1],
        "objects": [2, 3, 5],
        "unit_strengths": [0, 1, 0],
        "defenses": [0, 1, 0],
        "province_ids": [0, 0, 1],
        "province_owners": [0, 1],
        "province_money": [10, 4],
        "province_profit": [4, 0],
        "rules": [
            {
                "economy": {
                    "unit_upkeep": [0, 2, 6, 18, 36],
                    "clear_hex_income": 1,
                    "farm_hex_income": 5,
                    "tower_upkeep": 1,
                    "strong_tower_upkeep": 6,
                }
            }
        ],
    }


def test_observed_score_reconstructs_cell_and_province_terms() -> None:
    state = observation()

    assert observed_position_score(state, 0, root_seat=0) == 317
    assert observed_position_score(state, 0, root_seat=1) == -317


def test_observed_score_rejects_inconsistent_profit() -> None:
    state = deepcopy(observation())
    state["province_profit"][0] = 5

    with pytest.raises(ValueError, match="province profit disagrees"):
        observed_position_score(state, 0, root_seat=0)
