import pytest

from python.audit_duel_fixed_snapshot_memorization import first_records
from python.train_three_turn_plan import FIT_MAPS, FIT_SEED


def test_first_records_selects_earliest_position_per_independent_map() -> None:
    records: list[dict[str, object]] = []
    for seed in reversed(range(FIT_SEED, FIT_SEED + FIT_MAPS)):
        records.extend(
            [
                {"seed": seed, "round": 8},
                {"seed": seed, "round": 16},
            ]
        )

    selected = first_records({"records": records})

    assert [record["seed"] for record in selected] == list(
        range(FIT_SEED, FIT_SEED + FIT_MAPS)
    )
    assert all(record["round"] == 8 for record in selected)


def test_first_records_rejects_missing_map() -> None:
    records = [{"seed": seed} for seed in range(FIT_SEED, FIT_SEED + FIT_MAPS - 1)]

    with pytest.raises(ValueError, match="predeclared map"):
        first_records({"records": records})
