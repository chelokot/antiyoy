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


def test_first_records_selects_first_position_in_each_map_and_seat() -> None:
    records: list[dict[str, object]] = []
    for seed in reversed(range(FIT_SEED, FIT_SEED + FIT_MAPS)):
        for seat in (1, 0):
            records.extend(
                [
                    {"seed": seed, "seat": seat, "round": 8},
                    {"seed": seed, "seat": seat, "round": 16},
                ]
            )

    selected = first_records({"records": records}, both_seats=True)

    assert [(record["seed"], record["seat"]) for record in selected] == [
        (seed, seat) for seed in range(FIT_SEED, FIT_SEED + FIT_MAPS) for seat in (0, 1)
    ]
    assert all(record["round"] == 8 for record in selected)


def test_first_records_rejects_missing_seat() -> None:
    records = [
        {"seed": seed, "seat": 0} for seed in range(FIT_SEED, FIT_SEED + FIT_MAPS)
    ]

    with pytest.raises(ValueError, match="predeclared map and seat"):
        first_records({"records": records}, both_seats=True)
