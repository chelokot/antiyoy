import numpy as np
import pytest

from antiyoy_rl import (
    GENERATOR_ROTATED_SCHEMA_VERSION,
    GENERATOR_SCHEMA_VERSION,
    ProceduralConfig,
    VectorEnv,
)
from python.benchmark_seat_balance import (
    benchmark_seat_balance,
    initial_region_sizes,
)


def test_initial_regions_use_farthest_start_order_and_axial_neighbours() -> None:
    observation = {
        "widths": np.array([5]),
        "heights": np.array([1]),
        "player_counts": np.array([2]),
        "cell_offsets": np.array([0, 5]),
        "playable": np.ones(5, dtype=np.uint8),
        "province_offsets": np.array([0, 2]),
        "province_owners": np.array([0, 1]),
        "province_capitals": np.array([0, 4]),
    }
    assert initial_region_sizes(observation, 0).tolist() == [3, 2]


@pytest.mark.parametrize(
    "schema_version", [GENERATOR_SCHEMA_VERSION, GENERATOR_ROTATED_SCHEMA_VERSION]
)
def test_greedy_seat_balance_is_batch_invariant(schema_version: int) -> None:
    generator = ProceduralConfig(
        width=11, height=9, players=2, seed=851, schema_version=schema_version
    )
    serial = benchmark_seat_balance(generator, "classic_generic_2022", 6, 1, 8)
    parallel = benchmark_seat_balance(generator, "classic_generic_2022", 6, 3, 8)
    assert serial["wins_by_seat"] == parallel["wins_by_seat"]
    assert serial["draws"] == parallel["draws"]
    assert serial["truncations"] == parallel["truncations"]
    assert sum(serial["wins_by_seat"]) + serial["draws"] == 6
    assert (
        serial["initial_region_sizes_by_seed"]
        == parallel["initial_region_sizes_by_seed"]
    )


def test_rotated_schema_preserves_geometry_and_shifts_region_labels() -> None:
    base = dict(width=19, height=15, players=5, seed=47)
    legacy = VectorEnv.procedural(
        1, ProceduralConfig(**base, schema_version=GENERATOR_SCHEMA_VERSION)
    ).observe()
    rotated = VectorEnv.procedural(
        1, ProceduralConfig(**base, schema_version=GENERATOR_ROTATED_SCHEMA_VERSION)
    ).observe()
    assert np.array_equal(legacy["playable"], rotated["playable"])
    assert np.array_equal(legacy["objects"], rotated["objects"])
    sizes = initial_region_sizes(legacy, 0)
    shifted = initial_region_sizes(rotated, 0, 2)
    assert shifted.tolist() == np.roll(sizes, 2).tolist()


def test_greedy_seat_balance_requires_positive_workload() -> None:
    generator = ProceduralConfig(width=11, height=9, players=2, seed=851)
    with pytest.raises(ValueError, match="must be positive"):
        benchmark_seat_balance(generator, "classic_generic_2022", 0, 1, 8)
