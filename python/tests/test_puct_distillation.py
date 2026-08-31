from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from python.distill_puct import (
    PuctDistillationConfig,
    distill_puct,
    validate_config,
)
from python.build_bundle import build_bundle
from python.tests.test_bundle import write_checkpoint


def test_puct_distillation_changes_only_the_action_head(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "distilled.pt"
    write_checkpoint(source, 1.0)

    report = distill_puct(
        source,
        output,
        PuctDistillationConfig(
            environments=2,
            updates=2,
            seed=812_000,
            device="cpu",
            width=7,
            height=5,
            action_limit=12,
            puct_nodes=4,
            puct_leaf_batch_size=8,
        ),
    )

    source_checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    distilled_checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    changed_action_parameters = 0
    for key, source_value in source_checkpoint["model"].items():
        distilled_value = distilled_checkpoint["model"][key]
        if key.startswith("action_head."):
            changed_action_parameters += int(
                not torch.equal(source_value, distilled_value)
            )
        else:
            assert torch.equal(source_value, distilled_value), key
    assert changed_action_parameters == report["changed_action_parameters"]
    assert changed_action_parameters > 0
    assert report["frozen_parameters_preserved"] is True
    assert report["policy_search"]["decisions"] == 4
    assert report["policy_search"]["total_nodes"] == 16
    assert distilled_checkpoint["summary"]["kind"] == (
        "policy_guided_puct_distillation"
    )


def test_puct_distillation_can_target_one_seat(tmp_path: Path) -> None:
    primary = tmp_path / "primary.pt"
    second_seat = tmp_path / "second-seat.pt"
    source = tmp_path / "source-bundle.pt"
    output = tmp_path / "seat-one.pt"
    write_checkpoint(primary, 1.0)
    write_checkpoint(second_seat, 7.0)
    build_bundle(
        primary,
        {},
        source,
        seat_context_route_paths={
            ("classic_generic_2022", "symmetric_duel_v1", 2, 1): second_seat
        },
    )

    report = distill_puct(
        source,
        output,
        PuctDistillationConfig(
            environments=2,
            updates=32,
            seed=813_000,
            device="cpu",
            width=7,
            height=5,
            action_limit=12,
            puct_nodes=4,
            puct_leaf_batch_size=8,
            training_seat=1,
        ),
    )

    assert report["training_seat"] == 1
    assert 0 < report["examples"] < report["visited_states"]
    assert 0 < report["training"]["optimization_updates"] < 32
    assert report["changed_action_parameters"] > 0
    assert report["source"]["expert"] == report["source"]["seat_experts"][1]
    assert report["source"]["seat_experts"][0] != report["source"]["seat_experts"][1]
    distilled = torch.load(output, map_location="cpu", weights_only=False)
    assert torch.all(distilled["model"]["missing_source"] == 7.0)


def test_puct_distillation_routes_a_procedural_multiplayer_seat(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary.pt"
    third_seat = tmp_path / "third-seat.pt"
    source = tmp_path / "procedural-bundle.pt"
    output = tmp_path / "seat-two.pt"
    write_checkpoint(primary, 1.0)
    write_checkpoint(third_seat, 8.0)
    build_bundle(
        primary,
        {},
        source,
        seat_context_route_paths={
            ("classic_generic_2022", "procedural_v1", 3, 2): third_seat
        },
    )

    report = distill_puct(
        source,
        output,
        PuctDistillationConfig(
            generator="procedural_v1",
            players=3,
            environments=3,
            updates=24,
            seed=814_000,
            device="cpu",
            width=9,
            height=7,
            starting_province_size=3,
            action_limit=18,
            puct_nodes=4,
            puct_leaf_batch_size=12,
            puct_value_perspective="root",
            puct_opponent_horizon="leaf",
            training_seat=2,
        ),
    )

    assert report["generator"] == "procedural_v1"
    assert report["domain_descriptor"]["players"] == 3
    assert report["domain_descriptor"]["starting_province_size"] == 3
    assert report["source"]["expert"] == report["source"]["seat_experts"][2]
    assert len(report["source"]["seat_experts"]) == 3
    assert report["policy_search"]["value_perspective"] == "root"
    assert report["policy_search"]["opponent_horizon"] == "leaf"
    assert 0 < report["examples"] < report["visited_states"]
    distilled = torch.load(output, map_location="cpu", weights_only=False)
    assert torch.all(distilled["model"]["missing_source"] == 8.0)


@pytest.mark.parametrize(
    "config",
    [
        PuctDistillationConfig(environments=0),
        PuctDistillationConfig(updates=0),
        PuctDistillationConfig(learning_rate=0),
        PuctDistillationConfig(retention_weight=-1),
        PuctDistillationConfig(rollin="unknown"),
        PuctDistillationConfig(target_mode="unknown"),
        PuctDistillationConfig(puct_nodes=1),
        PuctDistillationConfig(puct_value_perspective="unknown"),
        PuctDistillationConfig(puct_opponent_horizon="unknown"),
        PuctDistillationConfig(training_seat=2),
        PuctDistillationConfig(generator="unknown"),
        PuctDistillationConfig(players=1),
        PuctDistillationConfig(players=3),
        PuctDistillationConfig(generator="procedural_v1", players=3, training_seat=3),
        PuctDistillationConfig(
            generator="procedural_v1", land_density_per_million=1_000_001
        ),
    ],
)
def test_puct_distillation_rejects_invalid_configuration(
    config: PuctDistillationConfig,
) -> None:
    with pytest.raises(ValueError):
        validate_config(config)
