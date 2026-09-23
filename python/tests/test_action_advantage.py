import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from antiyoy_rl import ProceduralConfig, VectorEnv
from antiyoy_rl.model import UniversalPolicy
from python.audit_action_q import audit_action_q
from python.build_bundle import digest
from python.collect_action_q import (
    DATASET_KIND,
    DATASET_SCHEMA_VERSION,
    create_replay_environment,
    observation_fingerprint,
    replay_dataset_example,
    verify_shared_action_representation,
)
from python.tests.test_bundle import write_checkpoint
from python.train_action_advantage import (
    load_action_q_dataset,
    split_by_episode,
    train_action_advantage,
)


def write_action_q_dataset(path: Path, checkpoint: Path) -> None:
    generator = torch.Generator().manual_seed(91)
    examples = 40
    feature_width = 80
    direct = torch.randn((examples, feature_width), generator=generator)
    direction = torch.randn((feature_width,), generator=generator)
    direction = direction / direction.norm()
    search = direct + direction
    regrets = torch.ones(examples)
    torch.save(
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "kind": DATASET_KIND,
            "source": {
                "path": str(checkpoint),
                "sha256": digest(checkpoint),
                "seat_experts": ["single"] * 2,
                "feature_expert": "single",
            },
            "model": {"hidden": 16, "layers": 1, "feature_width": feature_width},
            "config": {
                "profile": "classic_generic_2022",
                "generator": "symmetric_duel_v1",
                "players": 2,
            },
            "examples": {
                "search_features": search.to(torch.float16),
                "direct_features": direct.to(torch.float16),
                "search_values": regrets,
                "direct_values": torch.zeros(examples),
                "regrets": regrets,
                "baseline_margins": torch.full((examples,), -0.5),
                "episode_seeds": torch.arange(examples) // 4,
                "seats": torch.arange(examples, dtype=torch.uint8) % 2,
                "rounds": torch.ones(examples, dtype=torch.int32),
                "state_fingerprints": [f"{index:064x}" for index in range(examples)],
                "replay_offsets": torch.arange(examples + 1, dtype=torch.int64),
                "replay_actions": torch.zeros(examples, dtype=torch.int32),
            },
        },
        path,
    )


def test_observation_fingerprint_is_state_and_environment_specific() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=601)
    observation = environment.observe()

    first = observation_fingerprint(observation, 0)
    repeated = observation_fingerprint(environment.observe(), 0)
    environment.step(torch.zeros(1, dtype=torch.uint64).numpy())
    second = observation_fingerprint(environment.observe(), 0)

    assert first == repeated
    assert first != second


def test_action_q_collection_requires_shared_non_value_parameters() -> None:
    first = UniversalPolicy(hidden=16, layers=1)
    second = UniversalPolicy(hidden=16, layers=1)
    second.load_state_dict(first.state_dict())
    second.value_head[-1].bias.data.add_(1)

    assert verify_shared_action_representation({"first": first, "second": second}) == (
        "first"
    )
    second.action_head[-1].bias.data.add_(1)
    with pytest.raises(ValueError, match="differ only in value heads"):
        verify_shared_action_representation({"first": first, "second": second})


def test_action_q_replay_reconstructs_the_labeled_state() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=603)
    fingerprint = observation_fingerprint(environment.observe(), 0)
    dataset = {
        "config": {
            "profile": "classic_generic_2022",
            "generator": "symmetric_duel_v1",
            "descriptor": {
                "width": 7,
                "height": 5,
                "players": 2,
                "action_limit": 1000,
                "fog": False,
                "diplomacy": False,
                "initial_relation": "neutral",
            },
        },
        "examples": {
            "episode_seeds": torch.tensor([603]),
            "replay_offsets": torch.tensor([0, 0]),
            "replay_actions": torch.empty(0, dtype=torch.int32),
        },
    }

    assert replay_dataset_example(dataset, 0) == fingerprint


def test_action_q_replay_preserves_rotated_procedural_seats() -> None:
    seed = 605
    environment = VectorEnv.procedural(
        1,
        ProceduralConfig(width=11, height=9, players=2, seed=seed, schema_version=2),
    )
    environment.reset(0, seed)
    fingerprint = observation_fingerprint(environment.observe(), 0)
    dataset = {
        "config": {
            "profile": "classic_generic_2022",
            "generator": "procedural_v2",
            "descriptor": {
                "width": 11,
                "height": 9,
                "players": 2,
                "action_limit": 1000,
                "fog": False,
                "diplomacy": False,
                "initial_relation": "neutral",
                "land_density_per_million": 650_000,
                "starting_province_size": 5,
                "starting_money": 10,
                "tree_density_per_million": 150_000,
                "neutral_tower_density_per_million": 20_000,
                "neutral_capital_density_per_million": 10_000,
                "grave_density_per_million": 15_000,
            },
        },
        "examples": {
            "episode_seeds": torch.tensor([seed]),
            "replay_offsets": torch.tensor([0, 0]),
            "replay_actions": torch.empty(0, dtype=torch.int32),
        },
    }

    assert replay_dataset_example(dataset, 0) == fingerprint


def test_action_q_audit_replays_and_finishes_both_rotated_duel_branches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    dataset_path = tmp_path / "pairs.pt"
    write_checkpoint(source, 1.0)
    config = {
        "profile": "classic_generic_2022",
        "generator": "procedural_v2",
        "route_generator": "procedural_v2",
        "players": 2,
        "descriptor": {
            "width": 11,
            "height": 9,
            "players": 2,
            "action_limit": 8,
            "fog": False,
            "diplomacy": False,
            "initial_relation": "neutral",
            "land_density_per_million": 650_000,
            "starting_province_size": 5,
            "starting_money": 10,
            "tree_density_per_million": 150_000,
            "neutral_tower_density_per_million": 20_000,
            "neutral_capital_density_per_million": 10_000,
            "grave_density_per_million": 15_000,
        },
    }
    environment = create_replay_environment(config, 607)
    observation = environment.observe()
    fingerprint = observation_fingerprint(observation, 0)
    assert int(observation["action_offsets"][1]) > 1
    torch.save(
        {
            "kind": DATASET_KIND,
            "source": {"sha256": digest(source), "seat_experts": ["single", "single"]},
            "config": config,
            "examples": {
                "episode_seeds": torch.tensor([607]),
                "seats": torch.tensor(
                    [int(observation["active_players"][0])], dtype=torch.uint8
                ),
                "rounds": torch.tensor([0]),
                "state_fingerprints": [fingerprint],
                "replay_offsets": torch.tensor([0, 0]),
                "replay_actions": torch.empty(0, dtype=torch.int32),
                "direct_actions": torch.tensor([0]),
                "search_actions": torch.tensor([1]),
                "direct_values": torch.tensor([-0.5]),
                "search_values": torch.tensor([0.5]),
                "baseline_margins": torch.tensor([-0.25]),
            },
        },
        dataset_path,
    )

    report = audit_action_q(dataset_path, source, 1, None, "cpu")

    assert report["sampled_positions"] == 1
    assert report["complete_positions"] == 1
    assert report["censored_positions"] == 0
    assert (
        report["search_better"] + report["direct_better"] + report["same_outcome"] == 1
    )
    assert report["records"][0]["root_value_delta"] == pytest.approx(1.0)
    assert report["records"][0]["policy_logit_margin"] == pytest.approx(-0.25)
    assert json.loads(json.dumps(report))["sampled_positions"] == 1


def test_episode_split_never_leaks_one_game_between_sets() -> None:
    groups = torch.tensor([10, 10, 11, 11, 12, 12, 13, 13])
    training, validation = split_by_episode(groups, 0.25, 77)

    assert training.any()
    assert validation.any()
    assert set(groups[training].tolist()).isdisjoint(groups[validation].tolist())


def test_action_q_loader_rejects_malformed_feature_arrays(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    dataset = tmp_path / "pairs.pt"
    write_checkpoint(checkpoint, 1.0)
    write_action_q_dataset(dataset, checkpoint)
    malformed = torch.load(dataset, map_location="cpu", weights_only=False)
    malformed["examples"]["direct_features"] = malformed["examples"]["direct_features"][
        :, :-1
    ]
    torch.save(malformed, dataset)

    with pytest.raises(ValueError, match="invalid shape"):
        load_action_q_dataset(dataset)


def test_offline_advantage_training_improves_game_disjoint_pairs(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source.pt"
    dataset = tmp_path / "pairs.pt"
    output = tmp_path / "candidate.pt"
    write_checkpoint(checkpoint, 1.0)
    write_action_q_dataset(dataset, checkpoint)

    report = train_action_advantage(
        checkpoint,
        [dataset],
        output,
        "cpu",
        epochs=24,
        batch_size=16,
        learning_rate=1e-3,
        advantage_scale=1.0,
        retention_weight=0.01,
        validation_fraction=0.2,
        seed=92,
        training_seat=1,
    )

    assert report["validation_after"]["mae"] < report["validation_before"]["mae"]
    assert report["examples"] == 20
    assert report["training_seat"] == 1
    assert report["frozen_parameters_preserved"] is True
    assert report["changed_action_parameters"] > 0
    assert output.is_file()
    assert load_action_q_dataset(dataset)["kind"] == DATASET_KIND
