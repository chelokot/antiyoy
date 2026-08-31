from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import UniversalPolicy, load_policy_state
from python.build_bundle import digest
from python.collect_action_q import observation_fingerprint
from python.collect_action_slate import (
    DATASET_KIND,
    DATASET_SCHEMA_VERSION,
    informative_states,
    replay_slate_state,
)
from python.evaluate import load_policy_checkpoint
from python.tests.test_bundle import write_checkpoint
from python.train_action_slate import (
    conservative_target_logits,
    load_action_slate_dataset,
    select_slate_states,
    train_action_slate,
)


def write_action_slate_dataset(path: Path, checkpoint_path: Path) -> None:
    checkpoint = load_policy_checkpoint(checkpoint_path, torch.device("cpu"))
    model = UniversalPolicy(hidden=16, layers=1)
    load_policy_state(model, checkpoint["model"])
    generator = torch.Generator().manual_seed(211)
    states = 40
    actions = states * 2
    bases = torch.randn((states, 80), generator=generator)
    direction = torch.randn(80, generator=generator)
    direction /= direction.norm()
    features = torch.stack((bases + direction, bases - direction), dim=1).reshape(
        actions, 80
    )
    with torch.no_grad():
        baseline_logits = model.action_head(features).squeeze(1)
    offsets = torch.arange(0, actions + 1, 2, dtype=torch.int64)
    root_values = torch.tensor([1.0, -1.0]).repeat(states)
    root_visits = torch.full((actions,), 4, dtype=torch.int32)
    torch.save(
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "kind": DATASET_KIND,
            "source": {
                "path": str(checkpoint_path),
                "sha256": digest(checkpoint_path),
                "seat_experts": ["single", "single"],
                "feature_expert": "single",
            },
            "model": {"hidden": 16, "layers": 1, "feature_width": 80},
            "config": {
                "profile": "classic_generic_2022",
                "generator": "symmetric_duel_v1",
                "players": 2,
            },
            "actions": {
                "offsets": offsets,
                "features": features.to(torch.float16),
                "baseline_logits": baseline_logits,
                "root_probabilities": torch.full((actions,), 0.5),
                "root_values": root_values,
                "root_visits": root_visits,
            },
            "states": {
                "episode_seeds": torch.arange(states) // 4,
                "episode_steps": torch.arange(states, dtype=torch.int32) % 4,
                "seats": torch.arange(states, dtype=torch.uint8) % 2,
                "rounds": torch.ones(states, dtype=torch.int32),
                "search_actions": torch.zeros(states, dtype=torch.int32),
                "direct_actions": torch.ones(states, dtype=torch.int32),
                "fingerprints": [f"{state:064x}" for state in range(states)],
            },
            "replay": {
                "episode_seeds": torch.arange(10, dtype=torch.int64),
                "action_offsets": torch.arange(0, 41, 4, dtype=torch.int64),
                "actions": torch.zeros(40, dtype=torch.int32),
            },
        },
        path,
    )


def test_conservative_targets_ignore_unmeasured_and_single_visit_slates() -> None:
    offsets = torch.tensor([0, 3, 5])
    baseline = torch.zeros(5)
    values = torch.tensor([1.0, -1.0, 0.0, 0.5, 0.0])
    visits = torch.tensor([4, 2, 0, 3, 0])

    targets = conservative_target_logits(
        baseline,
        values,
        visits,
        offsets,
        advantage_scale=1.0,
        visit_prior=2.0,
        advantage_clip=2.0,
    )

    assert targets[0] > 0
    assert targets[1] < 0
    assert targets[2:].tolist() == [0.0, 0.0, 0.0]
    assert informative_states(offsets, values, visits) == 1


def test_action_slate_replay_reconstructs_normalized_episode_prefix() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=613)
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
        "states": {
            "episode_seeds": torch.tensor([613]),
            "episode_steps": torch.tensor([0]),
        },
        "replay": {
            "episode_seeds": torch.tensor([613]),
            "action_offsets": torch.tensor([0, 0]),
            "actions": torch.empty(0, dtype=torch.int32),
        },
    }

    assert replay_slate_state(dataset, 0) == fingerprint


def test_action_slate_selection_rebuilds_flat_offsets(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    dataset_path = tmp_path / "slates.pt"
    write_checkpoint(checkpoint, 1.0)
    write_action_slate_dataset(dataset_path, checkpoint)
    dataset = load_action_slate_dataset(dataset_path)
    examples = {
        **dataset["actions"],
        **{
            name: value
            for name, value in dataset["states"].items()
            if isinstance(value, torch.Tensor)
        },
        "groups": dataset["states"]["episode_seeds"],
    }

    selected = select_slate_states(
        examples, torch.arange(40, dtype=torch.int64) % 3 == 0
    )

    assert selected["offsets"].shape == (15,)
    assert selected["features"].shape == (28, 80)
    assert selected["groups"].shape == (14,)


def test_action_slate_loader_rejects_malformed_flat_actions(tmp_path: Path) -> None:
    checkpoint = tmp_path / "source.pt"
    dataset_path = tmp_path / "slates.pt"
    write_checkpoint(checkpoint, 1.0)
    write_action_slate_dataset(dataset_path, checkpoint)
    malformed = torch.load(dataset_path, map_location="cpu", weights_only=False)
    malformed["actions"]["root_visits"] = malformed["actions"]["root_visits"][:-1]
    torch.save(malformed, dataset_path)

    with pytest.raises(ValueError, match="different lengths"):
        load_action_slate_dataset(dataset_path)


def test_conservative_slate_training_improves_game_disjoint_targets(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source.pt"
    dataset = tmp_path / "slates.pt"
    output = tmp_path / "candidate.pt"
    write_checkpoint(checkpoint, 1.0)
    write_action_slate_dataset(dataset, checkpoint)

    report = train_action_slate(
        checkpoint,
        [dataset],
        output,
        "cpu",
        epochs=24,
        batch_size=8,
        learning_rate=1e-3,
        advantage_scale=1.0,
        visit_prior=1.0,
        advantage_clip=2.0,
        retention_weight=0.1,
        validation_fraction=0.2,
        seed=212,
        training_seat=1,
    )

    assert (
        report["validation_after"]["target_kl"]
        < report["validation_before"]["target_kl"]
    )
    assert report["validation_after"]["measured_pair_accuracy"] > 0.5
    assert report["frozen_parameters_preserved"] is True
    assert report["changed_action_parameters"] > 0
    assert report["states"] == 20
    assert report["training_seat"] == 1
    assert output.is_file()
