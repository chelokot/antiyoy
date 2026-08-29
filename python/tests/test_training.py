from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from python.train import (
    Rollout,
    TrainingConfig,
    make_environment,
    rollout_targets,
    validate_config,
)


def training_config() -> TrainingConfig:
    return TrainingConfig(
        environments=1,
        updates=1,
        procedural=False,
        width=7,
        height=5,
        players=2,
        seed=1,
        land_density_per_million=650_000,
        starting_province_size=5,
        starting_money=10,
        tree_density_per_million=150_000,
        neutral_tower_density_per_million=20_000,
        neutral_capital_density_per_million=10_000,
        grave_density_per_million=15_000,
        objective_json=None,
        action_limit=100,
        hidden=16,
        layers=1,
        learning_rate=3e-4,
        gamma=0.9,
        gae_lambda=0.8,
        rollout_steps=2,
        epochs=1,
        clip_ratio=0.2,
        imitation_updates=0,
        imitation_teacher="greedy",
        search_nodes=256,
        search_beam_width=12,
        search_branch_width=20,
        search_maximum_actions_per_turn=12,
        entropy_weight=0.01,
        value_weight=0.5,
        territory_weight=0.03,
        treasury_weight=0.002,
        unit_weight=0.01,
        profile="classic_generic_2022",
        profiles=None,
        fog=False,
        diplomacy=False,
        initial_relation="neutral",
        device="cpu",
        resume=None,
        checkpoint=Path("unused.pt"),
    )


def test_gae_changes_perspective_when_the_active_player_changes() -> None:
    rollout = Rollout(
        observations=[{}, {}],
        actions=[],
        log_probabilities=torch.zeros((2, 1)),
        values=torch.zeros((2, 1)),
        rewards=torch.zeros((2, 1)),
        perspectives=-torch.ones((2, 1)),
        continuations=torch.ones((2, 1)),
        bootstrap=torch.ones(1),
    )

    _, returns = rollout_targets(rollout, training_config())

    assert torch.allclose(returns[1], torch.tensor([-0.9]))
    assert torch.allclose(returns[0], torch.tensor([0.648]))


def test_training_environment_uses_procedural_domain_randomization() -> None:
    config = replace(
        training_config(),
        environments=2,
        procedural=True,
        width=17,
        height=13,
        players=4,
        land_density_per_million=600_000,
    )
    environment = make_environment(config)
    observation = environment.observe()
    assert observation["player_counts"].tolist() == [4, 4]
    assert observation["playable"][:221].sum() == 133
    assert not torch.equal(
        torch.from_numpy(observation["playable"][:221]),
        torch.from_numpy(observation["playable"][221:]),
    )


def test_training_rejects_invalid_procedural_density() -> None:
    config = replace(
        training_config(), procedural=True, land_density_per_million=1_000_001
    )
    with pytest.raises(ValueError, match="densities"):
        validate_config(config)


def test_training_rejects_invalid_search_teacher_configuration() -> None:
    with pytest.raises(ValueError, match="imitation_teacher"):
        validate_config(replace(training_config(), imitation_teacher="unknown"))
    with pytest.raises(ValueError, match="search_nodes"):
        validate_config(replace(training_config(), search_nodes=1))
