from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from python.train import Rollout, TrainingConfig, rollout_targets


def training_config() -> TrainingConfig:
    return TrainingConfig(
        environments=1,
        updates=1,
        width=7,
        height=5,
        seed=1,
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
