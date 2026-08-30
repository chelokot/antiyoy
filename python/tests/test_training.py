from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from python.train import (
    Rollout,
    TrainingConfig,
    imitation_weights,
    initialization_state,
    make_environment,
    parsed_slice_weights,
    policy_rollin_mask,
    recovery_checkpoint_path,
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
        imitation_rollin="teacher",
        imitation_symmetry_augmentation=False,
        imitation_reference_weight=0.0,
        imitation_slice_weights=[],
        imitation_policy_rollin_slices=[],
        checkpoint_every=0,
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
        initialize=None,
        initialize_profile=None,
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


def test_training_allows_imitation_without_ppo() -> None:
    validate_config(replace(training_config(), updates=0, imitation_updates=1))


def test_training_rejects_empty_curriculum() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_config(replace(training_config(), updates=0, imitation_updates=0))


def test_training_rejects_negative_reference_weight() -> None:
    with pytest.raises(ValueError, match="reference_weight"):
        validate_config(replace(training_config(), imitation_reference_weight=-0.1))


def test_slice_weights_target_profile_and_active_player() -> None:
    config = replace(
        training_config(),
        environments=4,
        profiles=["classic_generic_2022", "online_duel_v1"],
        profile=None,
        imitation_slice_weights=[
            "online_duel_v1:0:3",
            "online_duel_v1:1:5",
        ],
    )
    observation = {
        "active_players": torch.tensor([0, 0, 1, 1], dtype=torch.uint8).numpy()
    }

    weights = imitation_weights(config, observation, torch.device("cpu"))

    assert weights.tolist() == [1.0, 3.0, 1.0, 5.0]


@pytest.mark.parametrize(
    "specification",
    [
        "online_duel_v1",
        "online_duel_v1:x:2",
        "online_duel_v1:2:2",
        "online_duel_v1:1:0",
        "classic_slay_2022:1:2",
    ],
)
def test_training_rejects_invalid_slice_weights(specification: str) -> None:
    config = replace(
        training_config(),
        profiles=["classic_generic_2022", "online_duel_v1"],
        profile=None,
        imitation_slice_weights=[specification],
    )

    with pytest.raises(ValueError, match="slice"):
        parsed_slice_weights(config)


def test_policy_rollin_mask_selects_only_configured_profile_seat() -> None:
    config = replace(
        training_config(),
        environments=4,
        profiles=["classic_generic_2022", "online_duel_v1"],
        profile=None,
        imitation_policy_rollin_slices=["online_duel_v1:1"],
    )
    observation = {
        "active_players": torch.tensor([0, 0, 1, 1], dtype=torch.uint8).numpy()
    }

    selected = policy_rollin_mask(config, observation, torch.device("cpu"))

    assert selected.tolist() == [False, False, False, True]


@pytest.mark.parametrize(
    "specification",
    ["online_duel_v1", "online_duel_v1:x", "online_duel_v1:2", "classic_slay_2022:1"],
)
def test_training_rejects_invalid_policy_rollin_slices(specification: str) -> None:
    config = replace(
        training_config(),
        profiles=["classic_generic_2022", "online_duel_v1"],
        profile=None,
        imitation_policy_rollin_slices=[specification],
    )

    with pytest.raises(ValueError, match="policy rollin"):
        validate_config(config)


def test_training_rejects_checkpoint_interval_without_destination() -> None:
    with pytest.raises(ValueError, match="requires a checkpoint"):
        validate_config(
            replace(training_config(), checkpoint=None, checkpoint_every=10)
        )


def test_recovery_checkpoint_uses_one_hidden_sibling() -> None:
    assert recovery_checkpoint_path(Path("runs/policy.pt")) == Path(
        "runs/.policy.latest.pt"
    )


def test_training_rejects_invalid_search_teacher_configuration() -> None:
    with pytest.raises(ValueError, match="imitation_teacher"):
        validate_config(replace(training_config(), imitation_teacher="unknown"))
    with pytest.raises(ValueError, match="imitation_rollin"):
        validate_config(replace(training_config(), imitation_rollin="unknown"))
    with pytest.raises(ValueError, match="search_nodes"):
        validate_config(replace(training_config(), search_nodes=1))


def test_training_rejects_initialize_with_resume() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_config(
            replace(
                training_config(),
                initialize=Path("initial.pt"),
                resume=Path("resume.pt"),
            )
        )


def test_bundle_initialization_selects_profile_route() -> None:
    primary = {"weight": torch.tensor([1.0])}
    specialist = {"weight": torch.tensor([2.0])}
    checkpoint = {
        "kind": "routed_policy_bundle",
        "bundle_version": 1,
        "routes": {
            "classic_generic_2022": "primary",
            "online_experimental_v2_260801": "specialist:v2",
        },
        "experts": {"primary": primary, "specialist:v2": specialist},
    }

    state, expert = initialization_state(
        checkpoint, "online_experimental_v2_260801"
    )

    assert state is specialist
    assert expert == "specialist:v2"


def test_bundle_initialization_requires_a_profile() -> None:
    checkpoint = {
        "kind": "routed_policy_bundle",
        "bundle_version": 1,
        "routes": {},
        "experts": {},
    }

    with pytest.raises(ValueError, match="requires initialize_profile"):
        initialization_state(checkpoint, None)


def test_training_rejects_initialize_profile_without_checkpoint() -> None:
    with pytest.raises(ValueError, match="requires an initialization checkpoint"):
        validate_config(
            replace(training_config(), initialize_profile="classic_generic_2022")
        )
