import numpy as np
import pytest
import torch

from antiyoy_rl.model import UniversalPolicy
from antiyoy_rl.turn_credit import TurnCreditPosition
from python.scout_turn_value import (
    EmbeddedPosition,
    embed_position,
    evaluate_head,
    pairwise_examples,
    train_head,
)


def position(
    seed: int, seat: int, features: list[list[float]], outcomes: list[int]
) -> EmbeddedPosition:
    return EmbeddedPosition(
        seed=seed,
        seat=seat,
        features=torch.tensor(features),
        outcomes=np.asarray(outcomes, dtype=np.int8),
        search_index=0,
    )


def test_pairwise_training_uses_only_complete_outcome_differences() -> None:
    training = [
        position(1, 0, [[0.0], [1.0], [2.0]], [0, 2, -1]),
        position(2, 1, [[0.0], [2.0]], [0, 2]),
    ]
    differences, example_weights = pairwise_examples(training)
    np.testing.assert_array_equal(differences.numpy().reshape(-1), [1.0, 2.0])
    np.testing.assert_array_equal(example_weights.numpy(), [1.0, 1.0])

    weight, scale = train_head(differences, example_weights, steps=40)

    assert weight[0] > 0
    assert scale[0] > 0


def test_multiple_pairs_and_seats_do_not_overweight_one_map() -> None:
    training = [
        position(1, 0, [[0.0], [1.0], [2.0]], [0, 1, 2]),
        position(1, 1, [[0.0], [1.0]], [0, 2]),
        position(2, 0, [[0.0], [1.0]], [0, 2]),
    ]

    _, example_weights = pairwise_examples(training)

    np.testing.assert_allclose(example_weights.numpy(), [1 / 6] * 3 + [1 / 2, 1])


def test_single_informative_pair_has_finite_training_scale() -> None:
    differences, example_weights = pairwise_examples(
        [position(1, 0, [[0.0], [1.0]], [0, 2])]
    )

    weight, scale = train_head(differences, example_weights, steps=4)

    assert torch.isfinite(weight).all()
    assert torch.isfinite(scale).all()


def test_scout_rejects_a_mismatched_model_domain() -> None:
    source = TurnCreditPosition(
        seed=1,
        seat=0,
        round=1,
        root={},
        post_turn={"player_counts": np.asarray([3])},
        root_rules_json=(),
        post_turn_rules_json=('{"profile":"ClassicGeneric"}',),
        outcome_scores=np.asarray([0]),
        static_scores=np.asarray([0]),
        search_index=0,
        greedy_index=0,
    )

    with pytest.raises(ValueError, match="five-player"):
        embed_position(source, UniversalPolicy(hidden=16, layers=1))


def test_outcome_scout_groups_seats_by_independent_map() -> None:
    positions = [
        position(10, 0, [[0.0], [1.0]], [0, 2]),
        position(10, 1, [[0.0], [1.0]], [0, 0]),
        position(11, 0, [[0.0], [1.0]], [2, 0]),
        position(12, 0, [[0.0], [1.0]], [0, -1]),
    ]

    report = evaluate_head(positions, torch.ones(1), torch.ones(1))

    assert report["changed_choices"] == 4
    assert (report["better"], report["worse"], report["same"], report["censored"]) == (
        1,
        1,
        1,
        1,
    )
    assert report["independent_maps"]["candidate_better"] == 1
    assert report["independent_maps"]["baseline_better"] == 1
    assert report["independent_maps"]["same"] == 0
    assert report["independent_maps"]["censored"] == 1
    assert report["independent_maps"]["exact_two_sided_sign_test_p"] == 1.0

    conservative = evaluate_head(positions, torch.ones(1), torch.ones(1), 1.0)

    assert conservative["changed_choices"] == 0
    assert conservative["censored"] == 0
