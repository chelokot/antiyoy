from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from antiyoy_rl.turn_credit import TurnCreditPosition
import python.scout_duel_teacher_choice as teacher_choice
from python.scout_duel_nonlinear_choice import choices, train_scorer
from python.scout_duel_teacher_choice import (
    agreement,
    agreement_for_choices,
    teacher_choice_examples,
    teacher_choice_pairs,
)
from python.scout_duel_turn_value import (
    FEATURE_NAMES,
    compare_choices,
    embed_position,
    pairwise_agreement,
    prediction_scores,
)


def duel_position() -> TurnCreditPosition:
    post_turn = {
        "cell_offsets": np.asarray([0, 2, 4]),
        "province_offsets": np.asarray([0, 2, 4]),
        "player_counts": np.asarray([2, 2]),
        "active_players": np.asarray([1, 1]),
        "owners": np.asarray([0, 1, 1, 0]),
        "objects": np.asarray([0, 0, 2, 0]),
        "unit_strengths": np.asarray([1, 0, 2, 0]),
        "ready": np.asarray([1, 0, 1, 0]),
        "defenses": np.asarray([1, 0, 2, 0]),
        "province_owners": np.asarray([0, 1, 0, 1]),
        "province_money": np.asarray([10, 5, 9, 6]),
        "province_profit": np.asarray([3, 1, 2, 2]),
        "province_sizes": np.asarray([1, 1, 1, 1]),
    }
    return TurnCreditPosition(
        seed=71,
        seat=0,
        round=8,
        root={},
        post_turn=post_turn,
        root_rules_json=(),
        post_turn_rules_json=("{}", "{}"),
        outcome_scores=np.asarray([2, 0]),
        static_scores=np.asarray([30, 20]),
        search_index=0,
        greedy_index=1,
        opponent_reply_scores=np.asarray([100, 200]),
        slate_indices=(0, 1),
        slate_first_actions=("move", "recruit"),
    )


def test_duel_features_keep_root_perspective_and_slate_order() -> None:
    position = embed_position(duel_position())
    features = position.position.features.numpy()

    assert features.shape == (2, len(FEATURE_NAMES))
    np.testing.assert_allclose(features[:, 0], [0.03, 0.02])
    np.testing.assert_allclose(
        features[:, FEATURE_NAMES.index("unit_one_delta")], [1, 0]
    )
    np.testing.assert_allclose(
        features[:, FEATURE_NAMES.index("unit_two_delta")], [0, -1]
    )
    np.testing.assert_allclose(features[:, FEATURE_NAMES.index("farm_delta")], [0, -1])
    assert position.reply_index == 1

    opposite = embed_position(replace(duel_position(), seat=1))
    opposite_features = opposite.position.features.numpy()
    np.testing.assert_allclose(
        opposite_features[:, FEATURE_NAMES.index("unit_one_delta")], [-1, 0]
    )
    np.testing.assert_allclose(
        opposite_features[:, FEATURE_NAMES.index("unit_two_delta")], [0, 1]
    )


def test_duel_choice_comparison_groups_by_independent_map() -> None:
    position = embed_position(duel_position())

    report = compare_choices([position], [position.reply_index], [0])

    assert report["changed_choices"] == 1
    assert report["better"] == 0
    assert report["worse"] == 1
    assert report["by_seat"][0]["worse"] == 1
    weight = torch.zeros(len(FEATURE_NAMES))
    weight[0] = 1
    assert pairwise_agreement([position], weight, torch.ones_like(weight)) == {
        "informative_pairs": 1,
        "concordant": 1,
        "discordant": 0,
        "tied": 0,
        "indistinguishable_features": 0,
    }


def test_identical_features_have_identical_prediction_scores() -> None:
    position = embed_position(duel_position())
    identical = replace(
        position,
        position=replace(
            position.position,
            features=position.position.features[[0, 0]],
        ),
    )
    weight = torch.arange(len(FEATURE_NAMES), dtype=torch.float32)

    scores = prediction_scores(identical, weight, torch.ones_like(weight))

    assert scores[0] == scores[1]


def test_teacher_choice_training_balances_static_and_override_positions() -> None:
    override = embed_position(duel_position())
    static = embed_position(
        replace(duel_position(), opponent_reply_scores=np.asarray([200, 100]))
    )

    differences, weights = teacher_choice_examples([override, static])
    chosen, alternatives, pair_weights = teacher_choice_pairs([override, static])

    assert differences.shape == (2, len(FEATURE_NAMES))
    np.testing.assert_allclose(weights.numpy(), [0.5, 0.5])
    torch.testing.assert_close(chosen - alternatives, differences)
    torch.testing.assert_close(pair_weights, weights)
    static_weight = torch.zeros(len(FEATURE_NAMES))
    static_weight[0] = 1
    report = agreement(
        [override, static], static_weight, torch.ones_like(static_weight)
    )
    assert report["teacher_matches"] == 1
    assert report["static_matches"] == 1
    assert report["teacher_overrides"] == 1
    assert report["correct_overrides"] == 0
    assert agreement_for_choices([override, static], [1, 0])["teacher_matches"] == 2


def test_nonlinear_teacher_scorer_fits_separable_turn_choices() -> None:
    torch.set_num_threads(1)
    base = embed_position(duel_position())

    def position(seed: int, first: float, second: float, selected: int):
        features = torch.zeros((2, len(FEATURE_NAMES)))
        features[:, 0] = torch.as_tensor([first, second])
        return replace(
            base,
            position=replace(base.position, seed=seed, features=features),
            reply_index=selected,
        )

    training = [
        position(71, 0, 1, 1),
        position(72, 2, 0, 0),
        position(73, 0, 1, 1),
        position(74, 2, 0, 0),
    ]
    scorer, mean, scale, losses, pair_count = train_scorer(training, epochs=20)

    assert pair_count == 4
    assert losses[-1] < losses[0]
    assert choices(training, scorer, mean, scale) == [1, 0, 1, 0]


def test_spatial_embedding_uses_root_seat_and_preserves_slate_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        teacher_choice,
        "encode_rules_batch",
        lambda rules, device: torch.zeros((len(rules), 1), device=device),
    )

    class Encoder:
        def forward_with_value_features(self, observation, rules):
            np.testing.assert_array_equal(observation["active_players"], [0, 0])
            assert rules.shape == (2, 1)
            return None, None, torch.as_tensor([[1.0, 2.0], [3.0, 4.0]])

    embedded = teacher_choice.embed_spatial(duel_position(), Encoder())

    assert embedded.position.features.shape == (2, len(FEATURE_NAMES) + 2)
    np.testing.assert_array_equal(
        embedded.position.features[:, -2:].numpy(), [[1, 2], [3, 4]]
    )
