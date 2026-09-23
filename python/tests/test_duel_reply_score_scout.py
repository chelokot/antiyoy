import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from antiyoy_rl.slate_dataset import TeacherSlatePosition
from python.scout_duel_nonlinear_choice import choices
from python.scout_duel_reply_score import (
    ScoredTurn,
    native_reply_comparison,
    position_at_stage,
    score_metrics,
    train_scorer,
    training_pairs,
)
from python.scout_duel_turn_value import DuelEmbedding
from python.scout_turn_value import EmbeddedPosition


def scored_turn(seed: int, features: list[float], scores: list[float]) -> ScoredTurn:
    selected = int(np.argmax(scores))
    return ScoredTurn(
        embedding=DuelEmbedding(
            position=EmbeddedPosition(
                seed=seed,
                seat=seed % 2,
                features=torch.as_tensor(features, dtype=torch.float32).reshape(2, 1),
                outcomes=np.asarray([-1, -1]),
                search_index=0,
            ),
            reply_index=selected,
            first_actions=("a", "b"),
        ),
        reply_scores=np.asarray(scores),
        target=torch.as_tensor(scores, dtype=torch.float32),
        terminal_magnitude_scores=0,
    )


def teacher_position(post_reply: dict[str, np.ndarray] | None) -> TeacherSlatePosition:
    return TeacherSlatePosition(
        seed=71,
        seat=0,
        round=4,
        post_turn={"widths": np.asarray([7])},
        post_turn_rules_json=("rules",),
        static_scores=np.asarray([10]),
        outcome_scores=np.asarray([-1]),
        opponent_reply_scores=np.asarray([20]),
        slate_indices=(0,),
        slate_first_actions=("EndTurn",),
        opponent_actions=None,
        post_reply=post_reply,
    )


def test_relative_reply_score_pairs_and_scorer() -> None:
    torch.set_num_threads(1)
    turns = [
        scored_turn(71, [0, 1], [0, 1]),
        scored_turn(72, [2, 0], [2, 0]),
        scored_turn(73, [0, 1], [0, 1]),
        scored_turn(74, [2, 0], [2, 0]),
    ]

    baselines, alternatives, targets, weights = training_pairs(turns)
    torch.testing.assert_close(baselines.flatten(), torch.tensor([0.0, 2, 0, 2]))
    torch.testing.assert_close(alternatives.flatten(), torch.tensor([1.0, 0, 1, 0]))
    torch.testing.assert_close(targets, torch.tensor([1.0, -2, 1, -2]))
    torch.testing.assert_close(weights, torch.full((4,), 0.25))

    scorer, mean, scale, losses, pair_count = train_scorer(turns)
    embeddings = [turn.embedding for turn in turns]

    assert pair_count == 4
    assert losses[-1] < losses[0]
    assert choices(embeddings, scorer, mean, scale) == [1, 0, 1, 0]
    metrics = score_metrics(turns, scorer, mean, scale)
    assert metrics["sign_correct"] == 4
    assert metrics["actual_positive_pairs"] == 2
    assert metrics["predicted_positive_pairs"] == 2
    assert metrics["true_positive_pairs"] == 2
    comparison = native_reply_comparison(turns, [1, 1, 1, 0])
    assert comparison["better"] == 2
    assert comparison["worse"] == 1
    assert comparison["same"] == 1


def test_post_reply_stage_uses_exact_successor_observation() -> None:
    source = teacher_position({"widths": np.asarray([9])})

    assert position_at_stage(source, "post_turn") is source
    successor = position_at_stage(source, "post_reply")
    np.testing.assert_array_equal(successor.post_turn["widths"], [9])
    np.testing.assert_array_equal(successor.static_scores, [10])


def test_post_reply_stage_requires_successor_observation() -> None:
    source = teacher_position(None)

    with pytest.raises(ValueError, match="no post-reply observation"):
        position_at_stage(source, "post_reply")
