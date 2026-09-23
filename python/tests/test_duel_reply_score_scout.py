import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from python.scout_duel_nonlinear_choice import choices
from python.scout_duel_reply_score import (
    ScoredTurn,
    native_reply_comparison,
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
