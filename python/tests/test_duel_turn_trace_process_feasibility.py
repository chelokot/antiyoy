from __future__ import annotations

import numpy as np
import torch

from python.audit_duel_turn_trace_process_feasibility import (
    TOKEN_WIDTH,
    TurnTraceScorer,
    encode_action,
    summarize,
)


def test_process_token_preserves_action_kind_location_and_root_relation() -> None:
    observation = {
        "widths": np.asarray([3]),
        "heights": np.asarray([2]),
        "action_kinds": np.asarray([1]),
        "action_parameters": np.asarray([2]),
        "action_sources": np.asarray([1]),
        "action_targets": np.asarray([4]),
        "owners": np.asarray([255, 0, 255, 255, 1, 255]),
        "unit_strengths": np.asarray([0, 2, 0, 0, 0, 0]),
        "defenses": np.asarray([0, 0, 0, 0, 3, 0]),
    }
    token = encode_action(observation, 0, root=0, score_delta=25)
    assert token.shape == (TOKEN_WIDTH,)
    assert token[1] == 1
    assert token[7] == 0.5
    assert token[9] == 0.5
    assert token[10] == 1
    assert token[11] == 1
    assert token[12] == -1
    assert token[13] == 0.5
    assert token[14] == 0.75
    assert token[15] == 0.25


def test_process_scorer_ignores_padding_after_last_real_action() -> None:
    torch.manual_seed(7)
    scorer = TurnTraceScorer().eval()
    tokens = torch.zeros((2, 4, TOKEN_WIDTH))
    tokens[:, :2] = 0.5
    tokens[1, 2:] = 10
    with torch.inference_mode():
        scores = scorer(tokens, torch.as_tensor([2, 2]))
    assert torch.allclose(scores[0], scores[1])


def test_feasibility_gate_counts_both_seats() -> None:
    samples: list[dict[str, float | int]] = [
        {
            "seat": seat,
            "plans": 8,
            "tokens": 24,
            "slate_seconds": 0.0002,
            "trace_seconds": 0.0004,
            "gru_seconds": 0.0003,
            "teacher_seconds": 0.001,
        }
        for seat in (0, 1)
        for _ in range(8)
    ]
    summary = summarize(samples)
    assert summary["advance_gate_passed"] is True
    assert summary["plans_replayed"] == 128
    samples[0]["gru_seconds"] = 0.01
    assert summarize(samples)["advance_gate_passed"] is False
