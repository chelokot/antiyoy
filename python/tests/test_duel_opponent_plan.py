from __future__ import annotations

import numpy as np
import torch

from python.train_duel_opponent_plan import action_loss, source_logits


def test_action_loss_respects_local_legal_action_offsets() -> None:
    student = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
    source = torch.zeros(5)
    objective, cross_entropy, retention = action_loss(
        student,
        source,
        np.asarray([0, 2, 5]),
        np.asarray([1, 2]),
    )

    assert torch.isclose(cross_entropy, torch.log(torch.tensor(6.0)) / 2)
    assert torch.isclose(retention, torch.tensor(0.0))
    objective.backward()
    assert student.grad is not None
    assert student.grad[1] < 0
    assert student.grad[4] < 0
    assert torch.isclose(student.grad[:2].sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(student.grad[2:].sum(), torch.tensor(0.0), atol=1e-6)


def test_source_logits_do_not_train_frozen_reference() -> None:
    head = torch.nn.Linear(3, 1)
    residual = torch.nn.Linear(3, 1)
    features = torch.ones((2, 3), requires_grad=True)

    logits = source_logits(features, head, residual)

    assert torch.allclose(logits, (head(features) + residual(features)).squeeze(1))
    assert not logits.requires_grad
