import pytest

pytest.importorskip("torch")

import torch
from python.audit_duel_reply_margins import Override, gated_choice, margin_prefixes


def test_margin_audit_uses_largest_predictions_and_independent_maps() -> None:
    overrides = [
        Override(margin=float(index), seed=70 + index, seat=index % 2, result=1)
        for index in range(1, 11)
    ]
    overrides[-2] = Override(margin=9.0, seed=79, seat=1, result=-1)

    report = margin_prefixes(overrides)

    assert report["10"]["positions"] == 1
    assert report["10"]["better"] == 1
    assert report["25"]["positions"] == 3
    assert report["25"]["better"] == 2
    assert report["25"]["worse"] == 1
    assert report["25"]["independent_maps"]["candidate_better"] == 2
    assert report["25"]["independent_maps"]["baseline_better"] == 1
    assert report["100"]["positions"] == 10
    assert margin_prefixes([]) == {}


def test_fixed_margin_keeps_static_turn_until_boundary() -> None:
    predictions = torch.as_tensor([0.0, 0.03125, 0.0625])

    assert gated_choice(predictions, 0.0626) == 0
    assert gated_choice(predictions, 0.0625) == 2
