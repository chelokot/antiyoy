import pytest
import torch

from python.evaluate_duel_pairwise_response import load_frozen, passes_offline_gate
from python.scout_duel_nonlinear_choice import TurnScorer


def test_frozen_pairwise_checkpoint_requires_matching_encoder(tmp_path) -> None:
    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"encoder")
    checkpoint = tmp_path / "pairwise.pt"
    scorer = TurnScorer(6)
    torch.save(
        {
            "state_dict": scorer.state_dict(),
            "mean": torch.zeros(2),
            "scale": torch.ones(2),
            "encoder_sha256": "wrong",
            "feature_transform": "asinh_static_score",
            "target": "candidate_reply_score_strictly_exceeds_static",
            "selected_threshold": 0.9,
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="encoder"):
        load_frozen(checkpoint, encoder)


def test_offline_gate_requires_both_seats_and_independent_maps() -> None:
    result = {
        "overrides": 30,
        "native_reply_vs_static": {
            "better": 20,
            "worse": 10,
            "by_seat": {
                0: {"better": 12, "worse": 4},
                1: {"better": 8, "worse": 6},
            },
            "independent_maps": {"exact_two_sided_sign_test_p": 0.02},
        },
    }

    assert passes_offline_gate(result)
    result["native_reply_vs_static"]["by_seat"][1]["worse"] = 9
    assert not passes_offline_gate(result)
