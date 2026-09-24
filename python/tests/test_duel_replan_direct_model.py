from __future__ import annotations

from typing import cast

import pytest

from python import audit_duel_replan_direct_model
from python.audit_duel_replan_direct_model import ReplannedDirectReport


def test_replanned_direct_audit_rejects_cached_baseline() -> None:
    raw = cast(ReplannedDirectReport, {"replan_reply_search": False})
    with pytest.raises(ValueError, match="did not use replanned"):
        audit_duel_replan_direct_model.audit(raw)


def test_replanned_direct_gate_requires_terminal_games_and_both_seats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = cast(ReplannedDirectReport, {"replan_reply_search": True})
    monkeypatch.setattr(
        audit_duel_replan_direct_model,
        "audit_direct",
        lambda report, seed, maps: {
            "nonterminal_games": 1,
            "native_wins_by_model_seat": [80, 40],
            "direct_model_wins_by_model_seat": [48, 88],
            "paired_independent_maps": {
                "candidate_better": 60,
                "baseline_better": 20,
                "exact_two_sided_sign_test_p": 0.001,
            },
            "map_bootstrap_95_head_to_head_elo": [10.0, 100.0],
        },
    )

    result = audit_duel_replan_direct_model.audit(raw)

    assert result["gate"]["all_512_games_terminal"] is False
    assert result["gate"]["replanned_better_in_both_seats"] is False
    assert result["direct_model_gate_passed"] is False
