from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from python.audit_turn_intervention import AuditConfig, audit
from python.tests.test_bundle import write_checkpoint


def test_whole_turn_intervention_reports_deterministic_censored_pairs(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source.pt"
    write_checkpoint(checkpoint, 1.0)
    config = AuditConfig(
        checkpoint=checkpoint,
        maps=2,
        seed=93_000,
        target_round=0,
        horizon=8,
        width=7,
        height=5,
        action_limit=24,
        search_nodes=32,
        reply_search_nodes=8,
        reply_slate_size=4,
        search_beam_width=12,
        search_branch_width=20,
        search_maximum_actions_per_turn=12,
    )

    first = audit(config)
    second = audit(config)

    assert first["records"] == second["records"]
    assert first["positions"] == len(first["records"])
    assert first["positions"] >= 1
    assert first["positions"] == (
        first["better"] + first["worse"] + first["same_outcome"] + first["censored"]
    )
    for record in first["records"]:
        assert record["seed"] in (93_000, 93_001)
        assert record["seat"] in (0, 1)
        assert record["source"]["actions"] <= config.horizon
        assert record["teacher"]["actions"] <= config.horizon
        assert (record["outcome_delta"] is None) == (
            record["source"]["censored"] or record["teacher"]["censored"]
        )


def test_whole_turn_intervention_rejects_invalid_horizon(tmp_path: Path) -> None:
    config = AuditConfig(checkpoint=tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="horizon must be positive"):
        audit(replace(config, horizon=0))
