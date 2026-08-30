from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from python.distill_puct import (
    PuctDistillationConfig,
    distill_puct,
    validate_config,
)
from python.tests.test_bundle import write_checkpoint


def test_puct_distillation_changes_only_the_action_head(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "distilled.pt"
    write_checkpoint(source, 1.0)

    report = distill_puct(
        source,
        output,
        PuctDistillationConfig(
            environments=2,
            updates=2,
            seed=812_000,
            device="cpu",
            width=7,
            height=5,
            action_limit=12,
            puct_nodes=4,
            puct_leaf_batch_size=8,
        ),
    )

    source_checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    distilled_checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    changed_action_parameters = 0
    for key, source_value in source_checkpoint["model"].items():
        distilled_value = distilled_checkpoint["model"][key]
        if key.startswith("action_head."):
            changed_action_parameters += int(
                not torch.equal(source_value, distilled_value)
            )
        else:
            assert torch.equal(source_value, distilled_value), key
    assert changed_action_parameters == report["changed_action_parameters"]
    assert changed_action_parameters > 0
    assert report["frozen_parameters_preserved"] is True
    assert report["policy_search"]["decisions"] == 4
    assert report["policy_search"]["total_nodes"] == 16
    assert distilled_checkpoint["summary"]["kind"] == (
        "policy_guided_puct_distillation"
    )


@pytest.mark.parametrize(
    "config",
    [
        PuctDistillationConfig(environments=0),
        PuctDistillationConfig(updates=0),
        PuctDistillationConfig(learning_rate=0),
        PuctDistillationConfig(retention_weight=-1),
        PuctDistillationConfig(rollin="unknown"),
        PuctDistillationConfig(target_mode="unknown"),
        PuctDistillationConfig(puct_nodes=1),
    ],
)
def test_puct_distillation_rejects_invalid_configuration(
    config: PuctDistillationConfig,
) -> None:
    with pytest.raises(ValueError):
        validate_config(config)
