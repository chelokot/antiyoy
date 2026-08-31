from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from antiyoy_rl.vector_value import VECTOR_VALUE_ARTIFACT_KIND
from python.distill_vector_value import (
    VectorValueDistillationConfig,
    distill_vector_value,
    validate_config,
)
from python.tests.test_bundle import write_checkpoint


def test_vector_value_distillation_writes_a_compatible_head(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "vector-value.pt"
    write_checkpoint(source, 1.0)

    report = distill_vector_value(
        source,
        output,
        VectorValueDistillationConfig(
            players=3,
            environments=2,
            updates=3,
            validation_environments=2,
            validation_steps=2,
            seed=1_081_000,
            device="cpu",
            width=9,
            height=7,
            action_limit=16,
            starting_province_size=3,
            learning_rate=1e-3,
        ),
    )

    artifact = torch.load(output, map_location="cpu", weights_only=False)
    assert artifact["kind"] == VECTOR_VALUE_ARTIFACT_KIND
    assert artifact["architecture"]["hidden"] == 16
    assert artifact["source"]["sha256"] == report["source"]["sha256"]
    assert report["encoder_pass_reduction"] == 3
    assert report["training"]["labels"] == 18
    assert report["validation"]["initial"]["labels"] == 12
    assert report["validation"]["final"]["labels"] == 12


@pytest.mark.parametrize(
    "config",
    [
        VectorValueDistillationConfig(players=1),
        VectorValueDistillationConfig(players=9),
        VectorValueDistillationConfig(environments=0),
        VectorValueDistillationConfig(updates=0),
        VectorValueDistillationConfig(validation_steps=0),
        VectorValueDistillationConfig(learning_rate=0),
        VectorValueDistillationConfig(land_density_per_million=1_000_001),
    ],
)
def test_vector_value_distillation_rejects_invalid_config(
    config: VectorValueDistillationConfig,
) -> None:
    with pytest.raises(ValueError):
        validate_config(config)
