from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from antiyoy_rl.vector_value import VECTOR_VALUE_ARTIFACT_KIND
from python.calibrate_vector_value import (
    VectorOutcomeCalibrationConfig,
    calibrate_vector_outcomes,
    relative_outcome_targets,
    validate_config,
)
from python.tests.test_bundle import write_checkpoint


def test_relative_outcome_targets_follow_the_active_player_order() -> None:
    assert relative_outcome_targets(1, 2, 4, "binary").tolist() == [
        -1,
        -1,
        -1,
        1,
    ]
    assert relative_outcome_targets(3, 3, 4, "zero_sum").tolist() == [
        1,
        pytest.approx(-1 / 3),
        pytest.approx(-1 / 3),
        pytest.approx(-1 / 3),
    ]
    assert torch.count_nonzero(relative_outcome_targets(255, 1, 4, "binary")) == 0


def test_outcome_calibration_writes_a_checkpoint_bound_vector_head(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "outcomes.pt"
    write_checkpoint(source, 1.0)

    report = calibrate_vector_outcomes(
        source,
        output,
        VectorOutcomeCalibrationConfig(
            players=3,
            games=4,
            validation_games=1,
            seed=1_111_000,
            device="cpu",
            width=9,
            height=7,
            action_limit=12,
            starting_province_size=3,
            sample_stride=2,
            epochs=2,
            batch_size=16,
            learning_rate=1e-3,
        ),
    )

    artifact = torch.load(output, map_location="cpu", weights_only=False)
    assert artifact["kind"] == VECTOR_VALUE_ARTIFACT_KIND
    assert artifact["source"]["sha256"] == report["source"]["sha256"]
    assert report["training_games"] == 3
    assert report["validation_games"] == 1
    assert report["training_samples"] > 0
    assert report["validation_samples"] > 0
    assert report["calibration"]["validation_after"]["labels"] == (
        report["validation_samples"] * 2
    )


@pytest.mark.parametrize(
    "config",
    [
        VectorOutcomeCalibrationConfig(players=1),
        VectorOutcomeCalibrationConfig(players=9),
        VectorOutcomeCalibrationConfig(games=1),
        VectorOutcomeCalibrationConfig(validation_games=0),
        VectorOutcomeCalibrationConfig(validation_games=320),
        VectorOutcomeCalibrationConfig(sample_stride=0),
        VectorOutcomeCalibrationConfig(epochs=0),
        VectorOutcomeCalibrationConfig(batch_size=0),
        VectorOutcomeCalibrationConfig(learning_rate=0),
        VectorOutcomeCalibrationConfig(exploration_probability=1.1),
        VectorOutcomeCalibrationConfig(exploration_top_k=1),
        VectorOutcomeCalibrationConfig(target_mode="unknown"),
    ],
)
def test_outcome_calibration_rejects_invalid_config(
    config: VectorOutcomeCalibrationConfig,
) -> None:
    with pytest.raises(ValueError):
        validate_config(config)
