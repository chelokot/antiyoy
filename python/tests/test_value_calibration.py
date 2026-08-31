from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import UniversalPolicy, encode_rules, select_environments
from python.calibrate_value import (
    ValueSample,
    calibrate_value,
    train_value_head,
    value_metrics,
)
from python.tests.test_bundle import write_checkpoint


def test_value_metrics_reports_error_sign_and_correlation() -> None:
    metrics = value_metrics(
        np.array([0.8, -0.4, 0.1]),
        np.array([1.0, -1.0, 0.0]),
    )

    assert metrics["samples"] == 3
    assert metrics["mae"] == pytest.approx(0.3)
    assert metrics["sign_accuracy"] == pytest.approx(2 / 3)
    assert metrics["correlation"] > 0.9


def test_value_training_changes_only_the_value_head() -> None:
    environment = VectorEnv(2, width=7, height=5, seed=701, action_limit=8)
    observation = environment.observe()
    model = UniversalPolicy(hidden=16, layers=1)
    samples = [
        ValueSample(select_environments(observation, [0]), 1.0),
        ValueSample(select_environments(observation, [1]), -1.0),
    ]
    before = {key: value.clone() for key, value in model.state_dict().items()}

    report = train_value_head(
        model,
        [samples[0]],
        [samples[1]],
        encode_rules(environment.rules_jsons()[0], torch.device("cpu")),
        epochs=2,
        batch_size=1,
        learning_rate=1e-2,
        seed=703,
    )

    assert len(report["epoch_losses"]) == 2
    assert any(
        not torch.equal(value, model.state_dict()[key])
        for key, value in before.items()
        if key.startswith("value_head.")
    )
    assert all(
        torch.equal(value, model.state_dict()[key])
        for key, value in before.items()
        if not key.startswith("value_head.")
    )


def test_calibration_writes_an_overlay_compatible_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "calibrated.pt"
    write_checkpoint(source, 1.0)

    report = calibrate_value(
        source,
        output,
        profile="classic_generic_2022",
        games=2,
        validation_games=1,
        seed=709,
        device_name="cpu",
        width=7,
        height=5,
        action_limit=6,
        sample_stride=1,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        exploration_probability=1.0,
        exploration_top_k=3,
    )
    calibrated = torch.load(output, map_location="cpu", weights_only=False)
    original = torch.load(source, map_location="cpu", weights_only=False)

    assert output.is_file()
    assert report["policy_parameters_frozen"] is True
    assert report["collection"]["exploratory_actions"] > 0
    assert calibrated["checkpoint_version"] == original["checkpoint_version"]
    assert all(
        torch.equal(value, calibrated["model"][key])
        for key, value in original["model"].items()
        if not key.startswith("value_head.")
    )


def test_calibration_supports_procedural_multiplayer(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "procedural-calibrated.pt"
    write_checkpoint(source, 1.0)

    report = calibrate_value(
        source,
        output,
        profile="classic_generic_2022",
        games=3,
        validation_games=1,
        seed=719,
        device_name="cpu",
        width=9,
        height=7,
        action_limit=12,
        sample_stride=1,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        exploration_probability=0.5,
        exploration_top_k=3,
        generator="procedural_v1",
        players=3,
        starting_province_size=3,
    )

    assert report["generator"] == "procedural_v1"
    assert report["domain_descriptor"]["players"] == 3
    assert report["collection"]["games"] == 3
    assert len(report["collection"]["wins_by_seat"]) == 3
    assert output.is_file()
