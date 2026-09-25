import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from antiyoy_rl import VectorEnv
from python.audit_duel_structured_process_calibration import (
    StructuredResponseModel,
    selector,
    summarize,
    validate_candidates,
)


def test_process_selector_replays_exact_native_candidate_states() -> None:
    torch.manual_seed(6600000)
    model = StructuredResponseModel().eval()
    environment = VectorEnv(1, width=7, height=5, seed=29)
    root = int(environment.observe()["active_players"][0])

    with torch.inference_mode():
        timings, plans, scores, components = selector(environment, model)
    validate_candidates(environment, plans, scores, components, root)

    assert 1 <= len(plans) <= 8
    assert len(plans) == len(scores) == len(components)
    assert 0 <= timings["selected_candidate"] < len(plans)
    assert all(len(row) == 16 for row in components)
    assert np.isfinite([timings["slate_seconds"], timings["model_seconds"]]).all()


def test_process_selector_rejects_changed_candidate_components() -> None:
    torch.manual_seed(6600000)
    model = StructuredResponseModel().eval()
    environment = VectorEnv(1, width=7, height=5, seed=29)
    root = int(environment.observe()["active_players"][0])

    with torch.inference_mode():
        _, plans, scores, components = selector(environment, model)
    components[0][0] += 1

    with pytest.raises(ValueError, match="components disagree"):
        validate_candidates(environment, plans, scores, components, root)


def test_censored_rollin_is_recorded_without_passing_terminal_gate() -> None:
    samples = [
        {
            "root_seat": seat,
            "teacher_seconds": 1.0,
            "slate_seconds": 0.2,
            "model_seconds": 0.1,
            "candidate_plans": 8,
        }
        for seat in (0, 1)
        for _ in range(16)
    ]
    games = [
        {"terminal": index != 0, "truncated": index == 0, "samples": 1}
        for index in range(32)
    ]

    strict = summarize(samples, games)
    censor_aware = summarize(samples, games, allow_rollin_censor=True)

    assert strict["censored_games"] == censor_aware["censored_games"] == 1
    assert not strict["gates"]["all_arms_terminal"]
    assert not strict["advance_gate_passed"]
    assert censor_aware["gates"]["all_arms_attempted"]
    assert censor_aware["advance_gate_passed"]
