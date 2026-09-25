import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from antiyoy_rl import VectorEnv
from python.audit_duel_structured_process_calibration import (
    StructuredResponseModel,
    selector,
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
