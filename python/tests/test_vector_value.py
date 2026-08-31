import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import UniversalPolicy, encode_rules
from antiyoy_rl.routed import RoutedPolicy
from antiyoy_rl.vector_value import (
    VECTOR_VALUE_ARTIFACT_KIND,
    VECTOR_VALUE_ARTIFACT_VERSION,
    RelativeValueHead,
    initialize_from_scalar_value_head,
    load_relative_value_head,
    relative_to_absolute_utilities,
)


def test_relative_head_emits_all_supported_player_slots() -> None:
    head = RelativeValueHead(hidden=16)

    values = head(torch.zeros((3, 32)))

    assert values.shape == (3, 8)


def test_relative_utilities_rotate_into_absolute_seat_order() -> None:
    relative = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15, 16, 17],
            [20, 21, 22, 23, 24, 25, 26, 27],
        ]
    )

    absolute = relative_to_absolute_utilities(
        relative,
        np.array([2, 1], dtype=np.uint8),
        np.array([3, 2], dtype=np.uint8),
    )

    assert absolute.tolist() == [11, 12, 10, 21, 20]


def test_vector_head_initializes_every_slot_from_the_scalar_value() -> None:
    policy = UniversalPolicy(hidden=16, layers=1)
    head = RelativeValueHead(hidden=16)

    initialize_from_scalar_value_head(head, policy.value_head)
    features = torch.randn((4, 32))

    scalar = policy.value_head(features)
    vector = head(features)
    torch.testing.assert_close(vector, scalar.expand(-1, 8))


def test_vector_head_artifact_loader_validates_kind_and_architecture() -> None:
    original = RelativeValueHead(hidden=16)
    artifact = {
        "kind": VECTOR_VALUE_ARTIFACT_KIND,
        "artifact_version": VECTOR_VALUE_ARTIFACT_VERSION,
        "architecture": {"hidden": 16},
        "model": original.state_dict(),
    }

    restored = load_relative_value_head(artifact, 16, torch.device("cpu"))

    for key, value in original.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)
    with pytest.raises(ValueError, match="architecture"):
        load_relative_value_head(artifact, 32, torch.device("cpu"))


def test_routed_maxn_preserves_the_exact_active_player_value() -> None:
    environment = VectorEnv(3, width=7, height=5, seed=411)
    observation = environment.observe()
    rules = encode_rules(environment.rules_json(), torch.device("cpu"))
    model = UniversalPolicy(hidden=16, layers=1)
    policy = RoutedPolicy({"shared": model}, ["shared", "shared"])
    head = RelativeValueHead(hidden=16)

    _, scalar_values = policy(observation, rules)
    _, utilities = policy.maxn(observation, rules, head)
    offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum(observation["player_counts"], dtype=np.int64),
        )
    )
    active_indices = offsets[:-1] + observation["active_players"]

    assert torch.equal(utilities[torch.as_tensor(active_indices)], scalar_values)
