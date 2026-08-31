import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from antiyoy_rl.vector_value import (
    RelativeValueHead,
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
