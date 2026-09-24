import numpy as np
import torch

from python.train_three_turn_plan import (
    listwise_objective,
    map_position_weights,
    plan_scores,
)


def test_plan_scores_use_every_action_in_each_complete_plan() -> None:
    log_probabilities = torch.tensor([-0.1, -1.5, -0.1, -0.3])

    scores = plan_scores(log_probabilities, np.asarray([0, 2, 4]))

    torch.testing.assert_close(scores, torch.tensor([-0.8, -0.2]))


def test_listwise_objective_rewards_later_action_correction() -> None:
    student = torch.zeros(8, requires_grad=True)
    source = torch.zeros(8)
    objective, cross_entropy, retention = listwise_objective(
        student,
        source,
        np.asarray([0, 2, 4, 6, 8]),
        np.asarray([0, 0, 0, 1]),
        np.asarray([0, 2, 4]),
        1,
    )

    torch.testing.assert_close(cross_entropy, torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(retention, torch.tensor(0.0))
    objective.backward()
    assert student.grad is not None
    assert student.grad[7] < 0
    assert student.grad[2] > 0


def test_map_weights_balance_teacher_override_and_static_positions() -> None:
    weights = map_position_weights(
        [{"selected_index": 0}, {"selected_index": 0}, {"selected_index": 3}]
    )

    assert weights == [0.25, 0.25, 0.5]
