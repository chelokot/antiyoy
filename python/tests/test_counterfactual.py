from pathlib import Path

import numpy as np
import pytest
import torch

from antiyoy_rl import ScenarioObjective, VectorEnv
from antiyoy_rl.counterfactual import CandidateOutcome, rollout_candidates
from python.benchmark_counterfactual import benchmark
from python.distill_puct import PuctDistillationConfig
from python.evaluate_counterfactual import evaluate_intervention, select_intervention
from python.tests.test_bundle import write_checkpoint


class EndTurnPolicy:
    def actions(
        self, observation: dict[str, np.ndarray], rule_features: torch.Tensor
    ) -> np.ndarray:
        return np.zeros(len(observation["active_players"]), dtype=np.uint64)


def test_candidate_rollouts_match_independent_branches() -> None:
    source = VectorEnv(1, width=7, height=5, seed=473, action_limit=8)
    source.step(np.array([0], dtype=np.uint64))
    before = source.observe()
    candidates = np.array([0, int(before["action_offsets"][1]) - 1], dtype=np.uint64)
    outcomes = rollout_candidates(
        source, 0, candidates, EndTurnPolicy(), torch.zeros((1, 45))
    )
    assert len(outcomes) == 2
    assert [outcome.action for outcome in outcomes] == candidates.tolist()
    for outcome in outcomes:
        branch = source.fork(np.array([0], dtype=np.uint64))
        action = outcome.action
        for step in range(outcome.rollout_steps):
            result = branch.step(np.array([action], dtype=np.uint64))
            action = 0
        winner = (
            result["adjudicated_winners"][0]
            if result["truncated"][0]
            else result["winners"][0]
        )
        assert int(winner) == outcome.winner
        assert bool(result["terminal"][0]) == outcome.terminal
        assert bool(result["truncated"][0]) == outcome.truncated
        observation = branch.observe()
        assert outcome.territory == int(
            np.count_nonzero(observation["owners"] == before["active_players"][0])
        )
        assert step + 1 == outcome.rollout_steps
    after = source.observe()
    for name in ("owners", "objects", "active_players", "rounds"):
        np.testing.assert_array_equal(before[name], after[name])


def test_candidate_rollouts_reject_empty_action_vector() -> None:
    source = VectorEnv(1, width=7, height=5)
    with pytest.raises(ValueError, match="nonempty vector"):
        rollout_candidates(
            source,
            0,
            np.array([], dtype=np.uint64),
            EndTurnPolicy(),
            torch.zeros((1, 45)),
        )


def test_horizon_censors_incomplete_counterfactuals() -> None:
    source = VectorEnv(1, width=7, height=5, seed=47, action_limit=8)
    outcomes = rollout_candidates(
        source,
        0,
        np.array([0], dtype=np.uint64),
        EndTurnPolicy(),
        torch.zeros((1, 45)),
        horizon=1,
    )
    assert outcomes[0].winner is None
    assert outcomes[0].censored
    assert not outcomes[0].terminal
    assert not outcomes[0].truncated
    assert outcomes[0].rollout_steps == 1


def test_finished_branch_is_pruned_without_losing_candidate_order() -> None:
    source = VectorEnv(
        1,
        width=7,
        height=5,
        seed=47,
        action_limit=8,
        objective=ScenarioObjective.survive_through_round(0, 1),
    )
    source.step(np.array([0], dtype=np.uint64))
    action_count = int(source.observe()["action_offsets"][1])
    outcomes = rollout_candidates(
        source,
        0,
        np.array([0, action_count - 1], dtype=np.uint64),
        EndTurnPolicy(),
        torch.zeros((1, 45)),
    )
    assert [outcome.action for outcome in outcomes] == [0, action_count - 1]
    assert [outcome.rollout_steps for outcome in outcomes] == [1, 2]
    assert all(outcome.terminal for outcome in outcomes)
    assert all(outcome.winner == 0 for outcome in outcomes)


def test_intervention_prefers_verified_win_then_horizon_territory() -> None:
    outcomes = [
        CandidateOutcome(2, None, False, False, True, 14, 96),
        CandidateOutcome(5, None, False, False, True, 17, 96),
        CandidateOutcome(7, 4, True, False, False, 3, 40),
    ]
    assert select_intervention(outcomes, 4) == 2
    assert select_intervention(outcomes[:2], 4) == 1


def test_counterfactual_scout_and_paired_gate_use_real_engine(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source.pt"
    write_checkpoint(checkpoint, 1.0)
    config = PuctDistillationConfig(
        generator="symmetric_duel_v1",
        players=2,
        environments=2,
        updates=3,
        seed=692_000,
        device="cpu",
        width=7,
        height=5,
        action_limit=8,
    )
    scout = benchmark(checkpoint, config, 1, 2, None, 1)
    assert 0 < scout["sample_count"] <= 6
    assert scout["completed_samples"] == 0
    assert all(len(sample["candidates"]) == 2 for sample in scout["samples"])
    gate = evaluate_intervention(checkpoint, config, 0, 2, 1, 0)
    assert len(gate["baseline_winners"]) == 2
    assert len(gate["candidate_winners"]) == 2
    assert 0 < len(gate["interventions"]) <= 2
    assert (
        gate["paired"]["candidate_better"]
        + gate["paired"]["baseline_better"]
        + gate["paired"]["same"]
        == 2
    )
