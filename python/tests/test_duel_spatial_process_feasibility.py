from antiyoy_rl import ProceduralConfig, VectorEnv
from python.audit_duel_spatial_process_feasibility import (
    replay_candidates,
    summarize,
)


def test_replayed_native_turn_candidates_match_static_scores() -> None:
    environment = VectorEnv.procedural(
        1,
        ProceduralConfig(width=7, height=5, players=2, seed=23),
        profile="classic_generic_2022",
    )
    root = int(environment.observe()["active_players"][0])
    plans, scores = environment.search_turn_plans(node_budget=64, slate_size=4)

    observations, terminal = replay_candidates(environment, plans[0], scores[0], root)

    assert len(observations) + terminal == len(plans[0])
    assert len(plans[0]) > 0
    assert all(int(item["active_players"][0]) != root for item in observations)


def test_latency_gate_compares_same_state_process_time() -> None:
    sample = {
        "teacher_search_seconds": 0.04,
        "root_slate_seconds": 0.01,
        "candidate_replay_seconds": 0.005,
        "batched_encoder_seconds": 0.005,
        "candidate_plans": 8,
        "terminal_candidates": 0,
    }

    summary = summarize([sample] * 16)

    assert summary["advance_gate_passed"] is True
    assert summary["total_candidate_plans"] == 128
