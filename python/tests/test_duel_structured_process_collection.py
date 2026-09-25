import pytest

pytest.importorskip("torch")

from antiyoy_rl import VectorEnv
from python.audit_duel_first_regret import native_teacher_action
from python.audit_duel_structured_process_calibration import MAXIMUM_ACTIONS_PER_TURN
from python.audit_duel_three_turn_intervention import FOLLOWUP_NODES
from python.collect_duel_structured_process import replay_record, state_record


def test_fit_record_matches_exact_native_teacher_choice() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=29)
    root = int(environment.observe()["active_players"][0])
    teacher = int(native_teacher_action(environment, FOLLOWUP_NODES, True)[0])

    record = state_record(environment, 29, root, "teacher", 0, teacher)

    assert record["root_seat"] == root
    assert record["candidates"][record["teacher_selected_index"]]["plan"][0] == teacher
    assert 1 <= len(record["candidates"]) <= 8
    assert all(
        len(candidate["post_components"]) == 16 for candidate in record["candidates"]
    )
    assert all(
        len(candidate["trace"]) == len(candidate["plan"]) <= MAXIMUM_ACTIONS_PER_TURN
        for candidate in record["candidates"]
    )


def test_response_replay_rejects_changed_followup_components() -> None:
    environment = VectorEnv(1, width=7, height=5, seed=29)
    root = int(environment.observe()["active_players"][0])
    plans, scores, _, _, posts = environment.search_turn_plan_process(
        node_budget=64, slate_size=4
    )
    _, targets = environment.search_turn_response_targets(
        node_budget=64,
        reply_nodes=32,
        followup_nodes=16,
        slate_size=4,
    )
    targets[0][0][3][0] += 1

    with pytest.raises(ValueError, match="followup components disagree"):
        replay_record(environment, plans[0], scores[0], posts[0], targets[0], root)
