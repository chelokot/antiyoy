from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import encode_rules_batch

from .audit_duel_first_regret import (
    CHECKPOINT_SHA256,
    create_environment,
    load_routed_policy,
    native_teacher_action,
)
from .audit_duel_structured_process_calibration import (
    ACTION_INDICES,
    MAXIMUM_ACTIONS_PER_TURN,
    PROTOCOL,
)
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .build_bundle import digest


FIRST_SEED = 6600000
MAPS = 128


def terminal_class(environment: VectorEnv, root: int) -> int:
    if not environment.done()[0]:
        return 1
    return 2 if int(environment.position_scores(root)[0]) > 0 else 0


def replay_record(
    environment: VectorEnv,
    plans: list[list[int]],
    scores: list[int],
    posts: list[list[int]],
    targets: list[tuple],
    root: int,
    branch_action_limit: int | None = None,
) -> None:
    for plan, static, post, target in zip(plans, scores, posts, targets, strict=True):
        reply, followup, reply_components, final_components, classes, score = target
        branch = environment.fork(
            np.asarray([0], dtype=np.uint64), action_limit=branch_action_limit
        )
        for stage, actions in enumerate((plan, reply, followup)):
            for action in actions:
                legal = int(branch.observe()["action_offsets"][1])
                if not 0 <= action < legal:
                    raise ValueError("searched continuation action is illegal")
                result = branch.step(np.asarray([action], dtype=np.uint64))
                if bool(result["truncated"][0]):
                    raise ValueError("searched continuation was action-limit censored")
            if terminal_class(branch, root) != classes[stage]:
                raise ValueError("searched continuation terminal class disagrees")
            if stage == 0:
                if post != branch.position_components(root)[0]:
                    raise ValueError("searched root post-turn components disagree")
                if static != int(branch.position_scores(root)[0]):
                    raise ValueError("searched root static score disagrees")
            elif stage == 1:
                if reply_components != branch.position_components(root)[0]:
                    raise ValueError("searched opponent-reply components disagree")
        if final_components != branch.position_components(root)[0]:
            raise ValueError("searched root-followup components disagree")
        if score != int(branch.position_scores(root)[0]):
            raise ValueError("searched root-followup score disagrees")


def state_record(
    environment: VectorEnv,
    seed: int,
    root: int,
    opponent: str,
    own_action_index: int,
    teacher_action: int,
    branch_action_limit: int | None = None,
) -> dict:
    plans, static, traces, roots, posts = environment.search_turn_plan_process(
        node_budget=256,
        slate_size=8,
        beam_width=32,
        branch_width=48,
        maximum_actions_per_turn=MAXIMUM_ACTIONS_PER_TURN,
    )
    response_plans, targets = environment.search_turn_response_targets(
        node_budget=256,
        reply_nodes=64,
        followup_nodes=32,
        slate_size=8,
        beam_width=32,
        branch_width=48,
        maximum_actions_per_turn=MAXIMUM_ACTIONS_PER_TURN,
    )
    if plans != response_plans:
        raise ValueError("process and response root slates disagree")
    if roots[0] != environment.position_components(root)[0]:
        raise ValueError("root components disagree with live state")
    replay_record(
        environment,
        plans[0],
        static[0],
        posts[0],
        targets[0],
        root,
        branch_action_limit,
    )
    chosen = max(
        range(len(plans[0])),
        key=lambda index: (targets[0][index][5], static[0][index], -index),
    )
    if plans[0][chosen][0] != teacher_action:
        raise ValueError("response target ranking disagrees with native teacher")
    candidates = [
        {
            "plan": plan,
            "trace": trace,
            "static_score": score,
            "post_components": post,
            "reply_plan": target[0],
            "followup_plan": target[1],
            "reply_components": target[2],
            "followup_components": target[3],
            "terminal_classes": target[4],
            "response_score": target[5],
        }
        for plan, trace, score, post, target in zip(
            plans[0], traces[0], static[0], posts[0], targets[0], strict=True
        )
    ]
    return {
        "type": "sample",
        "seed": seed,
        "root_seat": root,
        "opponent": opponent,
        "own_action_index": own_action_index,
        "root_components": roots[0],
        "teacher_selected_index": chosen,
        "candidates": candidates,
    }


def collect(
    source_path: Path,
    output: Path,
    first_seed: int = FIRST_SEED,
    maps: int = MAPS,
    protocol: str = PROTOCOL,
    allow_rollin_censor: bool = False,
    branch_action_limit: int | None = None,
) -> dict:
    if digest(source_path) != CHECKPOINT_SHA256:
        raise ValueError("frozen source checkpoint disagrees with the protocol")
    torch.set_num_threads(1)
    source, experts = load_routed_policy(source_path)
    temporary = output.with_name(output.name + ".tmp")
    games = 0
    samples = 0
    candidates = 0
    terminal = 0
    censored = 0
    with (
        torch.inference_mode(),
        temporary.open("wb") as compressed,
        gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as encoded,
        io.TextIOWrapper(encoded, encoding="utf-8") as raw,
    ):
        raw.write(
            json.dumps(
                {
                    "type": "header",
                    "kind": "structured_multi_turn_response_process_fit",
                    "protocol": protocol,
                    "source_sha256": CHECKPOINT_SHA256,
                    "first_seed": first_seed,
                    "maps": maps,
                    "experts": experts,
                },
                sort_keys=True,
            )
            + "\n"
        )
        for seed in range(first_seed, first_seed + maps):
            for root in (0, 1):
                for opponent in ("source", "teacher"):
                    environment = create_environment(seed)
                    rules = encode_rules_batch(
                        environment.rules_jsons(), torch.device("cpu")
                    )
                    own_action_index = 0
                    game_records = []
                    steps = 0
                    result = None
                    while not environment.done()[0]:
                        observation = environment.observe()
                        active = int(observation["active_players"][0])
                        if active == root:
                            action = native_teacher_action(
                                environment, FOLLOWUP_NODES, True
                            )
                            if own_action_index in ACTION_INDICES:
                                record = state_record(
                                    environment,
                                    seed,
                                    root,
                                    opponent,
                                    own_action_index,
                                    int(action[0]),
                                    branch_action_limit,
                                )
                                game_records.append(record)
                            own_action_index += 1
                        elif opponent == "teacher":
                            action = native_teacher_action(
                                environment, FOLLOWUP_NODES, True
                            )
                        else:
                            action = source.actions(observation, rules)
                        result = environment.step(np.asarray(action, dtype=np.uint64))
                        steps += 1
                    if result is None:
                        raise ValueError("fit game ended without an action")
                    is_terminal = bool(result["terminal"][0])
                    is_censored = bool(result["truncated"][0])
                    if not is_terminal and not is_censored:
                        raise ValueError(
                            "fit collection game ended without terminal or censor"
                        )
                    for record in game_records:
                        if allow_rollin_censor:
                            record["rollin_censored"] = is_censored
                        raw.write(json.dumps(record, sort_keys=True) + "\n")
                        samples += 1
                        candidates += len(record["candidates"])
                    game_record = {
                        "type": "game",
                        "seed": seed,
                        "root_seat": root,
                        "opponent": opponent,
                        "steps": steps,
                        "samples": len(game_records),
                        "terminal": is_terminal,
                        "truncated": is_censored,
                        "winner": int(result["winners"][0]),
                    }
                    if allow_rollin_censor:
                        game_record["adjudicated_winner"] = (
                            int(result["adjudicated_winners"][0])
                            if is_censored
                            else None
                        )
                    raw.write(json.dumps(game_record, sort_keys=True) + "\n")
                    games += 1
                    terminal += is_terminal
                    censored += is_censored
                    if is_censored and not allow_rollin_censor:
                        raise ValueError("fit collection game was not terminal")
    if games != maps * 4:
        raise ValueError("fit collection missed a predeclared game")
    os.replace(temporary, output)
    with output.open("rb") as raw:
        raw_sha256 = hashlib.file_digest(raw, "sha256").hexdigest()
    return {
        "kind": "structured_multi_turn_response_process_fit_collection",
        "protocol": protocol,
        "first_seed": first_seed,
        "maps": maps,
        "games": games,
        "terminal_games": terminal,
        "censored_games": censored,
        "samples": samples,
        "candidate_branches": candidates,
        "raw_path": str(output),
        "raw_bytes": output.stat().st_size,
        "raw_sha256": raw_sha256,
        "qualification": "Fit data only; no model trained or validation/game outcome claimed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(collect(arguments.source, arguments.output), sort_keys=True))


if __name__ == "__main__":
    main()
