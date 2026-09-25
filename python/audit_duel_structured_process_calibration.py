from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from antiyoy_rl import SCORE_COMPONENT_WEIGHTS, VectorEnv
from antiyoy_rl.model import encode_rules_batch

from .audit_duel_first_regret import (
    CHECKPOINT_SHA256,
    create_environment,
    load_routed_policy,
    native_teacher_action,
)
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .audit_duel_turn_trace_process_feasibility import percentile
from .build_bundle import digest


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-structured-response-process-distillation-v1.json"
FIRST_SEED = 6601000
MAPS = 8
ACTION_INDICES = frozenset((0, 1, 2, 4, 8, 16, 32, 64, 128, 256))
MAXIMUM_ACTIONS_PER_TURN = 24
COMPONENT_SCALES = np.asarray([20] * 12 + [50, 20, 20, 5], dtype=np.float32)


class StructuredResponseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent = nn.GRU(16, 32, batch_first=True)
        self.hidden = nn.Linear(65, 32)
        self.readout = nn.Linear(32, 35)

    def forward(
        self,
        traces: torch.Tensor,
        lengths: torch.Tensor,
        root: torch.Tensor,
        post: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states, _ = self.recurrent(traces)
        last = states[torch.arange(len(lengths)), lengths - 1]
        features = torch.cat((last, root, post, static), dim=1)
        result = self.readout(torch.relu(self.hidden(features)))
        return result[:, :32].reshape(-1, 2, 16), result[:, 32:]


def selector(
    environment: VectorEnv, model: StructuredResponseModel
) -> tuple[dict[str, float | int], list[list[int]], list[int], list[list[int]]]:
    started = time.perf_counter()
    plans, scores, traces, roots, posts = environment.search_turn_plan_process(
        node_budget=256,
        slate_size=8,
        beam_width=32,
        branch_width=48,
        maximum_actions_per_turn=MAXIMUM_ACTIONS_PER_TURN,
    )
    slate_seconds = time.perf_counter() - started
    started = time.perf_counter()
    lengths = torch.as_tensor([len(trace) for trace in traces[0]], dtype=torch.int64)
    tokens = np.zeros((len(traces[0]), MAXIMUM_ACTIONS_PER_TURN, 16), dtype=np.float32)
    for index, trace in enumerate(traces[0]):
        tokens[index, : len(trace)] = np.asarray(trace, dtype=np.float32)
    post = np.asarray(posts[0], dtype=np.float32)
    normalized_post = post / COMPONENT_SCALES
    normalized_root = np.broadcast_to(
        np.asarray(roots[0], dtype=np.float32) / COMPONENT_SCALES,
        normalized_post.shape,
    )
    response, terminal = model(
        torch.from_numpy(tokens),
        lengths,
        torch.from_numpy(normalized_root.copy()),
        torch.from_numpy(normalized_post),
        torch.as_tensor(scores[0], dtype=torch.float32).reshape(-1, 1) / 1000,
    )
    followup = post + response[:, 1].numpy() * COMPONENT_SCALES
    probabilities = torch.softmax(terminal, dim=1).numpy()
    predicted = followup @ np.asarray(SCORE_COMPONENT_WEIGHTS, dtype=np.float32)
    predicted += 1000 * (probabilities[:, 2] - probabilities[:, 0])
    terminal_candidates = np.abs(np.asarray(scores[0], dtype=np.int64)) >= 10**12
    predicted[terminal_candidates] = np.asarray(scores[0], dtype=np.float32)[
        terminal_candidates
    ]
    choice = max(
        range(len(plans[0])),
        key=lambda index: (float(predicted[index]), scores[0][index], -index),
    )
    if not np.isfinite(predicted).all():
        raise ValueError("process selector produced a nonfinite candidate score")
    model_seconds = time.perf_counter() - started
    return (
        {
            "slate_seconds": slate_seconds,
            "model_seconds": model_seconds,
            "candidate_plans": len(plans[0]),
            "candidate_tokens": sum(len(trace) for trace in traces[0]),
            "selected_candidate": choice,
        },
        plans[0],
        scores[0],
        posts[0],
    )


def validate_candidates(
    environment: VectorEnv,
    plans: list[list[int]],
    scores: list[int],
    posts: list[list[int]],
    root: int,
) -> None:
    for plan, score, components in zip(plans, scores, posts, strict=True):
        branch = environment.fork(np.asarray([0], dtype=np.uint64))
        for action in plan:
            legal = int(branch.observe()["action_offsets"][1])
            if not 0 <= action < legal:
                raise ValueError("native candidate action is illegal on replay")
            result = branch.step(np.asarray([action], dtype=np.uint64))
            if bool(result["truncated"][0]):
                raise ValueError("native candidate was action-limit censored")
        if components != branch.position_components(root)[0]:
            raise ValueError("native candidate components disagree with replay")
        if score != int(branch.position_scores(root)[0]):
            raise ValueError("native candidate score disagrees with replay")
        if not branch.done()[0] and int(branch.observe()["active_players"][0]) == root:
            raise ValueError("native candidate did not complete the root turn")


def summarize(samples: list[dict[str, float | int | str]], games: list[dict]) -> dict:
    teacher = [float(sample["teacher_seconds"]) for sample in samples]
    candidate = [
        float(sample["slate_seconds"]) + float(sample["model_seconds"])
        for sample in samples
    ]
    seats = [
        (
            statistics.median(
                candidate[index]
                for index, sample in enumerate(samples)
                if sample["root_seat"] == seat
            ),
            statistics.median(
                teacher[index]
                for index, sample in enumerate(samples)
                if sample["root_seat"] == seat
            ),
        )
        for seat in (0, 1)
    ]
    censored = sum(game["truncated"] for game in games)
    gates = {
        "all_arms_terminal": len(games) == MAPS * 4
        and censored == 0
        and all(game["terminal"] for game in games),
        "every_game_sampled": all(game["samples"] > 0 for game in games),
        "candidate_replay_exact": True,
        "median_at_most_eighty_percent_teacher": statistics.median(candidate)
        <= 0.8 * statistics.median(teacher),
        "p95_no_worse_than_teacher": percentile(candidate, 0.95)
        <= percentile(teacher, 0.95),
        "both_seat_medians_at_most_eighty_percent_teacher": all(
            proposed <= 0.8 * native for proposed, native in seats
        ),
    }
    return {
        "games": len(games),
        "samples": len(samples),
        "censored_games": censored,
        "candidate_plans": sum(int(sample["candidate_plans"]) for sample in samples),
        "median_teacher_seconds": statistics.median(teacher),
        "p95_teacher_seconds": percentile(teacher, 0.95),
        "median_selector_seconds": statistics.median(candidate),
        "p95_selector_seconds": percentile(candidate, 0.95),
        "seat_medians_selector_teacher_seconds": seats,
        "gates": gates,
        "advance_gate_passed": all(gates.values()),
    }


def audit(source_path: Path) -> dict:
    if digest(source_path) != CHECKPOINT_SHA256:
        raise ValueError("frozen source checkpoint disagrees with the protocol")
    torch.set_num_threads(1)
    torch.manual_seed(6600000)
    model = StructuredResponseModel().eval()
    with torch.inference_mode():
        model(
            torch.zeros((8, MAXIMUM_ACTIONS_PER_TURN, 16)),
            torch.full((8,), MAXIMUM_ACTIONS_PER_TURN, dtype=torch.int64),
            torch.zeros((8, 16)),
            torch.zeros((8, 16)),
            torch.zeros((8, 1)),
        )
        policy, experts = load_routed_policy(source_path)
        samples: list[dict[str, float | int | str]] = []
        games: list[dict] = []
        for seed in range(FIRST_SEED, FIRST_SEED + MAPS):
            for root in (0, 1):
                for opponent in ("source", "teacher"):
                    environment = create_environment(seed)
                    rules = encode_rules_batch(
                        environment.rules_jsons(), torch.device("cpu")
                    )
                    own_actions = 0
                    game_samples = 0
                    steps = 0
                    result = None
                    while not environment.done()[0]:
                        observation = environment.observe()
                        active = int(observation["active_players"][0])
                        if active == root:
                            if own_actions in ACTION_INDICES:
                                if (seed + root + own_actions) % 2 == 0:
                                    candidate, plans, scores, posts = selector(
                                        environment, model
                                    )
                                    started = time.perf_counter()
                                    action = native_teacher_action(
                                        environment, FOLLOWUP_NODES, True
                                    )
                                    teacher_seconds = time.perf_counter() - started
                                else:
                                    started = time.perf_counter()
                                    action = native_teacher_action(
                                        environment, FOLLOWUP_NODES, True
                                    )
                                    teacher_seconds = time.perf_counter() - started
                                    candidate, plans, scores, posts = selector(
                                        environment, model
                                    )
                                validate_candidates(
                                    environment, plans, scores, posts, root
                                )
                                samples.append(
                                    {
                                        "seed": seed,
                                        "root_seat": root,
                                        "opponent": opponent,
                                        "own_action_index": own_actions,
                                        "teacher_seconds": teacher_seconds,
                                        **candidate,
                                    }
                                )
                                game_samples += 1
                            else:
                                action = native_teacher_action(
                                    environment, FOLLOWUP_NODES, True
                                )
                            own_actions += 1
                        elif opponent == "teacher":
                            action = native_teacher_action(
                                environment, FOLLOWUP_NODES, True
                            )
                        else:
                            action = policy.actions(observation, rules)
                        result = environment.step(np.asarray(action, dtype=np.uint64))
                        steps += 1
                    if result is None:
                        raise ValueError("calibration game ended without actions")
                    games.append(
                        {
                            "seed": seed,
                            "root_seat": root,
                            "opponent": opponent,
                            "steps": steps,
                            "samples": game_samples,
                            "terminal": bool(result["terminal"][0]),
                            "truncated": bool(result["truncated"][0]),
                            "winner": int(result["winners"][0]),
                        }
                    )
    return {
        "kind": "structured_multi_turn_response_process_resource_calibration",
        "protocol": PROTOCOL,
        "source_sha256": CHECKPOINT_SHA256,
        "selected_experts_by_seat": experts,
        "summary": summarize(samples, games),
        "samples": samples,
        "games": games,
        "qualification": "Random process model feasibility only; no training, autonomous games, strength or Elo",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = json.dumps(audit(arguments.source), sort_keys=True)
    if arguments.output:
        arguments.output.write_text(f"{report}\n")
        print(json.dumps(json.loads(report)["summary"], sort_keys=True))
    else:
        print(report)


if __name__ == "__main__":
    main()
