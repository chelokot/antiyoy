from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from antiyoy_rl import VectorEnv

from .audit_duel_first_regret import create_environment, native_teacher_action
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-turn-trace-process-feasibility-v1.json"
FIRST_SEED = 6590000
MAPS = 8
TOKEN_WIDTH = 16
MAXIMUM_ACTIONS_PER_TURN = 24


class TurnTraceScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recurrent = nn.GRU(TOKEN_WIDTH, 32, batch_first=True)
        self.readout = nn.Linear(32, 1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        states, _ = self.recurrent(tokens)
        last = states[torch.arange(len(lengths)), lengths - 1]
        return self.readout(last).squeeze(1)


def owner_relation(owner: int, root: int) -> float:
    if owner == 255:
        return 0.0
    return 1.0 if owner == root else -1.0


def encode_action(
    observation: dict[str, np.ndarray],
    index: int,
    root: int,
    score_delta: int,
) -> np.ndarray:
    width = int(observation["widths"][0])
    height = int(observation["heights"][0])
    cells = width * height
    kind = int(observation["action_kinds"][index])
    source = int(observation["action_sources"][index])
    target = int(observation["action_targets"][index])
    source_valid = source < cells
    target_valid = target < cells and kind != 5
    token = np.zeros(TOKEN_WIDTH, dtype=np.float32)
    token[kind] = 1.0
    token[6] = int(observation["action_parameters"][index]) / 5.0
    if source_valid:
        token[7] = (source % width) / (width - 1)
        token[8] = (source // width) / (height - 1)
        token[11] = owner_relation(int(observation["owners"][source]), root)
        token[13] = int(observation["unit_strengths"][source]) / 4.0
    if target_valid:
        token[9] = (target % width) / (width - 1)
        token[10] = (target // width) / (height - 1)
        token[12] = owner_relation(int(observation["owners"][target]), root)
        token[14] = int(observation["defenses"][target]) / 4.0
    token[15] = score_delta / 100.0
    return token


def candidate_trace(
    environment: VectorEnv,
    plan: list[int],
    native_score: int,
    root: int,
) -> np.ndarray:
    branch = environment.fork(np.asarray([0], dtype=np.uint64))
    tokens = []
    previous_score = int(branch.position_scores(root)[0])
    for index in plan:
        observation = branch.observe()
        legal = int(observation["action_offsets"][1])
        if not 0 <= index < legal:
            raise ValueError("indexed candidate action is not legal")
        result = branch.step(np.asarray([index], dtype=np.uint64))
        if bool(result["truncated"][0]):
            raise ValueError("candidate plan was censored by the action limit")
        score = int(branch.position_scores(root)[0])
        tokens.append(encode_action(observation, index, root, score - previous_score))
        previous_score = score
    if previous_score != native_score:
        raise ValueError("candidate trace disagrees with native static score")
    if not branch.done()[0] and int(branch.observe()["active_players"][0]) == root:
        raise ValueError("candidate trace did not complete the root turn")
    return np.stack(tokens)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def summarize(samples: list[dict[str, float | int]]) -> dict[str, object]:
    pipeline = [
        sum(float(sample[key]) for key in ("slate_seconds", "trace_seconds", "gru_seconds"))
        for sample in samples
    ]
    teacher = [float(sample["teacher_seconds"]) for sample in samples]
    seat_medians = [
        statistics.median(
            pipeline[index] for index, sample in enumerate(samples) if sample["seat"] == seat
        )
        for seat in (0, 1)
    ]
    gate = {
        "all_both_seat_samples": len(samples) == MAPS * 2
        and sorted(int(sample["seat"]) for sample in samples) == [0] * MAPS + [1] * MAPS,
        "median_pipeline_at_most_three_ms": statistics.median(pipeline) <= 0.003,
        "p95_pipeline_at_most_six_ms": percentile(pipeline, 0.95) <= 0.006,
        "both_seat_medians_at_most_three_ms": all(value <= 0.003 for value in seat_medians),
    }
    return {
        "samples": len(samples),
        "plans_replayed": sum(int(sample["plans"]) for sample in samples),
        "tokens_replayed": sum(int(sample["tokens"]) for sample in samples),
        "median_teacher_seconds": statistics.median(teacher),
        "median_pipeline_seconds": statistics.median(pipeline),
        "p95_pipeline_seconds": percentile(pipeline, 0.95),
        "median_pipeline_seconds_by_seat": seat_medians,
        "gate": gate,
        "advance_gate_passed": all(gate.values()),
    }


def audit() -> dict[str, object]:
    torch.set_num_threads(1)
    torch.manual_seed(FIRST_SEED)
    model = TurnTraceScorer().eval()
    with torch.inference_mode():
        model(
            torch.zeros((8, MAXIMUM_ACTIONS_PER_TURN, TOKEN_WIDTH)),
            torch.full((8,), MAXIMUM_ACTIONS_PER_TURN, dtype=torch.int64),
        )
        samples = []
        for seed in range(FIRST_SEED, FIRST_SEED + MAPS):
            environment = create_environment(seed)
            seen = set()
            while not environment.done()[0] and len(seen) < 2:
                observation = environment.observe()
                seat = int(observation["active_players"][0])

                def trace_pipeline() -> dict[str, float | int]:
                    started = time.perf_counter()
                    plans, scores = environment.search_turn_plans(
                        node_budget=256,
                        slate_size=8,
                        beam_width=32,
                        branch_width=48,
                        maximum_actions_per_turn=MAXIMUM_ACTIONS_PER_TURN,
                    )
                    slate_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    traces = [
                        candidate_trace(environment, plan, score, seat)
                        for plan, score in zip(plans[0], scores[0], strict=True)
                    ]
                    lengths = torch.as_tensor([len(trace) for trace in traces])
                    tokens = np.zeros(
                        (len(traces), MAXIMUM_ACTIONS_PER_TURN, TOKEN_WIDTH),
                        dtype=np.float32,
                    )
                    for index, trace in enumerate(traces):
                        tokens[index, : len(trace)] = trace
                    trace_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    scores_tensor = model(torch.from_numpy(tokens), lengths)
                    if not bool(torch.isfinite(scores_tensor).all()):
                        raise ValueError("random process scorer returned a nonfinite score")
                    gru_seconds = time.perf_counter() - started
                    return {
                        "slate_seconds": slate_seconds,
                        "trace_seconds": trace_seconds,
                        "gru_seconds": gru_seconds,
                        "plans": len(traces),
                        "tokens": sum(len(trace) for trace in traces),
                    }

                if seat not in seen:
                    if seed % 2 == seat:
                        candidate = trace_pipeline()
                        started = time.perf_counter()
                        action = native_teacher_action(environment, FOLLOWUP_NODES, True)
                        teacher_seconds = time.perf_counter() - started
                    else:
                        started = time.perf_counter()
                        action = native_teacher_action(environment, FOLLOWUP_NODES, True)
                        teacher_seconds = time.perf_counter() - started
                        candidate = trace_pipeline()
                    samples.append(
                        {
                            "seed": seed,
                            "seat": seat,
                            "teacher_seconds": teacher_seconds,
                            **candidate,
                        }
                    )
                    seen.add(seat)
                else:
                    action = native_teacher_action(environment, FOLLOWUP_NODES, True)
                result = environment.step(action)
                if bool(result["truncated"][0]):
                    raise ValueError("teacher self-play was action-limit censored")
            if seen != {0, 1}:
                raise ValueError("teacher game did not expose both root seats")
    return {
        "kind": "read_only_turn_trace_process_runtime_feasibility",
        "protocol": PROTOCOL,
        "summary": summarize(samples),
        "samples": samples,
        "qualification": "Warm CPU interface feasibility for a randomly initialized process scorer only; no trained policy, game strength, Elo or browser latency",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = json.dumps(audit(), sort_keys=True)
    if arguments.output:
        arguments.output.write_text(f"{report}\n")
        print(json.dumps(json.loads(report)["summary"], sort_keys=True))
    else:
        print(report)


if __name__ == "__main__":
    main()
