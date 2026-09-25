from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .audit_duel_first_regret import create_environment, native_teacher_action
from .audit_duel_three_turn_intervention import FOLLOWUP_NODES
from .audit_duel_turn_trace_process_feasibility import (
    MAXIMUM_ACTIONS_PER_TURN,
    TOKEN_WIDTH,
    TurnTraceScorer,
    candidate_trace,
    percentile,
)


PROTOCOL = "benchmarks/protocols/2026-09-25-duel-native-turn-trace-interface-v1.json"
FIRST_SEED = 6591000
MAPS = 8


def summarize(samples: list[dict[str, float | int]]) -> dict[str, object]:
    pipeline = [
        sum(float(sample[key]) for key in ("slate_trace_seconds", "gru_seconds"))
        for sample in samples
    ]
    seat_medians = [
        statistics.median(
            pipeline[index]
            for index, sample in enumerate(samples)
            if sample["seat"] == seat
        )
        for seat in (0, 1)
    ]
    gate = {
        "all_both_seat_samples": len(samples) == MAPS * 2
        and sorted(int(sample["seat"]) for sample in samples)
        == [0] * MAPS + [1] * MAPS,
        "median_pipeline_at_most_three_ms": statistics.median(pipeline) <= 0.003,
        "p95_pipeline_at_most_six_ms": percentile(pipeline, 0.95) <= 0.006,
        "both_seat_medians_at_most_three_ms": all(
            value <= 0.003 for value in seat_medians
        ),
    }
    return {
        "samples": len(samples),
        "plans_replayed": sum(int(sample["plans"]) for sample in samples),
        "tokens_replayed": sum(int(sample["tokens"]) for sample in samples),
        "median_teacher_seconds": statistics.median(
            float(sample["teacher_seconds"]) for sample in samples
        ),
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
                seat = int(environment.observe()["active_players"][0])

                def trace_pipeline() -> tuple[dict[str, float | int], list, list, list]:
                    started = time.perf_counter()
                    plans, scores, traces = environment.search_turn_plan_traces(
                        node_budget=256,
                        slate_size=8,
                        beam_width=32,
                        branch_width=48,
                        maximum_actions_per_turn=MAXIMUM_ACTIONS_PER_TURN,
                    )
                    slate_trace_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    lengths = torch.as_tensor([len(trace) for trace in traces[0]])
                    tokens = np.zeros(
                        (len(traces[0]), MAXIMUM_ACTIONS_PER_TURN, TOKEN_WIDTH),
                        dtype=np.float32,
                    )
                    for index, trace in enumerate(traces[0]):
                        tokens[index, : len(trace)] = np.asarray(
                            trace, dtype=np.float32
                        )
                    scores_tensor = model(torch.from_numpy(tokens), lengths)
                    if not bool(torch.isfinite(scores_tensor).all()):
                        raise ValueError(
                            "random process scorer returned a nonfinite score"
                        )
                    gru_seconds = time.perf_counter() - started
                    return (
                        {
                            "slate_trace_seconds": slate_trace_seconds,
                            "gru_seconds": gru_seconds,
                            "plans": len(traces[0]),
                            "tokens": sum(len(trace) for trace in traces[0]),
                        },
                        plans,
                        scores,
                        traces,
                    )

                if seat not in seen:
                    if seed % 2 == seat:
                        candidate, plans, scores, traces = trace_pipeline()
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
                        candidate, plans, scores, traces = trace_pipeline()
                    indexed, static = environment.search_turn_plans(
                        node_budget=256,
                        slate_size=8,
                        beam_width=32,
                        branch_width=48,
                        maximum_actions_per_turn=MAXIMUM_ACTIONS_PER_TURN,
                    )
                    if plans != indexed or scores != static:
                        raise ValueError(
                            "native trace plans disagree with indexed search slate"
                        )
                    for plan, score, trace in zip(
                        plans[0], scores[0], traces[0], strict=True
                    ):
                        np.testing.assert_allclose(
                            np.asarray(trace, dtype=np.float32),
                            candidate_trace(environment, plan, score, seat),
                            rtol=0,
                            atol=1e-6,
                        )
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
        "kind": "read_only_native_turn_trace_interface_feasibility",
        "protocol": PROTOCOL,
        "summary": summarize(samples),
        "samples": samples,
        "qualification": "Random process scorer and exact trace interface only; no trained policy, game strength, Elo or browser latency",
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
