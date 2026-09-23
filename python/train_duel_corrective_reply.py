from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import torch

from antiyoy_rl.slate_dataset import load_corrective_positions, load_teacher_slates

from .build_bundle import digest
from .train_duel_opponent_plan import load_head_student, train_action_head


FIT_SEED = 6350000
EPOCHS = 2


def train(
    old_fit_path: Path,
    corrective_path: Path,
    source_path: Path,
    initial_head_path: Path,
    output_path: Path,
) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.manual_seed(FIT_SEED)
    with gzip.open(corrective_path, "rt", encoding="utf-8") as source_file:
        metadata = json.load(source_file)
    if (
        metadata["source_sha256"] != digest(source_path)
        or metadata["rollout_student_head_sha256"] != digest(initial_head_path)
        or metadata["maps"] != 64
        or metadata["sampled_maps"] != 64
    ):
        raise ValueError("corrective fit data does not match the frozen protocol")
    old = load_teacher_slates(old_fit_path)
    corrective = load_corrective_positions(corrective_path)
    old_maps = {position.seed for position in old}
    corrective_maps = {position.seed for position in corrective}
    if (
        len(old_maps) != 64
        or corrective_maps != set(range(FIT_SEED, FIT_SEED + 64))
        or old_maps & corrective_maps
    ):
        raise ValueError("corrective and original fit maps are not disjoint")
    source, student = load_head_student(source_path, initial_head_path)
    traces = []
    for position in old:
        if position.opponent_decisions is None:
            raise ValueError("original fit data lack searched opponent decisions")
        traces.append((position.seed, position.opponent_decisions))
    traces.extend((position.seed, position.decisions) for position in corrective)
    started = time.perf_counter()
    history = train_action_head(traces, student, source, EPOCHS, FIT_SEED)
    elapsed = time.perf_counter() - started
    student.eval()
    initial = torch.load(initial_head_path, map_location="cpu", weights_only=True)
    checkpoint = {
        "kind": "corrective_opponent_action_head",
        "source_sha256": digest(source_path),
        "initial_head_sha256": digest(initial_head_path),
        "selected_expert": initial["selected_expert"],
        "action_head": student.action_head.state_dict(),
        "action_residual": (
            student.action_residual.state_dict()
            if student.action_residual is not None
            else None
        ),
    }
    torch.save(checkpoint, output_path)
    return {
        "kind": "procedural_duel_corrective_opponent_imitation_scout",
        "old_fit_sha256": digest(old_fit_path),
        "corrective_fit_sha256": digest(corrective_path),
        "source_sha256": digest(source_path),
        "initial_head_sha256": digest(initial_head_path),
        "checkpoint_sha256": digest(output_path),
        "checkpoint_bytes": output_path.stat().st_size,
        "old_fit_maps": len(old_maps),
        "corrective_fit_maps": len(corrective_maps),
        "corrective_candidate_states": metadata["candidate_states"],
        "corrective_actions": metadata["corrective_actions"],
        "corrective_censored_replies": metadata["censored_candidate_replies"],
        "training_elapsed_seconds": elapsed,
        "epochs": history,
        "qualification": "Corrective searched-action training only; no autonomous or game strength claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old_fit", type=Path)
    parser.add_argument("corrective_fit", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("initial_head", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            train(
                arguments.old_fit,
                arguments.corrective_fit,
                arguments.source,
                arguments.initial_head,
                arguments.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
