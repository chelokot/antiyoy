from __future__ import annotations

import argparse
import copy
import json
import resource
import time
from pathlib import Path
from typing import cast

import torch

from .audit_duel_first_regret import CHECKPOINT_SHA256, load_routed_policy
from .build_bundle import digest
from .collect_duel_episode_preference_window import PROTOCOL
from . import train_duel_episode_preference as preference


def verify_small_replay_gradients(
    student: preference.RoutedPolicy,
    reference: preference.RoutedPolicy,
    preferred: preference.DecisionTrace,
    dispreferred: preference.DecisionTrace,
) -> None:
    preferred = preference.DecisionTrace(
        preferred.observations[:2], preferred.rules_json, preferred.actions[:2]
    )
    dispreferred = preference.DecisionTrace(
        dispreferred.observations[:2], dispreferred.rules_json, dispreferred.actions[:2]
    )
    streamed = copy.deepcopy(student)
    direct = copy.deepcopy(student)
    original_chunk_size = preference.CHUNK_SIZE
    preference.CHUNK_SIZE = 1
    try:
        streamed_loss, streamed_margin = preference.accumulate_pair_gradient(
            streamed, reference, preferred, dispreferred, 1
        )
    finally:
        preference.CHUNK_SIZE = original_chunk_size
    direct_margin = (
        preference.decision_log_probabilities(direct, preferred, 0, 2).sum()
        - preference.decision_log_probabilities(direct, dispreferred, 0, 2).sum()
    )
    with torch.no_grad():
        reference_margin = (
            preference.decision_log_probabilities(reference, preferred, 0, 2).sum()
            - preference.decision_log_probabilities(reference, dispreferred, 0, 2).sum()
        )
    direct_margin = direct_margin - reference_margin
    direct_loss = preference.preference_loss(direct_margin)
    direct_loss.backward()
    torch.testing.assert_close(
        torch.tensor(streamed_loss), direct_loss.detach(), atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        torch.tensor(streamed_margin), direct_margin.detach(), atol=1e-5, rtol=1e-4
    )
    for name in streamed.models:
        for streamed_parameter, direct_parameter in zip(
            streamed.models[name].parameters(),
            direct.models[name].parameters(),
            strict=True,
        ):
            if direct_parameter.grad is None:
                if streamed_parameter.grad is not None:
                    raise ValueError("streamed gradient unexpectedly exists")
            else:
                if streamed_parameter.grad is None:
                    raise ValueError("streamed gradient is missing")
                torch.testing.assert_close(
                    streamed_parameter.grad,
                    direct_parameter.grad,
                    atol=1e-5,
                    rtol=1e-4,
                )


def calibrate(
    source_path: Path, fit_path: Path, calibration_path: Path
) -> dict[str, object]:
    if digest(source_path) != CHECKPOINT_SHA256:
        raise ValueError("frozen source checkpoint disagrees with the protocol")
    fit = preference.checked_dataset(fit_path, "fit")
    calibration = preference.checked_dataset(
        calibration_path, "resource_calibration"
    )
    fit_summary = cast(dict[str, object], fit["summary"])
    calibration_summary = cast(dict[str, object], calibration["summary"])
    if not fit_summary["data_gate_passed"] or not calibration_summary[
        "integrity_gate_passed"
    ]:
        raise ValueError("episode data gate failed before resource calibration")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    student, student_experts = load_routed_policy(source_path)
    reference, reference_experts = load_routed_policy(source_path)
    if student_experts != reference_experts:
        raise ValueError("student and reference routes differ")
    first_seed = preference.WINDOWS["resource_calibration"][0]
    records = cast(list[dict[str, object]], calibration["records"])
    indexed = {
        (record["seed"], record["teacher_seat"]): record for record in records
    }
    source = indexed[first_seed, None]
    teacher = indexed[first_seed, 0]
    preferred = preference.replay_decisions(teacher, 0)
    dispreferred = preference.replay_decisions(source, 0)
    started = time.perf_counter()
    verify_small_replay_gradients(student, reference, preferred, dispreferred)
    gradient_seconds = time.perf_counter() - started
    parameters = [
        parameter
        for model in student.models.values()
        for name, parameter in model.named_parameters()
        if not name.startswith("value_head.")
    ]
    for model in student.models.values():
        model.value_head.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=preference.LEARNING_RATE,
        weight_decay=preference.WEIGHT_DECAY,
    )
    started = time.perf_counter()
    loss, margin = preference.fit_map(
        student,
        reference,
        [preference.PreferencePair(0, teacher, source)],
        optimizer,
        parameters,
    )
    update_seconds = time.perf_counter() - started
    fit_pairs = cast(int, fit_summary["informative_pairs"])
    projected_fit_seconds = update_seconds * fit_pairs * preference.EPOCHS
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {
        "kind": "terminal_episode_preference_resource_calibration",
        "protocol": PROTOCOL,
        "source_sha256": CHECKPOINT_SHA256,
        "fit_dataset_sha256": digest(fit_path),
        "calibration_dataset_sha256": digest(calibration_path),
        "gradient_equivalence_passed": True,
        "small_replay_gradient_seconds": gradient_seconds,
        "one_pair_full_map_update_seconds": update_seconds,
        "one_pair_full_map_loss": loss,
        "one_pair_full_map_margin": margin,
        "projected_fit_seconds": projected_fit_seconds,
        "peak_resident_bytes": peak_rss_bytes,
        "resource_gate_passed": (
            projected_fit_seconds <= 3600 and peak_rss_bytes <= 4 * 1024**3
        ),
        "qualification": "Resource and numerical calibration only; the temporary model update was not saved or trained on fit maps",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("fit_dataset", type=Path)
    parser.add_argument("calibration_dataset", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            calibrate(
                arguments.source, arguments.fit_dataset, arguments.calibration_dataset
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
