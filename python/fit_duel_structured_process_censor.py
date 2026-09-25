from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .audit_duel_first_regret import CHECKPOINT_SHA256
from .audit_duel_structured_process_calibration import (
    COMPONENT_SCALES,
    MAXIMUM_ACTIONS_PER_TURN,
    StructuredResponseModel,
)
from .build_bundle import digest


PROTOCOL = (
    "benchmarks/protocols/2026-09-25-duel-structured-response-censor-aware-v2.json"
)
FIRST_SEED = 6610000
MAPS = 128
INITIALIZATION_SEED = 6600000


def load_dataset(
    path: Path, first_seed: int, maps: int
) -> tuple[dict[int, list[dict]], dict]:
    records: dict[int, list[dict]] = defaultdict(list)
    games = {}
    with gzip.open(path, "rt", encoding="utf-8") as source:
        header = json.loads(next(source))
        if (
            header["protocol"] != PROTOCOL
            or header["source_sha256"] != CHECKPOINT_SHA256
            or header["first_seed"] != first_seed
            or header["maps"] != maps
        ):
            raise ValueError("process dataset header disagrees with the protocol")
        for line in source:
            entry = json.loads(line)
            key = (entry["seed"], entry["root_seat"], entry["opponent"])
            if not first_seed <= entry["seed"] < first_seed + maps:
                raise ValueError("process dataset has an out-of-window map")
            if entry["root_seat"] not in (0, 1) or entry["opponent"] not in (
                "source",
                "teacher",
            ):
                raise ValueError("process dataset has an unexpected arm")
            if entry["type"] == "sample":
                if not entry["candidates"] or len(entry["candidates"]) > 8:
                    raise ValueError("process dataset has an invalid candidate slate")
                if not 0 <= entry["teacher_selected_index"] < len(entry["candidates"]):
                    raise ValueError("process dataset has an invalid teacher choice")
                for candidate in entry["candidates"]:
                    if not 1 <= len(candidate["trace"]) <= MAXIMUM_ACTIONS_PER_TURN:
                        raise ValueError("process dataset has an invalid trace")
                    if len(candidate["trace"]) != len(candidate["plan"]):
                        raise ValueError("process trace and plan lengths disagree")
                    if any(len(vector) != 16 for vector in candidate["trace"]):
                        raise ValueError("process dataset has an invalid trace vector")
                    if any(
                        len(candidate[field]) != 16
                        for field in (
                            "post_components",
                            "reply_components",
                            "followup_components",
                        )
                    ):
                        raise ValueError("process dataset has an invalid score vector")
                    if len(candidate["terminal_classes"]) != 3 or any(
                        value not in (0, 1, 2)
                        for value in candidate["terminal_classes"]
                    ):
                        raise ValueError(
                            "process dataset has an invalid terminal class"
                        )
                records[entry["seed"]].append(entry)
            elif entry["type"] == "game":
                if key in games or entry["terminal"] == entry["truncated"]:
                    raise ValueError(
                        "process dataset has duplicate or invalid game status"
                    )
                if entry["truncated"]:
                    if entry["adjudicated_winner"] not in (0, 1, 255):
                        raise ValueError("process dataset lacks censored adjudication")
                elif entry["adjudicated_winner"] is not None:
                    raise ValueError("terminal game has censored adjudication")
                games[key] = entry
            else:
                raise ValueError("process dataset has an unknown record type")
    expected = {
        (seed, seat, opponent)
        for seed in range(first_seed, first_seed + maps)
        for seat in (0, 1)
        for opponent in ("source", "teacher")
    }
    if games.keys() != expected:
        raise ValueError("process dataset is missing a predeclared game")
    samples_by_game = defaultdict(int)
    for samples in records.values():
        for sample in samples:
            key = (sample["seed"], sample["root_seat"], sample["opponent"])
            if sample["rollin_censored"] != games[key]["truncated"]:
                raise ValueError("process sample censor tag disagrees with its game")
            samples_by_game[key] += 1
    if any(games[key]["samples"] != samples_by_game[key] for key in expected):
        raise ValueError("process sample counts disagree with game records")
    if any(samples_by_game[key] == 0 for key in expected):
        raise ValueError("process dataset has an unsampled game")
    return dict(records), games


def tensor_batch(samples: list[dict]) -> tuple[tuple[torch.Tensor, ...], ...]:
    candidates = [
        (sample, candidate) for sample in samples for candidate in sample["candidates"]
    ]
    traces = np.zeros((len(candidates), MAXIMUM_ACTIONS_PER_TURN, 16), dtype=np.float32)
    lengths = np.empty(len(candidates), dtype=np.int64)
    roots = np.empty((len(candidates), 16), dtype=np.float32)
    posts = np.empty((len(candidates), 16), dtype=np.float32)
    static = np.empty((len(candidates), 1), dtype=np.float32)
    responses = np.empty((len(candidates), 2, 16), dtype=np.float32)
    terminals = np.empty(len(candidates), dtype=np.int64)
    for index, (sample, candidate) in enumerate(candidates):
        trace = np.asarray(candidate["trace"], dtype=np.float32)
        post = np.asarray(candidate["post_components"], dtype=np.float32)
        traces[index, : len(trace)] = trace
        lengths[index] = len(trace)
        roots[index] = np.asarray(sample["root_components"]) / COMPONENT_SCALES
        posts[index] = post / COMPONENT_SCALES
        static[index, 0] = candidate["static_score"] / 1000
        responses[index, 0] = (
            np.asarray(candidate["reply_components"]) - post
        ) / COMPONENT_SCALES
        responses[index, 1] = (
            np.asarray(candidate["followup_components"]) - post
        ) / COMPONENT_SCALES
        terminals[index] = candidate["terminal_classes"][2]
    if not all(
        np.isfinite(array).all() for array in (traces, roots, posts, static, responses)
    ):
        raise ValueError("process dataset contains a nonfinite model input")
    inputs = (
        torch.from_numpy(traces),
        torch.from_numpy(lengths),
        torch.from_numpy(roots),
        torch.from_numpy(posts),
        torch.from_numpy(static),
    )
    targets = (torch.from_numpy(responses), torch.from_numpy(terminals))
    return inputs, targets


def fit(source_path: Path, dataset_path: Path, checkpoint_path: Path) -> dict:
    if digest(source_path) != CHECKPOINT_SHA256:
        raise ValueError("frozen source checkpoint disagrees with the protocol")
    torch.set_num_threads(1)
    records, games = load_dataset(dataset_path, FIRST_SEED, MAPS)
    torch.manual_seed(INITIALIZATION_SEED)
    model = StructuredResponseModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    per_map = [
        tensor_batch(records[seed]) for seed in range(FIRST_SEED, FIRST_SEED + MAPS)
    ]
    epoch_losses = []
    for _ in range(2):
        losses = []
        for map_index in torch.randperm(MAPS):
            inputs, targets = per_map[int(map_index)]
            predicted_response, predicted_terminal = model(*inputs)
            response_loss = nn.functional.huber_loss(
                predicted_response, targets[0], reduction="mean"
            )
            terminal_loss = nn.functional.cross_entropy(predicted_terminal, targets[1])
            loss = response_loss + terminal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_losses.append(sum(losses) / MAPS)
    temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(
        {
            "protocol": PROTOCOL,
            "source_sha256": CHECKPOINT_SHA256,
            "fit_seed": FIRST_SEED,
            "fit_maps": MAPS,
            "model": model.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    with dataset_path.open("rb") as source:
        dataset_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    with checkpoint_path.open("rb") as checkpoint:
        checkpoint_sha256 = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    return {
        "kind": "censor_aware_structured_response_process_fit",
        "protocol": PROTOCOL,
        "source_sha256": CHECKPOINT_SHA256,
        "fit_raw_sha256": dataset_sha256,
        "fit_maps": MAPS,
        "fit_games": len(games),
        "fit_terminal_games": sum(game["terminal"] for game in games.values()),
        "fit_censored_games": sum(game["truncated"] for game in games.values()),
        "fit_samples": sum(len(samples) for samples in records.values()),
        "fit_candidate_branches": sum(
            len(sample["candidates"])
            for samples in records.values()
            for sample in samples
        ),
        "epoch_map_balanced_losses": epoch_losses,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "qualification": "Fit only; no independent offline or autonomous game result",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = json.dumps(
        fit(arguments.source, arguments.dataset, arguments.checkpoint), sort_keys=True
    )
    if arguments.output:
        arguments.output.write_text(f"{report}\n")
        print(json.dumps(json.loads(report)["epoch_map_balanced_losses"]))
    else:
        print(report)


if __name__ == "__main__":
    main()
