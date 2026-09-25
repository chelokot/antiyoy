import gzip
import json

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from python.audit_duel_first_regret import CHECKPOINT_SHA256
from python.audit_duel_structured_process_calibration import StructuredResponseModel
from python.fit_duel_structured_process_censor import (
    PROTOCOL,
    load_dataset,
    tensor_batch,
)


def dataset_records(corrupt_censor_tag: bool) -> list[dict]:
    records = [
        {
            "protocol": PROTOCOL,
            "source_sha256": CHECKPOINT_SHA256,
            "first_seed": 42,
            "maps": 1,
        }
    ]
    for seat in (0, 1):
        for opponent in ("source", "teacher"):
            censored = seat == 0 and opponent == "source"
            records.append(
                {
                    "type": "sample",
                    "seed": 42,
                    "root_seat": seat,
                    "opponent": opponent,
                    "own_action_index": 0,
                    "root_components": [0] * 16,
                    "teacher_selected_index": 0,
                    "rollin_censored": not censored
                    if corrupt_censor_tag and censored
                    else censored,
                    "candidates": [
                        {
                            "plan": [0],
                            "trace": [[0] * 16],
                            "static_score": 0,
                            "post_components": [0] * 16,
                            "reply_components": [1] * 16,
                            "followup_components": [2] * 16,
                            "terminal_classes": [1, 1, 1],
                        }
                    ],
                }
            )
            records.append(
                {
                    "type": "game",
                    "seed": 42,
                    "root_seat": seat,
                    "opponent": opponent,
                    "samples": 1,
                    "terminal": not censored,
                    "truncated": censored,
                    "winner": 255 if censored else seat,
                    "adjudicated_winner": 1 if censored else None,
                }
            )
    return records


def test_censored_rollin_keeps_branch_targets_without_game_outcome_label(
    tmp_path,
) -> None:
    path = tmp_path / "fit.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for record in dataset_records(False):
            output.write(json.dumps(record) + "\n")

    records, games = load_dataset(path, 42, 1)
    inputs, targets = tensor_batch(records[42])
    model = StructuredResponseModel()
    predicted_response, predicted_terminal = model(*inputs)
    loss = nn.functional.huber_loss(predicted_response, targets[0])
    loss += nn.functional.cross_entropy(predicted_terminal, targets[1])
    loss.backward()

    assert len(games) == 4
    assert sum(game["truncated"] for game in games.values()) == 1
    assert len(records[42]) == 4
    assert torch.isfinite(loss)
    assert model.readout.weight.grad is not None


def test_fit_loader_rejects_sample_censor_mismatch(tmp_path) -> None:
    path = tmp_path / "fit.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for record in dataset_records(True):
            output.write(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="sample censor tag disagrees"):
        load_dataset(path, 42, 1)
