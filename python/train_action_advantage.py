from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from antiyoy_rl.model import UniversalPolicy, load_policy_state

try:
    from .build_bundle import digest
    from .collect_action_q import DATASET_KIND, DATASET_SCHEMA_VERSION
    from .evaluate import load_policy_checkpoint
except ImportError:
    from build_bundle import digest
    from collect_action_q import DATASET_KIND, DATASET_SCHEMA_VERSION
    from evaluate import load_policy_checkpoint


def load_action_q_dataset(path: Path) -> dict[str, object]:
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    if dataset.get("kind") != DATASET_KIND:
        raise ValueError("unsupported action-Q dataset kind")
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported action-Q dataset schema version")
    examples = dataset.get("examples")
    if not isinstance(examples, dict):
        raise ValueError("action-Q dataset has no examples")
    count = int(examples["regrets"].shape[0])
    required = (
        "search_features",
        "direct_features",
        "baseline_margins",
        "episode_seeds",
    )
    if any(int(examples[name].shape[0]) != count for name in required):
        raise ValueError("action-Q dataset example arrays have different lengths")
    if count == 0:
        raise ValueError("action-Q dataset is empty")
    feature_width = int(dataset["model"]["feature_width"])
    for name in ("search_features", "direct_features"):
        if examples[name].shape != (count, feature_width):
            raise ValueError("action-Q feature array has an invalid shape")
    return dataset


def feature_expert_state(
    checkpoint: dict[str, object], expert: str
) -> dict[str, Tensor]:
    if checkpoint.get("kind") == "routed_policy_bundle":
        return checkpoint["experts"][expert]
    return checkpoint["model"]


def pair_metrics(
    predicted_margins: Tensor, target_margins: Tensor, baseline_margins: Tensor
) -> dict[str, float | int | None]:
    predicted = predicted_margins.detach().cpu().to(torch.float64)
    targets = target_margins.detach().cpu().to(torch.float64)
    baseline = baseline_margins.detach().cpu().to(torch.float64)
    errors = predicted - targets
    predicted_improvements = predicted - baseline
    target_improvements = targets - baseline
    correlation = (
        float(
            np.corrcoef(predicted_improvements.numpy(), target_improvements.numpy())[
                0, 1
            ]
        )
        if predicted.numel() > 1
        and float(predicted_improvements.std()) > 0
        and float(target_improvements.std()) > 0
        else None
    )
    return {
        "examples": predicted.numel(),
        "mae": float(errors.abs().mean()),
        "rmse": float(errors.square().mean().sqrt()),
        "preference_accuracy": float(
            ((predicted > 0) == (targets > 0)).to(torch.float32).mean()
        ),
        "search_action_rate": float((predicted > 0).to(torch.float32).mean()),
        "improvement_correlation": correlation,
    }


def frozen_parameters(model: UniversalPolicy) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("action_head.")
    }


def concatenate_datasets(
    datasets: list[dict[str, object]],
) -> tuple[dict[str, Tensor], list[dict[str, object]]]:
    source_hashes = {str(dataset["source"]["sha256"]) for dataset in datasets}
    feature_widths = {int(dataset["model"]["feature_width"]) for dataset in datasets}
    if len(source_hashes) != 1:
        raise ValueError("action-Q datasets must share one source checkpoint")
    if len(feature_widths) != 1:
        raise ValueError("action-Q datasets must share one action feature width")
    tensor_names = (
        "search_features",
        "direct_features",
        "regrets",
        "baseline_margins",
        "episode_seeds",
        "seats",
        "rounds",
    )
    combined = {
        name: torch.cat([dataset["examples"][name] for dataset in datasets])
        for name in tensor_names
    }
    groups = []
    for dataset_index, dataset in enumerate(datasets):
        seeds = dataset["examples"]["episode_seeds"].to(torch.int64)
        groups.append(seeds + dataset_index * (1 << 48))
    combined["groups"] = torch.cat(groups)
    return combined, [dataset["source"] for dataset in datasets]


def split_by_episode(
    groups: Tensor, validation_fraction: float, seed: int
) -> tuple[Tensor, Tensor]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    unique = torch.unique(groups)
    if unique.numel() < 2:
        raise ValueError("advantage training requires at least two episodes")
    generator = torch.Generator().manual_seed(seed)
    shuffled = unique[torch.randperm(unique.numel(), generator=generator)]
    validation_count = max(1, round(unique.numel() * validation_fraction))
    validation_groups = shuffled[:validation_count]
    validation = torch.isin(groups, validation_groups)
    return ~validation, validation


def train_action_advantage(
    checkpoint_path: Path,
    dataset_paths: list[Path],
    output_path: Path,
    device_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    advantage_scale: float,
    retention_weight: float,
    validation_fraction: float,
    seed: int,
    training_seat: int | None = None,
) -> dict[str, object]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("advantage optimization values must be positive")
    if advantage_scale <= 0 or retention_weight < 0:
        raise ValueError("advantage and retention weights are invalid")
    datasets = [load_action_q_dataset(path) for path in dataset_paths]
    examples, sources = concatenate_datasets(datasets)
    if training_seat is not None:
        if training_seat < 0:
            raise ValueError("training seat must be non-negative")
        selected = examples["seats"].to(torch.long) == training_seat
        if not selected.any():
            raise ValueError("training seat has no action-Q examples")
        examples = {name: values[selected] for name, values in examples.items()}
    checkpoint_sha256 = digest(checkpoint_path)
    if any(str(source["sha256"]) != checkpoint_sha256 for source in sources):
        raise ValueError("action-Q dataset source does not match the checkpoint")
    checkpoint = load_policy_checkpoint(checkpoint_path, torch.device("cpu"))
    feature_expert = str(sources[0]["feature_expert"])
    seat_experts = tuple(str(expert) for expert in sources[0]["seat_experts"])
    if training_seat is not None and training_seat >= len(seat_experts):
        raise ValueError("training seat has no source expert")
    output_expert = (
        seat_experts[training_seat] if training_seat is not None else feature_expert
    )
    state = feature_expert_state(checkpoint, output_expert)
    hidden = int(datasets[0]["model"]["hidden"])
    layers = int(datasets[0]["model"]["layers"])
    model = UniversalPolicy(hidden, layers)
    load_policy_state(model, state)
    device = torch.device(device_name)
    model.to(device)
    preserved = frozen_parameters(model)
    initial_action_head = {
        name: value.detach().clone()
        for name, value in model.action_head.state_dict().items()
    }
    model.requires_grad_(False)
    for parameter in model.action_head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.action_head.parameters(), learning_rate, weight_decay=0.0
    )
    training_mask, validation_mask = split_by_episode(
        examples["groups"], validation_fraction, seed
    )
    search_features = examples["search_features"].to(torch.float32)
    direct_features = examples["direct_features"].to(torch.float32)
    baseline_margins = examples["baseline_margins"].to(torch.float32)
    target_margins = baseline_margins + advantage_scale * examples["regrets"]

    def predict(mask: Tensor) -> Tensor:
        predictions = []
        indices = torch.nonzero(mask).flatten()
        model.eval()
        with torch.no_grad():
            for start in range(0, indices.numel(), batch_size):
                selected = indices[start : start + batch_size]
                search_logits = model.action_head(
                    search_features[selected].to(device)
                ).squeeze(1)
                direct_logits = model.action_head(
                    direct_features[selected].to(device)
                ).squeeze(1)
                predictions.append((search_logits - direct_logits).cpu())
        return torch.cat(predictions)

    training_before = pair_metrics(
        predict(training_mask),
        target_margins[training_mask],
        baseline_margins[training_mask],
    )
    validation_before = pair_metrics(
        predict(validation_mask),
        target_margins[validation_mask],
        baseline_margins[validation_mask],
    )
    training_indices = torch.nonzero(training_mask).flatten()
    random = torch.Generator().manual_seed(seed)
    epoch_losses = []
    model.train()
    for _ in range(epochs):
        order = training_indices[
            torch.randperm(training_indices.numel(), generator=random)
        ]
        epoch_loss = 0.0
        batches = 0
        for start in range(0, order.numel(), batch_size):
            selected = order[start : start + batch_size]
            search_logits = model.action_head(
                search_features[selected].to(device)
            ).squeeze(1)
            direct_logits = model.action_head(
                direct_features[selected].to(device)
            ).squeeze(1)
            margins = search_logits - direct_logits
            target = target_margins[selected].to(device)
            fit = torch.nn.functional.smooth_l1_loss(margins, target)
            retention = torch.stack(
                [
                    (parameter - initial_action_head[name]).square().mean()
                    for name, parameter in model.action_head.named_parameters()
                ]
            ).mean()
            loss = fit + retention_weight * retention
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), 1.0)
            optimizer.step()
            batches += 1
            epoch_loss += (float(loss.item()) - epoch_loss) / batches
        epoch_losses.append(epoch_loss)
    model.eval()
    training_after = pair_metrics(
        predict(training_mask),
        target_margins[training_mask],
        baseline_margins[training_mask],
    )
    validation_after = pair_metrics(
        predict(validation_mask),
        target_margins[validation_mask],
        baseline_margins[validation_mask],
    )
    for name, expected in preserved.items():
        if not torch.equal(model.state_dict()[name].detach().cpu(), expected):
            raise RuntimeError(f"advantage training changed frozen parameter: {name}")
    changed_action_parameters = sum(
        not torch.equal(value.detach(), initial_action_head[name])
        for name, value in model.action_head.state_dict().items()
    )
    if changed_action_parameters == 0:
        raise RuntimeError("advantage training did not change the action head")
    output_config = copy.deepcopy(checkpoint["config"])
    for name in (
        "profiles",
        "routes",
        "context_routes",
        "seat_context_routes",
        "domain_routes",
        "selected_expert",
        "policy_kind",
    ):
        output_config.pop(name, None)
    output_config.update(
        {
            "hidden": hidden,
            "layers": layers,
            "profile": datasets[0]["config"]["profile"],
        }
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "offline_action_advantage_distillation",
        "source": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "feature_expert": feature_expert,
            "output_expert": output_expert,
        },
        "datasets": [
            {
                "path": str(path),
                "sha256": digest(path),
                "examples": int(dataset["examples"]["regrets"].shape[0]),
                "config": dataset["config"],
            }
            for path, dataset in zip(dataset_paths, datasets, strict=True)
        ],
        "examples": int(examples["regrets"].shape[0]),
        "training_examples": int(training_mask.sum()),
        "validation_examples": int(validation_mask.sum()),
        "validation_fraction": validation_fraction,
        "seed": seed,
        "training_seat": training_seat,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "advantage_scale": advantage_scale,
        "retention_weight": retention_weight,
        "epoch_losses": epoch_losses,
        "training_before": training_before,
        "training_after": training_after,
        "validation_before": validation_before,
        "validation_after": validation_after,
        "changed_action_parameters": changed_action_parameters,
        "frozen_parameters_preserved": True,
    }
    output = {
        "model": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "checkpoint_version": checkpoint["checkpoint_version"],
        "observation_version": checkpoint["observation_version"],
        "rule_features": checkpoint["rule_features"],
        "config": output_config,
        "summary": report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(output, temporary)
    temporary.replace(output_path)
    report["output"] = {
        "path": str(output_path),
        "sha256": digest(output_path),
        "size_bytes": output_path.stat().st_size,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--advantage-scale", type=float, default=1.0)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1_400_000)
    parser.add_argument("--training-seat", type=int)
    arguments = parser.parse_args()
    report = train_action_advantage(
        arguments.checkpoint,
        arguments.datasets,
        arguments.output,
        arguments.device,
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.advantage_scale,
        arguments.retention_weight,
        arguments.validation_fraction,
        arguments.seed,
        arguments.training_seat,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
