from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import Tensor

from antiyoy_rl.model import UniversalPolicy, load_policy_state

try:
    from .build_bundle import digest
    from .collect_action_slate import DATASET_KIND, DATASET_SCHEMA_VERSION
    from .evaluate import load_policy_checkpoint
    from .train_action_advantage import (
        feature_expert_state,
        frozen_parameters,
        split_by_episode,
    )
except ImportError:
    from build_bundle import digest
    from collect_action_slate import DATASET_KIND, DATASET_SCHEMA_VERSION
    from evaluate import load_policy_checkpoint
    from train_action_advantage import (
        feature_expert_state,
        frozen_parameters,
        split_by_episode,
    )


def load_action_slate_dataset(path: Path) -> dict[str, object]:
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    if dataset.get("kind") != DATASET_KIND:
        raise ValueError("unsupported action-slate dataset kind")
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported action-slate dataset schema version")
    actions = dataset.get("actions")
    states = dataset.get("states")
    replay = dataset.get("replay")
    if not isinstance(actions, dict) or not isinstance(states, dict):
        raise ValueError("action-slate dataset has no actions or states")
    if not isinstance(replay, dict):
        raise ValueError("action-slate dataset has no replay ledger")
    state_count = int(states["episode_seeds"].shape[0])
    offsets = actions["offsets"].to(torch.int64)
    if state_count == 0 or offsets.shape != (state_count + 1,):
        raise ValueError("action-slate state offsets have an invalid shape")
    if int(offsets[0]) != 0 or bool((offsets[1:] <= offsets[:-1]).any()):
        raise ValueError("action-slate state offsets must be strictly increasing")
    action_count = int(offsets[-1])
    feature_width = int(dataset["model"]["feature_width"])
    if actions["features"].shape != (action_count, feature_width):
        raise ValueError("action-slate features have an invalid shape")
    for name in (
        "baseline_logits",
        "root_probabilities",
        "root_values",
        "root_visits",
    ):
        if actions[name].shape != (action_count,):
            raise ValueError("action-slate action arrays have different lengths")
    if not bool(torch.isfinite(actions["features"]).all()):
        raise ValueError("action-slate features must be finite")
    for name in ("baseline_logits", "root_probabilities", "root_values"):
        if not bool(torch.isfinite(actions[name]).all()):
            raise ValueError("action-slate action values must be finite")
    if bool((actions["root_probabilities"] < 0).any()) or bool(
        (actions["root_visits"] < 0).any()
    ):
        raise ValueError("action-slate probabilities and visits must be non-negative")
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        probability_mass = actions["root_probabilities"][int(start) : int(end)].sum()
        if not torch.isclose(probability_mass, torch.tensor(1.0), atol=1e-5):
            raise ValueError("action-slate root probabilities must sum to one")
    for name in (
        "episode_steps",
        "seats",
        "rounds",
        "search_actions",
        "direct_actions",
    ):
        if states[name].shape != (state_count,):
            raise ValueError("action-slate state arrays have different lengths")
    if len(states["fingerprints"]) != state_count:
        raise ValueError("action-slate fingerprints have an invalid length")
    replay_count = int(replay["episode_seeds"].shape[0])
    if replay["action_offsets"].shape != (replay_count + 1,):
        raise ValueError("action-slate replay offsets have an invalid shape")
    if int(replay["action_offsets"][-1]) != int(replay["actions"].shape[0]):
        raise ValueError("action-slate replay actions have an invalid length")
    replay_seeds = replay["episode_seeds"].to(torch.int64)
    if torch.unique(replay_seeds).numel() != replay_count:
        raise ValueError("action-slate replay episode seeds must be unique")
    if not bool(
        torch.isin(states["episode_seeds"].to(torch.int64), replay_seeds).all()
    ):
        raise ValueError("action-slate replay is missing a labeled episode")
    return dataset


def concatenate_slate_datasets(
    datasets: list[dict[str, object]],
) -> tuple[dict[str, Tensor], list[dict[str, object]]]:
    source_hashes = {str(dataset["source"]["sha256"]) for dataset in datasets}
    feature_widths = {int(dataset["model"]["feature_width"]) for dataset in datasets}
    if len(source_hashes) != 1:
        raise ValueError("action-slate datasets must share one source checkpoint")
    if len(feature_widths) != 1:
        raise ValueError("action-slate datasets must share one feature width")
    action_names = (
        "features",
        "baseline_logits",
        "root_probabilities",
        "root_values",
        "root_visits",
    )
    combined = {
        name: torch.cat([dataset["actions"][name] for dataset in datasets])
        for name in action_names
    }
    widths = torch.cat(
        [
            dataset["actions"]["offsets"][1:] - dataset["actions"]["offsets"][:-1]
            for dataset in datasets
        ]
    ).to(torch.int64)
    combined["offsets"] = torch.cat(
        (torch.zeros(1, dtype=torch.int64), widths.cumsum(dim=0))
    )
    state_names = (
        "episode_seeds",
        "episode_steps",
        "seats",
        "rounds",
        "search_actions",
        "direct_actions",
    )
    combined.update(
        {
            name: torch.cat([dataset["states"][name] for dataset in datasets])
            for name in state_names
        }
    )
    groups = []
    for dataset_index, dataset in enumerate(datasets):
        seeds = dataset["states"]["episode_seeds"].to(torch.int64)
        groups.append(seeds + dataset_index * (1 << 48))
    combined["groups"] = torch.cat(groups)
    return combined, [dataset["source"] for dataset in datasets]


def select_slate_states(
    examples: dict[str, Tensor], selected: Tensor
) -> dict[str, Tensor]:
    state_indices = torch.nonzero(selected).flatten()
    if state_indices.numel() == 0:
        raise ValueError("action-slate selection contains no states")
    offsets = examples["offsets"]
    action_indices = torch.cat(
        [
            torch.arange(int(offsets[state]), int(offsets[state + 1]))
            for state in state_indices
        ]
    )
    widths = offsets[state_indices + 1] - offsets[state_indices]
    result = {
        name: examples[name][action_indices]
        for name in (
            "features",
            "baseline_logits",
            "root_probabilities",
            "root_values",
            "root_visits",
        )
    }
    result["offsets"] = torch.cat(
        (torch.zeros(1, dtype=torch.int64), widths.cumsum(dim=0))
    )
    for name in (
        "episode_seeds",
        "episode_steps",
        "seats",
        "rounds",
        "search_actions",
        "direct_actions",
        "groups",
    ):
        result[name] = examples[name][state_indices]
    return result


def conservative_target_logits(
    baseline_logits: Tensor,
    root_values: Tensor,
    root_visits: Tensor,
    offsets: Tensor,
    advantage_scale: float,
    visit_prior: float,
    advantage_clip: float,
) -> Tensor:
    targets = baseline_logits.to(torch.float32).clone()
    values = root_values.to(torch.float32)
    visits = root_visits.to(torch.float32)
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        start_index = int(start)
        end_index = int(end)
        state_visits = visits[start_index:end_index]
        measured = state_visits > 0
        if int(measured.sum()) < 2:
            continue
        measured_visits = state_visits[measured]
        measured_values = values[start_index:end_index][measured]
        center = (measured_values * measured_visits).sum() / measured_visits.sum()
        confidence = measured_visits / (measured_visits + visit_prior)
        advantages = (measured_values - center).clamp(-advantage_clip, advantage_clip)
        state_targets = targets[start_index:end_index]
        state_targets[measured] += advantage_scale * confidence * advantages
    return targets


def segment_log_softmax(values: Tensor, segments: Tensor, count: int) -> Tensor:
    maxima = torch.full((count,), -torch.inf, dtype=values.dtype, device=values.device)
    maxima.scatter_reduce_(0, segments, values, reduce="amax", include_self=True)
    shifted = values - maxima[segments]
    totals = torch.zeros(count, dtype=values.dtype, device=values.device)
    totals.scatter_add_(0, segments, shifted.exp())
    return shifted - totals[segments].log()


def action_indices_for_states(offsets: Tensor, states: Tensor) -> tuple[Tensor, Tensor]:
    widths = offsets[states + 1] - offsets[states]
    action_indices = torch.cat(
        [torch.arange(int(offsets[state]), int(offsets[state + 1])) for state in states]
    )
    segments = torch.repeat_interleave(torch.arange(states.numel()), widths)
    return action_indices, segments


def slate_metrics(
    model: UniversalPolicy,
    examples: dict[str, Tensor],
    selected_states: Tensor,
    target_logits: Tensor,
    device: torch.device,
) -> dict[str, float | int | None]:
    action_indices, segments = action_indices_for_states(
        examples["offsets"], selected_states
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, action_indices.numel(), 65_536):
            batch = action_indices[start : start + 65_536]
            predictions.append(
                model.action_head(
                    examples["features"][batch].to(device, dtype=torch.float32)
                )
                .squeeze(1)
                .cpu()
            )
    predicted_logits = torch.cat(predictions).to(torch.float64)
    baseline_logits = examples["baseline_logits"][action_indices].to(torch.float64)
    selected_targets = target_logits[action_indices].to(torch.float64)
    segment_count = int(selected_states.numel())
    predicted_log_probabilities = segment_log_softmax(
        predicted_logits, segments, segment_count
    )
    baseline_log_probabilities = segment_log_softmax(
        baseline_logits, segments, segment_count
    )
    target_log_probabilities = segment_log_softmax(
        selected_targets, segments, segment_count
    )
    target_probabilities = target_log_probabilities.exp()
    baseline_probabilities = baseline_log_probabilities.exp()
    target_kl_actions = target_probabilities * (
        target_log_probabilities - predicted_log_probabilities
    )
    source_kl_actions = baseline_probabilities * (
        baseline_log_probabilities - predicted_log_probabilities
    )
    target_kl = torch.zeros(segment_count, dtype=torch.float64)
    source_kl = torch.zeros(segment_count, dtype=torch.float64)
    target_kl.scatter_add_(0, segments, target_kl_actions)
    source_kl.scatter_add_(0, segments, source_kl_actions)
    target_top_matches = 0
    source_top_matches = 0
    correct_pairs = 0
    measured_pairs = 0
    local_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64),
            torch.bincount(segments, minlength=segment_count).cumsum(dim=0),
        )
    )
    selected_values = examples["root_values"][action_indices]
    selected_visits = examples["root_visits"][action_indices]
    for start, end in zip(local_offsets[:-1], local_offsets[1:], strict=True):
        state = slice(int(start), int(end))
        predicted = predicted_logits[state]
        target = selected_targets[state]
        baseline = baseline_logits[state]
        target_top_matches += int(predicted.argmax() == target.argmax())
        source_top_matches += int(predicted.argmax() == baseline.argmax())
        measured = selected_visits[state] > 0
        values = selected_values[state][measured]
        ranked = predicted[measured]
        for first in range(values.numel()):
            for second in range(first + 1, values.numel()):
                difference = float(values[first] - values[second])
                if abs(difference) <= 1e-6:
                    continue
                measured_pairs += 1
                correct_pairs += int(
                    (float(ranked[first] - ranked[second]) > 0) == (difference > 0)
                )
    return {
        "states": segment_count,
        "actions": int(action_indices.numel()),
        "target_kl": float(target_kl.mean()),
        "source_kl": float(source_kl.mean()),
        "target_top1_accuracy": target_top_matches / segment_count,
        "source_top1_preservation": source_top_matches / segment_count,
        "measured_pairs": measured_pairs,
        "measured_pair_accuracy": (
            correct_pairs / measured_pairs if measured_pairs else None
        ),
    }


def train_action_slate(
    checkpoint_path: Path,
    dataset_paths: list[Path],
    output_path: Path,
    device_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    advantage_scale: float,
    visit_prior: float,
    advantage_clip: float,
    retention_weight: float,
    validation_fraction: float,
    seed: int,
    training_seat: int | None = None,
) -> dict[str, object]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("action-slate optimization values must be positive")
    if advantage_scale <= 0 or visit_prior <= 0 or advantage_clip <= 0:
        raise ValueError("action-slate target values must be positive")
    if retention_weight < 0:
        raise ValueError("action-slate retention weight must be non-negative")
    datasets = [load_action_slate_dataset(path) for path in dataset_paths]
    examples, sources = concatenate_slate_datasets(datasets)
    if training_seat is not None:
        if training_seat < 0:
            raise ValueError("training seat must be non-negative")
        examples = select_slate_states(
            examples, examples["seats"].to(torch.long) == training_seat
        )
    checkpoint_sha256 = digest(checkpoint_path)
    if any(str(source["sha256"]) != checkpoint_sha256 for source in sources):
        raise ValueError("action-slate dataset source does not match the checkpoint")
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
    target_logits = conservative_target_logits(
        examples["baseline_logits"],
        examples["root_values"],
        examples["root_visits"],
        examples["offsets"],
        advantage_scale,
        visit_prior,
        advantage_clip,
    )
    training_mask, validation_mask = split_by_episode(
        examples["groups"], validation_fraction, seed
    )
    training_states = torch.nonzero(training_mask).flatten()
    validation_states = torch.nonzero(validation_mask).flatten()
    training_before = slate_metrics(
        model, examples, training_states, target_logits, device
    )
    validation_before = slate_metrics(
        model, examples, validation_states, target_logits, device
    )
    optimizer = torch.optim.AdamW(
        model.action_head.parameters(), learning_rate, weight_decay=0.0
    )
    random = torch.Generator().manual_seed(seed)
    epoch_losses = []
    model.train()
    for _ in range(epochs):
        order = training_states[
            torch.randperm(training_states.numel(), generator=random)
        ]
        epoch_loss = 0.0
        batches = 0
        for start in range(0, order.numel(), batch_size):
            selected_states = order[start : start + batch_size]
            action_indices, segments = action_indices_for_states(
                examples["offsets"], selected_states
            )
            segments = segments.to(device)
            selected_targets = target_logits[action_indices].to(device)
            selected_baseline = examples["baseline_logits"][action_indices].to(device)
            predicted = model.action_head(
                examples["features"][action_indices].to(device, dtype=torch.float32)
            ).squeeze(1)
            segment_count = int(selected_states.numel())
            predicted_log_probabilities = segment_log_softmax(
                predicted, segments, segment_count
            )
            target_log_probabilities = segment_log_softmax(
                selected_targets, segments, segment_count
            )
            baseline_log_probabilities = segment_log_softmax(
                selected_baseline, segments, segment_count
            )
            target_probabilities = target_log_probabilities.exp()
            baseline_probabilities = baseline_log_probabilities.exp()
            fit_actions = target_probabilities * (
                target_log_probabilities - predicted_log_probabilities
            )
            retention_actions = baseline_probabilities * (
                baseline_log_probabilities - predicted_log_probabilities
            )
            fit = torch.zeros(segment_count, device=device)
            retention = torch.zeros(segment_count, device=device)
            fit.scatter_add_(0, segments, fit_actions)
            retention.scatter_add_(0, segments, retention_actions)
            loss = fit.mean() + retention_weight * retention.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), 1.0)
            optimizer.step()
            batches += 1
            epoch_loss += (float(loss.item()) - epoch_loss) / batches
        epoch_losses.append(epoch_loss)
    model.eval()
    training_after = slate_metrics(
        model, examples, training_states, target_logits, device
    )
    validation_after = slate_metrics(
        model, examples, validation_states, target_logits, device
    )
    for name, expected in preserved.items():
        if not torch.equal(model.state_dict()[name].detach().cpu(), expected):
            raise RuntimeError(
                f"action-slate training changed frozen parameter: {name}"
            )
    changed_action_parameters = sum(
        not torch.equal(value.detach(), initial_action_head[name])
        for name, value in model.action_head.state_dict().items()
    )
    if changed_action_parameters == 0:
        raise RuntimeError("action-slate training did not change the action head")
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
        "kind": "conservative_action_slate_distillation",
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
                "states": int(dataset["states"]["episode_seeds"].shape[0]),
                "actions": int(dataset["actions"]["features"].shape[0]),
                "config": dataset["config"],
            }
            for path, dataset in zip(dataset_paths, datasets, strict=True)
        ],
        "states": int(examples["groups"].shape[0]),
        "actions": int(examples["features"].shape[0]),
        "training_states": int(training_states.numel()),
        "validation_states": int(validation_states.numel()),
        "validation_fraction": validation_fraction,
        "seed": seed,
        "training_seat": training_seat,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "advantage_scale": advantage_scale,
        "visit_prior": visit_prior,
        "advantage_clip": advantage_clip,
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
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--advantage-scale", type=float, default=0.5)
    parser.add_argument("--visit-prior", type=float, default=4.0)
    parser.add_argument("--advantage-clip", type=float, default=2.0)
    parser.add_argument("--retention-weight", type=float, default=4.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2_500_000)
    parser.add_argument("--training-seat", type=int)
    arguments = parser.parse_args()
    report = train_action_slate(
        arguments.checkpoint,
        arguments.datasets,
        arguments.output,
        arguments.device,
        arguments.epochs,
        arguments.batch_size,
        arguments.learning_rate,
        arguments.advantage_scale,
        arguments.visit_prior,
        arguments.advantage_clip,
        arguments.retention_weight,
        arguments.validation_fraction,
        arguments.seed,
        arguments.training_seat,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
