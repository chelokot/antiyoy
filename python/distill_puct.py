from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from antiyoy_rl import VectorEnv
from antiyoy_rl.model import (
    UniversalPolicy,
    action_distribution,
    domain_key,
    encode_rules_batch,
    rotate_observation_180,
)
from antiyoy_rl.puct import PolicySearchConfig, policy_search_actions
from antiyoy_rl.routed import RoutedPolicy

try:
    from .build_bundle import digest
    from .evaluate import (
        instantiate_policy,
        load_policy_checkpoint,
        select_policy_state,
    )
except ImportError:
    from build_bundle import digest
    from evaluate import (
        instantiate_policy,
        load_policy_checkpoint,
        select_policy_state,
    )


@dataclass(frozen=True)
class PuctDistillationConfig:
    profile: str = "classic_generic_2022"
    environments: int = 64
    updates: int = 1_000
    seed: int = 800_000
    device: str = "cuda"
    width: int = 11
    height: int = 9
    action_limit: int = 1_000
    learning_rate: float = 1e-4
    retention_weight: float = 1.0
    rollin: str = "teacher"
    symmetry_augmentation: bool = True
    puct_nodes: int = 8
    puct_exploration: float = 1.5
    puct_virtual_loss: float = 1.0
    puct_maximum_depth: int = 128
    puct_root_value_weight: float | None = 1.0
    puct_leaf_batch_size: int = 512


def validate_config(config: PuctDistillationConfig) -> None:
    if config.environments < 1 or config.updates < 1:
        raise ValueError("distillation environments and updates must be positive")
    if config.width < 3 or config.height < 3 or config.action_limit < 1:
        raise ValueError("distillation arena dimensions and action limit are invalid")
    if config.learning_rate <= 0 or config.retention_weight < 0:
        raise ValueError("distillation optimization values are invalid")
    if config.rollin not in ("teacher", "student"):
        raise ValueError("distillation roll-in must be teacher or student")
    if config.puct_nodes < 2 or config.puct_leaf_batch_size < 1:
        raise ValueError("distillation PUCT budgets are invalid")
    if config.puct_exploration <= 0 or config.puct_virtual_loss < 0:
        raise ValueError("distillation PUCT search values are invalid")
    if config.puct_maximum_depth < 1:
        raise ValueError("distillation PUCT maximum depth must be positive")
    if config.puct_root_value_weight is not None and config.puct_root_value_weight < 0:
        raise ValueError("distillation PUCT root value weight must be non-negative")


def frozen_policy_parameters(model: UniversalPolicy) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("action_head.")
    }


def configure_action_head_training(model: UniversalPolicy) -> list[torch.nn.Parameter]:
    model.requires_grad_(False)
    parameters = list(model.action_head.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def verify_frozen_policy_parameters(
    model: UniversalPolicy, preserved: dict[str, torch.Tensor]
) -> None:
    for key, expected in preserved.items():
        actual = model.state_dict()[key].detach().cpu()
        if not torch.equal(actual, expected):
            raise RuntimeError(f"PUCT distillation changed frozen parameter: {key}")


def distill_puct(
    checkpoint_path: Path,
    output_path: Path,
    config: PuctDistillationConfig,
) -> dict[str, object]:
    validate_config(config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    checkpoint = load_policy_checkpoint(checkpoint_path, device)
    checkpoint_config = dict(checkpoint["config"])
    descriptor = {
        "width": config.width,
        "height": config.height,
        "players": 2,
        "action_limit": config.action_limit,
        "fog": checkpoint_config["fog"],
        "diplomacy": checkpoint_config.get("diplomacy", False),
        "initial_relation": checkpoint_config.get("initial_relation", "neutral"),
    }
    evaluation_domain = domain_key("symmetric_duel_v1", descriptor)
    selected_states: list[dict[str, torch.Tensor]] = []
    selected_configs: list[dict[str, object]] = []
    selected_experts: list[str] = []
    for seat in range(2):
        state, selected_config = select_policy_state(
            checkpoint,
            config.profile,
            "symmetric_duel_v1",
            2,
            seat,
            evaluation_domain,
        )
        selected_states.append(state)
        selected_configs.append(selected_config)
        selected_experts.append(str(selected_config["selected_expert"]))
    if selected_experts[0] != selected_experts[1]:
        raise ValueError("PUCT distillation requires one shared expert for both seats")
    teacher = instantiate_policy(selected_states[0], selected_configs[0], device)
    teacher.requires_grad_(False)
    student = copy.deepcopy(teacher)
    preserved = frozen_policy_parameters(student)
    initial_action_head = {
        key: value.detach().cpu().clone()
        for key, value in student.state_dict().items()
        if key.startswith("action_head.")
    }
    optimizer = torch.optim.AdamW(
        configure_action_head_training(student),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    environment = VectorEnv(
        config.environments,
        width=config.width,
        height=config.height,
        seed=config.seed,
        action_limit=config.action_limit,
        profile=config.profile,
        fog=bool(checkpoint_config["fog"]),
        diplomacy=bool(checkpoint_config.get("diplomacy", False)),
        initial_relation=str(checkpoint_config.get("initial_relation", "neutral")),
    )
    for environment_index in range(config.environments):
        environment.reset(environment_index, config.seed + environment_index)
    rules = encode_rules_batch(environment.rules_jsons(), device)
    teacher_policy = RoutedPolicy({"teacher": teacher}, ("teacher", "teacher"))
    search_config = PolicySearchConfig(
        node_budget=config.puct_nodes,
        exploration=config.puct_exploration,
        virtual_loss=config.puct_virtual_loss,
        maximum_depth=config.puct_maximum_depth,
        root_value_weight=config.puct_root_value_weight,
        leaf_batch_size=config.puct_leaf_batch_size,
    )
    active_mask = np.ones(config.environments, dtype=np.uint8)
    next_reset_seed = config.seed + config.environments
    completed_games = 0
    truncations = 0
    draws = 0
    wins_by_seat = np.zeros(2, dtype=np.int64)
    imitation_loss_average = 0.0
    retention_average = 0.0
    accuracy_average = 0.0
    disagreement_average = 0.0
    evaluated_leaves = 0
    leaf_batches = 0
    total_nodes = 0
    total_root_visits = 0
    maximum_reached_depth = 0
    started = time.perf_counter()
    student.train()
    for update in range(1, config.updates + 1):
        observation = environment.observe()
        teacher_actions, search_metrics = policy_search_actions(
            environment,
            teacher_policy,
            rules,
            active_mask,
            search_config,
        )
        with torch.no_grad():
            direct_logits, _ = teacher(observation, rules)
            direct_distribution = action_distribution(
                direct_logits, observation["action_offsets"]
            )
            direct_actions = direct_distribution.logits.argmax(dim=1)
        model_observation = observation
        if config.symmetry_augmentation:
            rotation_mask = (
                np.arange(config.environments, dtype=np.uint64) + update
            ) % 2 == 0
            model_observation = rotate_observation_180(observation, rotation_mask)
        with torch.no_grad():
            reference_logits, _ = teacher(model_observation, rules)
            reference_distribution = action_distribution(
                reference_logits, model_observation["action_offsets"]
            )
        student_logits, _ = student(model_observation, rules)
        student_distribution = action_distribution(
            student_logits, model_observation["action_offsets"]
        )
        targets = torch.as_tensor(teacher_actions, dtype=torch.long, device=device)
        imitation_loss = -student_distribution.log_prob(targets).mean()
        retention = torch.distributions.kl_divergence(
            reference_distribution, student_distribution
        ).mean()
        loss = imitation_loss + config.retention_weight * retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.action_head.parameters(), 1.0)
        optimizer.step()
        accuracy = float(
            (student_distribution.logits.argmax(dim=1) == targets).float().mean().item()
        )
        disagreement = float((direct_actions.cpu().numpy() != teacher_actions).mean())
        imitation_loss_average += (
            float(imitation_loss.item()) - imitation_loss_average
        ) / update
        retention_average += (float(retention.item()) - retention_average) / update
        accuracy_average += (accuracy - accuracy_average) / update
        disagreement_average += (disagreement - disagreement_average) / update
        evaluated_leaves += int(search_metrics["evaluated_leaves"])
        leaf_batches += int(search_metrics["leaf_batches"])
        total_nodes += int(search_metrics["nodes"].sum())
        total_root_visits += int(search_metrics["root_visits"].sum())
        maximum_reached_depth = max(
            maximum_reached_depth,
            int(search_metrics["maximum_depth"].max(initial=0)),
        )
        if config.rollin == "teacher":
            rollin_actions = teacher_actions
        else:
            student.eval()
            with torch.no_grad():
                rollin_logits, _ = student(observation, rules)
                rollin_distribution = action_distribution(
                    rollin_logits, observation["action_offsets"]
                )
                rollin_actions = (
                    rollin_distribution.logits.argmax(dim=1)
                    .cpu()
                    .numpy()
                    .astype(np.uint64)
                )
            student.train()
        result = environment.step(rollin_actions)
        done = np.logical_or(result["terminal"], result["truncated"])
        for environment_index in np.flatnonzero(done):
            winner = int(result["winners"][environment_index])
            if bool(result["truncated"][environment_index]):
                truncations += 1
                winner = int(result["adjudicated_winners"][environment_index])
            if winner == 255:
                draws += 1
            else:
                wins_by_seat[winner] += 1
            completed_games += 1
            environment.reset(int(environment_index), next_reset_seed)
            next_reset_seed += 1
        if update == 1 or update % 100 == 0 or update == config.updates:
            print(
                json.dumps(
                    {
                        "stage": "puct_distillation",
                        "update": update,
                        "imitation_loss": float(imitation_loss.item()),
                        "retention_kl": float(retention.item()),
                        "teacher_accuracy": accuracy,
                        "teacher_direct_disagreement": disagreement,
                        "completed_games": completed_games,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    student.eval()
    verify_frozen_policy_parameters(student, preserved)
    changed_action_parameters = sum(
        not torch.equal(student.state_dict()[key].detach().cpu(), value)
        for key, value in initial_action_head.items()
    )
    if changed_action_parameters == 0:
        raise RuntimeError("PUCT distillation did not change the action head")
    output_config = dict(selected_configs[0])
    output_config.pop("selected_expert", None)
    output_config.pop("policy_kind", None)
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "policy_guided_puct_distillation",
        "source": {
            "path": str(checkpoint_path),
            "sha256": digest(checkpoint_path),
            "expert": selected_experts[0],
        },
        "profile": config.profile,
        "generator": "symmetric_duel_v1",
        "domain": evaluation_domain,
        "domain_descriptor": descriptor,
        "seed": config.seed,
        "environments": config.environments,
        "updates": config.updates,
        "examples": config.environments * config.updates,
        "learning_rate": config.learning_rate,
        "retention_weight": config.retention_weight,
        "rollin": config.rollin,
        "symmetry_augmentation": config.symmetry_augmentation,
        "policy_parameters": "action_head_only",
        "frozen_parameters_preserved": True,
        "changed_action_parameters": changed_action_parameters,
        "training": {
            "seconds": time.perf_counter() - started,
            "imitation_loss": imitation_loss_average,
            "retention_kl": retention_average,
            "teacher_accuracy": accuracy_average,
            "teacher_direct_disagreement": disagreement_average,
            "completed_games": completed_games,
            "wins_by_seat": wins_by_seat.tolist(),
            "draws": draws,
            "truncations": truncations,
        },
        "policy_search": {
            "node_budget": config.puct_nodes,
            "exploration": config.puct_exploration,
            "virtual_loss": config.puct_virtual_loss,
            "maximum_depth": config.puct_maximum_depth,
            "root_value_weight": config.puct_root_value_weight,
            "leaf_batch_size": config.puct_leaf_batch_size,
            "decisions": config.environments * config.updates,
            "evaluated_leaves": evaluated_leaves,
            "leaf_batches": leaf_batches,
            "total_nodes": total_nodes,
            "total_root_visits": total_root_visits,
            "maximum_reached_depth": maximum_reached_depth,
        },
    }
    output = {
        "model": {
            key: value.detach().cpu() for key, value in student.state_dict().items()
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
    parser.add_argument("--profile", default="classic_generic_2022")
    parser.add_argument("--environments", type=int, default=64)
    parser.add_argument("--updates", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=800_000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--action-limit", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--rollin", choices=("teacher", "student"), default="teacher")
    parser.add_argument(
        "--symmetry-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--puct-nodes", type=int, default=8)
    parser.add_argument("--puct-exploration", type=float, default=1.5)
    parser.add_argument("--puct-virtual-loss", type=float, default=1.0)
    parser.add_argument("--puct-maximum-depth", type=int, default=128)
    parser.add_argument("--puct-root-value-weight", type=float, default=1.0)
    parser.add_argument("--puct-leaf-batch-size", type=int, default=512)
    arguments = parser.parse_args()
    report = distill_puct(
        arguments.checkpoint,
        arguments.output,
        PuctDistillationConfig(
            profile=arguments.profile,
            environments=arguments.environments,
            updates=arguments.updates,
            seed=arguments.seed,
            device=arguments.device,
            width=arguments.width,
            height=arguments.height,
            action_limit=arguments.action_limit,
            learning_rate=arguments.learning_rate,
            retention_weight=arguments.retention_weight,
            rollin=arguments.rollin,
            symmetry_augmentation=arguments.symmetry_augmentation,
            puct_nodes=arguments.puct_nodes,
            puct_exploration=arguments.puct_exploration,
            puct_virtual_loss=arguments.puct_virtual_loss,
            puct_maximum_depth=arguments.puct_maximum_depth,
            puct_root_value_weight=arguments.puct_root_value_weight,
            puct_leaf_batch_size=arguments.puct_leaf_batch_size,
        ),
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
