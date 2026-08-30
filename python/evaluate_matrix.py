from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from .evaluate import evaluate
    from .evaluate_suite import (
        ALL_PROFILES,
        aggregate_results,
        checkpoint_digest,
        minimum_seat_slice,
        write_result,
    )
except ImportError:
    from evaluate import evaluate
    from evaluate_suite import (
        ALL_PROFILES,
        aggregate_results,
        checkpoint_digest,
        minimum_seat_slice,
        write_result,
    )


MATRIX_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Domain:
    identifier: str
    generator: str
    width: int
    height: int
    players: int
    games_per_seed: int
    seeds: tuple[int, ...]
    action_limit: int
    land_density_per_million: int = 650_000
    starting_province_size: int = 5
    starting_money: int = 10
    tree_density_per_million: int = 150_000
    neutral_tower_density_per_million: int = 20_000
    neutral_capital_density_per_million: int = 10_000
    grave_density_per_million: int = 15_000

    @property
    def procedural(self) -> bool:
        return self.generator == "procedural_v1"


@dataclass(frozen=True)
class Search:
    nodes: int
    beam_width: int
    branch_width: int
    maximum_actions_per_turn: int


@dataclass(frozen=True)
class Gates:
    minimum_domain_relative_elo: float
    minimum_slice_score_delta: float
    maximum_truncation_rate_delta: float


@dataclass(frozen=True)
class Matrix:
    profiles: tuple[str, ...]
    domains: tuple[Domain, ...]
    baseline: str
    search: Search
    gates: Gates


def parse_matrix(path: Path) -> Matrix:
    payload = json.loads(path.read_text())
    if payload["schema_version"] != MATRIX_SCHEMA_VERSION:
        raise ValueError("evaluation matrix schema version does not match")
    profiles = tuple(payload["profiles"])
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("profiles must be non-empty and unique")
    unknown_profiles = set(profiles).difference(ALL_PROFILES)
    if unknown_profiles:
        raise ValueError(f"unknown profiles: {sorted(unknown_profiles)}")
    domains = tuple(_parse_domain(domain) for domain in payload["domains"])
    if not domains or len({domain.identifier for domain in domains}) != len(domains):
        raise ValueError("domains must be non-empty and uniquely named")
    baseline = payload["baseline"]
    if baseline not in ("search", "greedy", "random"):
        raise ValueError(f"unsupported baseline: {baseline}")
    search = Search(**payload["search"])
    if min(
        search.nodes,
        search.beam_width,
        search.branch_width,
        search.maximum_actions_per_turn,
    ) < 1:
        raise ValueError("search limits must be positive")
    gates = Gates(**payload["gates"])
    if gates.minimum_slice_score_delta < -1 or gates.minimum_slice_score_delta > 1:
        raise ValueError("minimum slice score delta must be between minus one and one")
    if (
        gates.maximum_truncation_rate_delta < -1
        or gates.maximum_truncation_rate_delta > 1
    ):
        raise ValueError(
            "maximum truncation rate delta must be between minus one and one"
        )
    return Matrix(profiles, domains, baseline, search, gates)


def _parse_domain(payload: dict[str, object]) -> Domain:
    values = dict(payload)
    values["identifier"] = values.pop("id")
    values["seeds"] = tuple(values["seeds"])
    domain = Domain(**values)
    if domain.generator not in ("symmetric_duel_v1", "procedural_v1"):
        raise ValueError(f"unsupported generator: {domain.generator}")
    if domain.players < 2 or domain.players > 8:
        raise ValueError(f"domain {domain.identifier} players must be between 2 and 8")
    if not domain.procedural and domain.players != 2:
        raise ValueError(f"domain {domain.identifier} symmetric duel requires two players")
    if domain.games_per_seed < domain.players or domain.games_per_seed % domain.players:
        raise ValueError(
            f"domain {domain.identifier} games_per_seed must be a positive multiple of players"
        )
    if not domain.seeds:
        raise ValueError(f"domain {domain.identifier} must contain seeds")
    if min(domain.width, domain.height, domain.action_limit) < 1:
        raise ValueError(f"domain {domain.identifier} dimensions and action limit must be positive")
    densities = (
        domain.land_density_per_million,
        domain.tree_density_per_million,
        domain.neutral_tower_density_per_million,
        domain.neutral_capital_density_per_million,
        domain.grave_density_per_million,
    )
    if any(density < 0 or density > 1_000_000 for density in densities):
        raise ValueError(f"domain {domain.identifier} densities must be between zero and one million")
    return domain


def matrix_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_domain(
    checkpoint: Path,
    domain: Domain,
    profiles: tuple[str, ...],
    baseline: str,
    search: Search,
    device: str,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for profile in profiles:
        for seed in domain.seeds:
            print(
                f"evaluating domain={domain.identifier} profile={profile} seed={seed}",
                file=sys.stderr,
                flush=True,
            )
            results.append(
                evaluate(
                    checkpoint,
                    domain.games_per_seed,
                    seed,
                    device,
                    baseline,
                    profile,
                    search.nodes,
                    search.beam_width,
                    search.branch_width,
                    search.maximum_actions_per_turn,
                    domain.width,
                    domain.height,
                    domain.action_limit,
                    domain.procedural,
                    domain.players,
                    domain.land_density_per_million,
                    domain.starting_province_size,
                    domain.starting_money,
                    domain.tree_density_per_million,
                    domain.neutral_tower_density_per_million,
                    domain.neutral_capital_density_per_million,
                    domain.grave_density_per_million,
                )
            )
    return {
        "id": domain.identifier,
        "generator": domain.generator,
        "width": domain.width,
        "height": domain.height,
        "players": domain.players,
        "games_per_seed": domain.games_per_seed,
        "unique_maps_per_profile_seed": domain.games_per_seed // domain.players,
        "seeds": domain.seeds,
        "action_limit": domain.action_limit,
        "results": results,
        "aggregate": aggregate_results(results),
        "weakest_seat": minimum_seat_slice(results),
    }


def summarize_domains(domains: list[dict[str, object]]) -> dict[str, object]:
    aggregates = [domain["aggregate"] for domain in domains]
    games = sum(int(aggregate["games"]) for aggregate in aggregates)
    wins = sum(int(aggregate["wins"]) for aggregate in aggregates)
    draws = sum(int(aggregate["draws"]) for aggregate in aggregates)
    losses = sum(int(aggregate["losses"]) for aggregate in aggregates)
    truncations = sum(int(aggregate["truncations"]) for aggregate in aggregates)
    baseline_truncations = sum(
        int(aggregate["baseline_truncations"]) for aggregate in aggregates
    )
    weakest_domain = min(
        domains, key=lambda domain: float(domain["aggregate"]["relative_elo"])
    )
    slices = [
        {
            "domain": domain["id"],
            **domain["weakest_seat"],
        }
        for domain in domains
    ]
    weakest_slice = min(slices, key=lambda result: float(result["score_delta"]))
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "truncations": truncations,
        "truncation_rate": truncations / games,
        "baseline_truncations": baseline_truncations,
        "baseline_truncation_rate": baseline_truncations / games,
        "truncation_rate_delta": (truncations - baseline_truncations) / games,
        "weakest_domain": {
            "id": weakest_domain["id"],
            **weakest_domain["aggregate"],
        },
        "weakest_slice": weakest_slice,
    }


def gate_failures(summary: dict[str, object], gates: Gates) -> list[str]:
    failures = []
    weakest_domain = summary["weakest_domain"]
    weakest_slice = summary["weakest_slice"]
    if float(weakest_domain["relative_elo"]) < gates.minimum_domain_relative_elo:
        failures.append(
            f"weakest domain relative Elo {weakest_domain['relative_elo']:.3f} is below "
            f"{gates.minimum_domain_relative_elo:.3f}"
        )
    if float(weakest_slice["score_delta"]) < gates.minimum_slice_score_delta:
        failures.append(
            f"weakest slice score delta {weakest_slice['score_delta']:.3f} "
            f"is below {gates.minimum_slice_score_delta:.3f}"
        )
    if (
        float(summary["truncation_rate_delta"])
        > gates.maximum_truncation_rate_delta
    ):
        failures.append(
            f"truncation rate delta {summary['truncation_rate_delta']:.6f} exceeds "
            f"{gates.maximum_truncation_rate_delta:.6f}"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--profiles", nargs="+", choices=ALL_PROFILES)
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--search-nodes", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ignore-gates", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    matrix = parse_matrix(arguments.matrix)
    profiles = matrix.profiles if arguments.profiles is None else tuple(arguments.profiles)
    requested_domains = (
        {domain.identifier for domain in matrix.domains}
        if arguments.domains is None
        else set(arguments.domains)
    )
    unknown_domains = requested_domains.difference(
        domain.identifier for domain in matrix.domains
    )
    if unknown_domains:
        parser.error(f"unknown domains: {sorted(unknown_domains)}")
    domains = tuple(
        domain for domain in matrix.domains if domain.identifier in requested_domains
    )
    search = (
        matrix.search
        if arguments.search_nodes is None
        else Search(
            arguments.search_nodes,
            matrix.search.beam_width,
            matrix.search.branch_width,
            matrix.search.maximum_actions_per_turn,
        )
    )
    started = time.perf_counter()
    domain_results = [
        evaluate_domain(
            arguments.checkpoint,
            domain,
            profiles,
            matrix.baseline,
            search,
            arguments.device,
        )
        for domain in domains
    ]
    summary = summarize_domains(domain_results)
    failures = gate_failures(summary, matrix.gates)
    result = {
        "schema_version": 1,
        "kind": "universal_policy_cross_domain_matrix",
        "checkpoint": {
            "path": str(arguments.checkpoint),
            "sha256": checkpoint_digest(arguments.checkpoint),
            "size_bytes": arguments.checkpoint.stat().st_size,
        },
        "matrix": {
            "path": str(arguments.matrix),
            "sha256": matrix_digest(arguments.matrix),
        },
        "profiles": profiles,
        "baseline": matrix.baseline,
        "search": search.__dict__,
        "rating_model": "equal-opponent Plackett-Luce skill delta; arena-specific",
        "device": arguments.device,
        "elapsed_seconds": time.perf_counter() - started,
        "domains": domain_results,
        "summary": summary,
        "gates": matrix.gates.__dict__,
        "gate_failures": failures,
        "passed": not failures,
    }
    print(json.dumps(result, sort_keys=True))
    if arguments.output is not None:
        write_result(arguments.output, result)
    if failures and not arguments.ignore_gates:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
