from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


BUNDLE_VERSION = 2
SUPPORTED_BUNDLE_VERSIONS = (1, BUNDLE_VERSION)
BUNDLE_KIND = "routed_policy_bundle"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def load_source(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "model",
        "checkpoint_version",
        "observation_version",
        "rule_features",
        "config",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {', '.join(sorted(missing))}")
    return checkpoint


def compatible(primary: dict[str, object], specialist: dict[str, object]) -> None:
    for field in ("checkpoint_version", "observation_version", "rule_features"):
        if specialist[field] != primary[field]:
            raise ValueError(f"bundle source {field} does not match")
    primary_config = primary["config"]
    specialist_config = specialist["config"]
    for field in ("hidden", "layers"):
        if specialist_config[field] != primary_config[field]:
            raise ValueError(f"bundle source architecture {field} does not match")


def source_profiles(checkpoint: dict[str, object]) -> list[str]:
    config = checkpoint["config"]
    profiles = config.get("profiles")
    if profiles is not None:
        return list(profiles)
    profile = config.get("profile")
    if profile is None:
        raise ValueError("checkpoint does not declare a profile curriculum")
    return [profile]


def build_bundle(
    primary_path: Path,
    route_paths: dict[str, Path],
    output_path: Path,
    context_route_paths: dict[tuple[str, str, int], Path] | None = None,
) -> dict[str, object]:
    primary = load_source(primary_path)
    profiles = source_profiles(primary)
    routes = {profile: "primary" for profile in profiles}
    experts = {"primary": primary["model"]}
    sources = {
        "primary": {
            "path": str(primary_path),
            "sha256": digest(primary_path),
            "size_bytes": primary_path.stat().st_size,
        }
    }
    source_experts = {sources["primary"]["sha256"]: "primary"}
    for profile, path in route_paths.items():
        specialist = load_source(path)
        compatible(primary, specialist)
        if profile not in source_profiles(specialist):
            raise ValueError(f"specialist curriculum does not contain routed profile: {profile}")
        if profile not in routes:
            profiles.append(profile)
        expert = f"specialist:{profile}"
        routes[profile] = expert
        experts[expert] = specialist["model"]
        source = {
            "path": str(path),
            "sha256": digest(path),
            "size_bytes": path.stat().st_size,
        }
        sources[expert] = source
        source_experts[source["sha256"]] = expert
    context_routes = []
    for (profile, generator, players), path in (context_route_paths or {}).items():
        specialist = load_source(path)
        compatible(primary, specialist)
        if profile not in source_profiles(specialist):
            raise ValueError(f"specialist curriculum does not contain routed profile: {profile}")
        if profile not in routes:
            profiles.append(profile)
        source_sha256 = digest(path)
        expert = source_experts.get(source_sha256)
        if expert is None:
            expert = f"context:{source_sha256[:16]}"
            experts[expert] = specialist["model"]
            sources[expert] = {
                "path": str(path),
                "sha256": source_sha256,
                "size_bytes": path.stat().st_size,
            }
            source_experts[source_sha256] = expert
        if profile not in routes:
            routes[profile] = expert
        context_routes.append(
            {
                "profile": profile,
                "generator": generator,
                "players": players,
                "expert": expert,
            }
        )
    config = dict(primary["config"])
    config["profile"] = None
    config["profiles"] = profiles
    bundle = {
        "kind": BUNDLE_KIND,
        "bundle_version": BUNDLE_VERSION,
        "checkpoint_version": primary["checkpoint_version"],
        "observation_version": primary["observation_version"],
        "rule_features": primary["rule_features"],
        "config": config,
        "experts": experts,
        "routes": routes,
        "context_routes": context_routes,
        "sources": sources,
        "summary": {
            "algorithm": "deterministic_profile_routed_experts",
            "experts": len(experts),
            "profiles": len(routes),
            "context_routes": len(context_routes),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(bundle, temporary)
    temporary.replace(output_path)
    return bundle


def parse_routes(specifications: list[str]) -> dict[str, Path]:
    routes: dict[str, Path] = {}
    for specification in specifications:
        profile, separator, path = specification.partition("=")
        if not separator or not profile or not path:
            raise ValueError("routes use PROFILE=CHECKPOINT")
        if profile in routes:
            raise ValueError(f"duplicate route: {profile}")
        routes[profile] = Path(path)
    return routes


def parse_context_routes(
    specifications: list[str],
) -> dict[tuple[str, str, int], Path]:
    routes: dict[tuple[str, str, int], Path] = {}
    for specification in specifications:
        context, separator, path = specification.partition("=")
        parts = context.rsplit(":", 2)
        if not separator or len(parts) != 3 or not path:
            raise ValueError(
                "context routes use PROFILE:GENERATOR:PLAYERS=CHECKPOINT"
            )
        profile, generator, players_text = parts
        if generator not in ("symmetric_duel_v1", "procedural_v1"):
            raise ValueError(f"unsupported context route generator: {generator}")
        try:
            players = int(players_text)
        except ValueError as error:
            raise ValueError(
                "context routes use PROFILE:GENERATOR:PLAYERS=CHECKPOINT"
            ) from error
        if players < 2 or players > 8:
            raise ValueError("context route players must be between two and eight")
        key = (profile, generator, players)
        if key in routes:
            raise ValueError(f"duplicate context route: {context}")
        routes[key] = Path(path)
    return routes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--route", action="append", default=[], metavar="PROFILE=CHECKPOINT")
    parser.add_argument(
        "--context-route",
        action="append",
        default=[],
        metavar="PROFILE:GENERATOR:PLAYERS=CHECKPOINT",
    )
    arguments = parser.parse_args()
    bundle = build_bundle(
        arguments.primary,
        parse_routes(arguments.route),
        arguments.output,
        parse_context_routes(arguments.context_route),
    )
    print(
        f"wrote {arguments.output} with {len(bundle['experts'])} experts and "
        f"{len(bundle['routes'])} routes"
    )


if __name__ == "__main__":
    main()
