from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


BUNDLE_VERSION = 1
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
    for profile, path in route_paths.items():
        if profile not in routes:
            raise ValueError(f"routed profile is not in the primary curriculum: {profile}")
        specialist = load_source(path)
        compatible(primary, specialist)
        expert = f"specialist:{profile}"
        routes[profile] = expert
        experts[expert] = specialist["model"]
        sources[expert] = {
            "path": str(path),
            "sha256": digest(path),
            "size_bytes": path.stat().st_size,
        }
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
        "sources": sources,
        "summary": {
            "algorithm": "deterministic_profile_routed_experts",
            "experts": len(experts),
            "profiles": len(routes),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--route", action="append", default=[], metavar="PROFILE=CHECKPOINT")
    arguments = parser.parse_args()
    bundle = build_bundle(arguments.primary, parse_routes(arguments.route), arguments.output)
    print(
        f"wrote {arguments.output} with {len(bundle['experts'])} experts and "
        f"{len(bundle['routes'])} routes"
    )


if __name__ == "__main__":
    main()
