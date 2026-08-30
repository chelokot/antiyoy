from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import torch


BUNDLE_VERSION = 4
SUPPORTED_BUNDLE_VERSIONS = (1, 2, 3, BUNDLE_VERSION)
BUNDLE_KIND = "routed_policy_bundle"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def source_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path),
        "sha256": digest(path),
        "size_bytes": path.stat().st_size,
    }


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


def register_context_expert(
    experts: dict[str, object],
    sources: dict[str, dict[str, str | int]],
    source_experts: dict[str, str],
    path: Path,
    specialist: dict[str, object],
) -> str:
    source = source_record(path)
    source_sha256 = str(source["sha256"])
    existing = source_experts.get(source_sha256)
    if existing is not None:
        return existing
    expert = f"context:{source_sha256[:16]}"
    if expert in experts:
        expert = f"context:{source_sha256}"
    experts[expert] = specialist["model"]
    sources[expert] = source
    source_experts[source_sha256] = expert
    return expert


def save_bundle(bundle: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(bundle, temporary)
    temporary.replace(output_path)


def build_bundle(
    primary_path: Path,
    route_paths: dict[str, Path],
    output_path: Path,
    context_route_paths: dict[tuple[str, str, int], Path] | None = None,
    seat_context_route_paths: dict[tuple[str, str, int, int], Path] | None = None,
    domain_route_paths: dict[tuple[str, str, int, int, str], Path] | None = None,
) -> dict[str, object]:
    primary = load_source(primary_path)
    profiles = source_profiles(primary)
    routes = {profile: "primary" for profile in profiles}
    experts = {"primary": primary["model"]}
    sources = {"primary": source_record(primary_path)}
    source_experts = {sources["primary"]["sha256"]: "primary"}

    for profile, path in route_paths.items():
        specialist = load_source(path)
        compatible(primary, specialist)
        if profile not in source_profiles(specialist):
            raise ValueError(
                f"specialist curriculum does not contain routed profile: {profile}"
            )
        if profile not in routes:
            profiles.append(profile)
        expert = f"specialist:{profile}"
        routes[profile] = expert
        experts[expert] = specialist["model"]
        source = source_record(path)
        sources[expert] = source
        source_experts[source["sha256"]] = expert
    context_routes = []
    for (profile, generator, players), path in (context_route_paths or {}).items():
        specialist = load_source(path)
        compatible(primary, specialist)
        if profile not in source_profiles(specialist):
            raise ValueError(
                f"specialist curriculum does not contain routed profile: {profile}"
            )
        if profile not in routes:
            profiles.append(profile)
        expert = register_context_expert(
            experts, sources, source_experts, path, specialist
        )
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
    seat_context_routes = []
    for (profile, generator, players, seat), path in (
        seat_context_route_paths or {}
    ).items():
        specialist = load_source(path)
        compatible(primary, specialist)
        if profile not in source_profiles(specialist):
            raise ValueError(f"specialist curriculum does not contain routed profile: {profile}")
        if profile not in routes:
            profiles.append(profile)
        expert = register_context_expert(
            experts, sources, source_experts, path, specialist
        )
        if profile not in routes:
            routes[profile] = expert
        seat_context_routes.append(
            {
                "profile": profile,
                "generator": generator,
                "players": players,
                "seat": seat,
                "expert": expert,
            }
        )
    domain_routes = []
    for (profile, generator, players, seat, domain), path in (
        domain_route_paths or {}
    ).items():
        specialist = load_source(path)
        compatible(primary, specialist)
        if profile not in source_profiles(specialist):
            raise ValueError(f"specialist curriculum does not contain routed profile: {profile}")
        if profile not in routes:
            profiles.append(profile)
        expert = register_context_expert(
            experts, sources, source_experts, path, specialist
        )
        if profile not in routes:
            routes[profile] = expert
        domain_routes.append(
            {
                "profile": profile,
                "generator": generator,
                "players": players,
                "seat": seat,
                "domain": domain,
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
        "seat_context_routes": seat_context_routes,
        "domain_routes": domain_routes,
        "sources": sources,
        "summary": {
            "algorithm": "deterministic_profile_routed_experts",
            "experts": len(experts),
            "profiles": len(routes),
            "context_routes": len(context_routes),
            "seat_context_routes": len(seat_context_routes),
            "domain_routes": len(domain_routes),
        },
    }
    save_bundle(bundle, output_path)
    return bundle


def overlay_bundle(
    base_path: Path,
    route_paths: dict[str, Path],
    output_path: Path,
    context_route_paths: dict[tuple[str, str, int], Path] | None = None,
    seat_context_route_paths: dict[tuple[str, str, int, int], Path] | None = None,
    domain_route_paths: dict[tuple[str, str, int, int, str], Path] | None = None,
) -> dict[str, object]:
    loaded = torch.load(base_path, map_location="cpu", weights_only=False)
    if loaded.get("kind") != BUNDLE_KIND:
        raise ValueError("overlay source is not a routed policy bundle")
    if loaded.get("bundle_version") not in SUPPORTED_BUNDLE_VERSIONS:
        raise ValueError("overlay source bundle version does not match")
    required = {
        "checkpoint_version",
        "observation_version",
        "rule_features",
        "config",
        "experts",
        "routes",
        "sources",
    }
    missing = required.difference(loaded)
    if missing:
        raise ValueError(
            f"overlay bundle is missing fields: {', '.join(sorted(missing))}"
        )
    bundle = copy.deepcopy(loaded)
    experts = dict(bundle["experts"])
    sources = dict(bundle["sources"])
    routes = dict(bundle["routes"])
    context_routes = [dict(route) for route in bundle.get("context_routes", [])]
    seat_context_routes = [
        dict(route) for route in bundle.get("seat_context_routes", [])
    ]
    domain_routes = [dict(route) for route in bundle.get("domain_routes", [])]
    config = dict(bundle["config"])
    profiles = list(config["profiles"])
    source_experts = {
        str(source["sha256"]): expert for expert, source in sources.items()
    }

    def register(profile: str, path: Path) -> str:
        specialist = load_source(path)
        compatible(bundle, specialist)
        if profile not in source_profiles(specialist):
            raise ValueError(
                f"specialist curriculum does not contain routed profile: {profile}"
            )
        if profile not in profiles:
            profiles.append(profile)
        return register_context_expert(
            experts, sources, source_experts, path, specialist
        )

    for profile, path in route_paths.items():
        routes[profile] = register(profile, path)
    for (profile, generator, players), path in (context_route_paths or {}).items():
        expert = register(profile, path)
        context_routes = [
            route
            for route in context_routes
            if (
                route["profile"],
                route["generator"],
                route["players"],
            )
            != (profile, generator, players)
        ]
        context_routes.append(
            {
                "profile": profile,
                "generator": generator,
                "players": players,
                "expert": expert,
            }
        )
        routes.setdefault(profile, expert)
    for (profile, generator, players, seat), path in (
        seat_context_route_paths or {}
    ).items():
        expert = register(profile, path)
        seat_context_routes = [
            route
            for route in seat_context_routes
            if (
                route["profile"],
                route["generator"],
                route["players"],
                route["seat"],
            )
            != (profile, generator, players, seat)
        ]
        seat_context_routes.append(
            {
                "profile": profile,
                "generator": generator,
                "players": players,
                "seat": seat,
                "expert": expert,
            }
        )
        routes.setdefault(profile, expert)
    for (profile, generator, players, seat, domain), path in (
        domain_route_paths or {}
    ).items():
        expert = register(profile, path)
        domain_routes = [
            route
            for route in domain_routes
            if (
                route["profile"],
                route["generator"],
                route["players"],
                route["seat"],
                route["domain"],
            )
            != (profile, generator, players, seat, domain)
        ]
        domain_routes.append(
            {
                "profile": profile,
                "generator": generator,
                "players": players,
                "seat": seat,
                "domain": domain,
                "expert": expert,
            }
        )
        routes.setdefault(profile, expert)

    referenced = set(routes.values())
    referenced.update(str(route["expert"]) for route in context_routes)
    referenced.update(str(route["expert"]) for route in seat_context_routes)
    referenced.update(str(route["expert"]) for route in domain_routes)
    missing_experts = referenced.difference(experts)
    if missing_experts:
        raise ValueError(
            f"overlay bundle references missing experts: {', '.join(sorted(missing_experts))}"
        )
    config["profiles"] = profiles
    bundle.update(
        {
            "bundle_version": BUNDLE_VERSION,
            "config": config,
            "experts": {
                expert: state for expert, state in experts.items() if expert in referenced
            },
            "routes": routes,
            "context_routes": context_routes,
            "seat_context_routes": seat_context_routes,
            "domain_routes": domain_routes,
            "sources": {
                expert: source for expert, source in sources.items() if expert in referenced
            },
            "summary": {
                "algorithm": "deterministic_profile_routed_experts",
                "experts": len(referenced),
                "profiles": len(routes),
                "context_routes": len(context_routes),
                "seat_context_routes": len(seat_context_routes),
                "domain_routes": len(domain_routes),
                "overlay_base": {
                    "path": str(base_path),
                    "sha256": digest(base_path),
                    "size_bytes": base_path.stat().st_size,
                },
            },
        }
    )
    save_bundle(bundle, output_path)
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


def parse_seat_context_routes(
    specifications: list[str],
) -> dict[tuple[str, str, int, int], Path]:
    routes: dict[tuple[str, str, int, int], Path] = {}
    for specification in specifications:
        context, separator, path = specification.partition("=")
        parts = context.rsplit(":", 3)
        if not separator or len(parts) != 4 or not path:
            raise ValueError(
                "seat context routes use PROFILE:GENERATOR:PLAYERS:SEAT=CHECKPOINT"
            )
        profile, generator, players_text, seat_text = parts
        if generator not in ("symmetric_duel_v1", "procedural_v1"):
            raise ValueError(f"unsupported seat context route generator: {generator}")
        try:
            players = int(players_text)
            seat = int(seat_text)
        except ValueError as error:
            raise ValueError(
                "seat context routes use PROFILE:GENERATOR:PLAYERS:SEAT=CHECKPOINT"
            ) from error
        if players < 2 or players > 8:
            raise ValueError("seat context route players must be between two and eight")
        if seat < 0 or seat >= players:
            raise ValueError("seat context route seat must belong to the player range")
        key = (profile, generator, players, seat)
        if key in routes:
            raise ValueError(f"duplicate seat context route: {context}")
        routes[key] = Path(path)
    return routes


def parse_domain_routes(
    specifications: list[str],
) -> dict[tuple[str, str, int, int, str], Path]:
    routes: dict[tuple[str, str, int, int, str], Path] = {}
    for specification in specifications:
        context, separator, path = specification.partition("=")
        parts = context.rsplit(":", 4)
        if not separator or len(parts) != 5 or not path:
            raise ValueError(
                "domain routes use PROFILE:GENERATOR:PLAYERS:SEAT:DOMAIN=CHECKPOINT"
            )
        profile, generator, players_text, seat_text, domain = parts
        if generator not in ("symmetric_duel_v1", "procedural_v1"):
            raise ValueError(f"unsupported domain route generator: {generator}")
        try:
            players = int(players_text)
            seat = int(seat_text)
        except ValueError as error:
            raise ValueError(
                "domain routes use PROFILE:GENERATOR:PLAYERS:SEAT:DOMAIN=CHECKPOINT"
            ) from error
        if players < 2 or players > 8:
            raise ValueError("domain route players must be between two and eight")
        if seat < 0 or seat >= players:
            raise ValueError("domain route seat must belong to the player range")
        if len(domain) != 64:
            raise ValueError("domain route key must be a SHA-256 digest")
        try:
            int(domain, 16)
        except ValueError as error:
            raise ValueError("domain route key must be a SHA-256 digest") from error
        key = (profile, generator, players, seat, domain)
        if key in routes:
            raise ValueError(f"duplicate domain route: {context}")
        routes[key] = Path(path)
    return routes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--route", action="append", default=[], metavar="PROFILE=CHECKPOINT")
    parser.add_argument(
        "--context-route",
        action="append",
        default=[],
        metavar="PROFILE:GENERATOR:PLAYERS=CHECKPOINT",
    )
    parser.add_argument(
        "--seat-context-route",
        action="append",
        default=[],
        metavar="PROFILE:GENERATOR:PLAYERS:SEAT=CHECKPOINT",
    )
    parser.add_argument(
        "--domain-route",
        action="append",
        default=[],
        metavar="PROFILE:GENERATOR:PLAYERS:SEAT:DOMAIN=CHECKPOINT",
    )
    arguments = parser.parse_args()
    builder = overlay_bundle if arguments.overlay else build_bundle
    bundle = builder(
        arguments.primary,
        parse_routes(arguments.route),
        arguments.output,
        parse_context_routes(arguments.context_route),
        parse_seat_context_routes(arguments.seat_context_route),
        parse_domain_routes(arguments.domain_route),
    )
    print(
        f"wrote {arguments.output} with {len(bundle['experts'])} experts and "
        f"{len(bundle['routes'])} routes"
    )


if __name__ == "__main__":
    main()
