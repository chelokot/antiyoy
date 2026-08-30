from __future__ import annotations

from collections.abc import Iterable


def matching_expert(
    routes: Iterable[dict[str, object]],
    selector: dict[str, object],
    route_kind: str,
) -> str | None:
    matches = [
        route
        for route in routes
        if all(route.get(field) == value for field, value in selector.items())
    ]
    if len(matches) > 1:
        raise ValueError(f"policy bundle contains duplicate {route_kind} routes")
    if not matches:
        return None
    return str(matches[0]["expert"])


def select_bundle_expert(
    checkpoint: dict[str, object],
    profile: str,
    generator: str | None = None,
    players: int | None = None,
    seat: int | None = None,
    domain: str | None = None,
) -> str:
    routes = checkpoint["routes"]
    selected = routes.get(profile)
    if selected is None:
        raise ValueError(f"policy bundle has no route for profile: {profile}")
    context = matching_expert(
        checkpoint.get("context_routes", []),
        {"profile": profile, "generator": generator, "players": players},
        "context",
    )
    if context is not None:
        selected = context
    seat_context = matching_expert(
        checkpoint.get("seat_context_routes", []),
        {
            "profile": profile,
            "generator": generator,
            "players": players,
            "seat": seat,
        },
        "seat context",
    )
    if seat_context is not None:
        selected = seat_context
    exact_domain = matching_expert(
        checkpoint.get("domain_routes", []),
        {
            "profile": profile,
            "generator": generator,
            "players": players,
            "seat": seat,
            "domain": domain,
        },
        "domain",
    )
    if exact_domain is not None:
        selected = exact_domain
    return str(selected)
