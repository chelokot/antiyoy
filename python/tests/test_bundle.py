from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from antiyoy_rl import OBSERVATION_VERSION
from antiyoy_rl.model import RULE_FEATURES, UniversalPolicy
from python.build_bundle import (
    BUNDLE_KIND,
    build_bundle,
    digest,
    overlay_bundle,
    parse_context_routes,
    parse_domain_routes,
    parse_routes,
    parse_seat_context_routes,
)
from python.evaluate import load_policy
from python.train import CHECKPOINT_VERSION


def write_checkpoint(
    path: Path,
    missing_source: float,
    layers: int = 1,
    profiles: list[str] | None = None,
) -> None:
    model = UniversalPolicy(hidden=16, layers=layers)
    model.missing_source.data.fill_(missing_source)
    torch.save(
        {
            "model": model.state_dict(),
            "checkpoint_version": CHECKPOINT_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "rule_features": RULE_FEATURES,
            "config": {
                "hidden": 16,
                "layers": layers,
                "profile": None,
                "profiles": profiles
                or [
                    "classic_generic_2022",
                    "online_experimental_v2_260801",
                ],
                "width": 11,
                "height": 9,
                "action_limit": 1000,
                "fog": False,
            },
        },
        path,
    )


def test_bundle_routes_profiles_to_verified_experts(tmp_path: Path) -> None:
    primary = tmp_path / "primary.pt"
    specialist = tmp_path / "specialist.pt"
    bundle_path = tmp_path / "bundle.pt"
    write_checkpoint(primary, 1.0)
    write_checkpoint(specialist, 7.0)

    bundle = build_bundle(
        primary,
        {"online_experimental_v2_260801": specialist},
        bundle_path,
    )

    assert bundle["kind"] == BUNDLE_KIND
    assert bundle["routes"] == {
        "classic_generic_2022": "primary",
        "online_experimental_v2_260801": (
            "specialist:online_experimental_v2_260801"
        ),
    }
    assert bundle_path.is_file()
    primary_model, primary_config = load_policy(
        bundle_path, torch.device("cpu"), "classic_generic_2022"
    )
    specialist_model, specialist_config = load_policy(
        bundle_path, torch.device("cpu"), "online_experimental_v2_260801"
    )
    assert torch.all(primary_model.missing_source == 1.0)
    assert torch.all(specialist_model.missing_source == 7.0)
    assert primary_config["selected_expert"] == "primary"
    assert specialist_config["selected_expert"].startswith("specialist:")


def test_bundle_rejects_incompatible_architectures(tmp_path: Path) -> None:
    primary = tmp_path / "primary.pt"
    specialist = tmp_path / "specialist.pt"
    write_checkpoint(primary, 1.0)
    write_checkpoint(specialist, 2.0, layers=2)

    with pytest.raises(ValueError, match="architecture layers"):
        build_bundle(
            primary,
            {"online_experimental_v2_260801": specialist},
            tmp_path / "bundle.pt",
        )


def test_bundle_accepts_a_verified_specialist_only_profile(tmp_path: Path) -> None:
    primary = tmp_path / "primary.pt"
    specialist = tmp_path / "specialist.pt"
    bundle_path = tmp_path / "bundle.pt"
    write_checkpoint(primary, 1.0, profiles=["classic_generic_2022"])
    write_checkpoint(
        specialist,
        7.0,
        profiles=["online_experimental_v2_260801"],
    )

    bundle = build_bundle(
        primary,
        {"online_experimental_v2_260801": specialist},
        bundle_path,
    )

    assert bundle["config"]["profiles"] == [
        "classic_generic_2022",
        "online_experimental_v2_260801",
    ]
    assert bundle["routes"]["online_experimental_v2_260801"].startswith(
        "specialist:"
    )


def test_bundle_rejects_a_route_outside_specialist_curriculum(tmp_path: Path) -> None:
    primary = tmp_path / "primary.pt"
    specialist = tmp_path / "specialist.pt"
    write_checkpoint(primary, 1.0, profiles=["classic_generic_2022"])
    write_checkpoint(specialist, 7.0, profiles=["online_default_v1"])

    with pytest.raises(ValueError, match="specialist curriculum"):
        build_bundle(
            primary,
            {"online_experimental_v2_260801": specialist},
            tmp_path / "bundle.pt",
        )


def test_route_parser_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_routes(["classic_generic_2022=one.pt", "classic_generic_2022=two.pt"])


def test_bundle_routes_exact_map_and_player_context(tmp_path: Path) -> None:
    primary = tmp_path / "primary.pt"
    procedural = tmp_path / "procedural.pt"
    bundle_path = tmp_path / "bundle.pt"
    write_checkpoint(primary, 1.0, profiles=["classic_generic_2022"])
    write_checkpoint(procedural, 9.0, profiles=["classic_generic_2022"])

    bundle = build_bundle(
        primary,
        {},
        bundle_path,
        {("classic_generic_2022", "procedural_v1", 4): procedural},
    )

    duel_model, duel_config = load_policy(
        bundle_path,
        torch.device("cpu"),
        "classic_generic_2022",
        "symmetric_duel_v1",
        2,
    )
    procedural_model, procedural_config = load_policy(
        bundle_path,
        torch.device("cpu"),
        "classic_generic_2022",
        "procedural_v1",
        4,
    )
    assert torch.all(duel_model.missing_source == 1.0)
    assert torch.all(procedural_model.missing_source == 9.0)
    assert duel_config["selected_expert"] == "primary"
    assert procedural_config["selected_expert"].startswith("context:")
    assert len(bundle["experts"]) == 2


def test_context_route_parser_rejects_duplicates() -> None:
    route = "classic_generic_2022:procedural_v1:4"
    with pytest.raises(ValueError, match="duplicate context"):
        parse_context_routes([f"{route}=one.pt", f"{route}=two.pt"])


def test_bundle_routes_exact_multiplayer_seat(tmp_path: Path) -> None:
    primary = tmp_path / "primary.pt"
    general = tmp_path / "general.pt"
    second_seat = tmp_path / "second-seat.pt"
    bundle_path = tmp_path / "bundle.pt"
    write_checkpoint(primary, 1.0, profiles=["classic_generic_2022"])
    write_checkpoint(general, 5.0, profiles=["classic_generic_2022"])
    write_checkpoint(second_seat, 9.0, profiles=["classic_generic_2022"])

    bundle = build_bundle(
        primary,
        {},
        bundle_path,
        {("classic_generic_2022", "procedural_v1", 3): general},
        {("classic_generic_2022", "procedural_v1", 3, 1): second_seat},
    )

    general_model, general_config = load_policy(
        bundle_path,
        torch.device("cpu"),
        "classic_generic_2022",
        "procedural_v1",
        3,
        0,
    )
    seat_model, seat_config = load_policy(
        bundle_path,
        torch.device("cpu"),
        "classic_generic_2022",
        "procedural_v1",
        3,
        1,
    )
    assert torch.all(general_model.missing_source == 5.0)
    assert torch.all(seat_model.missing_source == 9.0)
    assert general_config["selected_expert"] != seat_config["selected_expert"]
    assert len(bundle["seat_context_routes"]) == 1


def test_seat_context_route_parser_validates_seat_range() -> None:
    route = "classic_generic_2022:procedural_v1:3:3=expert.pt"
    with pytest.raises(ValueError, match="player range"):
        parse_seat_context_routes([route])


def test_bundle_overlay_replaces_route_and_prunes_old_expert(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary.pt"
    previous = tmp_path / "previous.pt"
    replacement = tmp_path / "replacement.pt"
    base_path = tmp_path / "base.pt"
    output_path = tmp_path / "overlay.pt"
    profile = "classic_generic_2022"
    context = (profile, "procedural_v1", 4, 2)
    write_checkpoint(primary, 1.0, profiles=[profile])
    write_checkpoint(previous, 5.0, profiles=[profile])
    write_checkpoint(replacement, 9.0, profiles=[profile])
    base = build_bundle(
        primary,
        {},
        base_path,
        seat_context_route_paths={context: previous},
    )

    overlay = overlay_bundle(
        base_path,
        {},
        output_path,
        seat_context_route_paths={context: replacement},
    )

    selected, config = load_policy(
        output_path,
        torch.device("cpu"),
        profile,
        "procedural_v1",
        4,
        2,
    )
    assert torch.all(selected.missing_source == 9.0)
    assert config["selected_expert"] in overlay["experts"]
    assert len(base["experts"]) == 2
    assert len(overlay["experts"]) == 2
    assert digest(previous) not in {
        source["sha256"] for source in overlay["sources"].values()
    }


def test_bundle_routes_an_exact_arena_domain(tmp_path: Path) -> None:
    primary = tmp_path / "primary.pt"
    specialist = tmp_path / "domain.pt"
    bundle_path = tmp_path / "bundle.pt"
    profile = "classic_generic_2022"
    domain = "ab" * 32
    write_checkpoint(primary, 1.0, profiles=[profile])
    write_checkpoint(specialist, 9.0, profiles=[profile])
    bundle = build_bundle(
        primary,
        {},
        bundle_path,
        domain_route_paths={
            (profile, "procedural_v1", 4, 2, domain): specialist
        },
    )

    fallback, _ = load_policy(
        bundle_path,
        torch.device("cpu"),
        profile,
        "procedural_v1",
        4,
        2,
    )
    selected, config = load_policy(
        bundle_path,
        torch.device("cpu"),
        profile,
        "procedural_v1",
        4,
        2,
        domain,
    )

    assert torch.all(fallback.missing_source == 1.0)
    assert torch.all(selected.missing_source == 9.0)
    assert config["selected_expert"] == bundle["domain_routes"][0]["expert"]


def test_domain_route_parser_requires_a_sha256_key() -> None:
    route = "classic_generic_2022:procedural_v1:4:2"
    with pytest.raises(ValueError, match="SHA-256"):
        parse_domain_routes([f"{route}:short=expert.pt"])
