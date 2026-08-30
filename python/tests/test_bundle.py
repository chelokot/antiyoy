from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from antiyoy_rl import OBSERVATION_VERSION
from antiyoy_rl.model import RULE_FEATURES, UniversalPolicy
from python.build_bundle import BUNDLE_KIND, build_bundle, parse_routes
from python.evaluate import load_policy
from python.train import CHECKPOINT_VERSION


def write_checkpoint(path: Path, missing_source: float, layers: int = 1) -> None:
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
                "profiles": [
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


def test_route_parser_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_routes(["classic_generic_2022=one.pt", "classic_generic_2022=two.pt"])
