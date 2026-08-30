from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from python.build_bundle import build_bundle
from python.extract_expert import extract_expert
from python.tests.test_bundle import write_checkpoint


def test_extract_expert_preserves_shared_default_routes(tmp_path: Path) -> None:
    primary = tmp_path / "primary.pt"
    specialist = tmp_path / "specialist.pt"
    bundle = tmp_path / "bundle.pt"
    extracted = tmp_path / "extracted.pt"
    write_checkpoint(primary, 1.0)
    write_checkpoint(specialist, 7.0)
    build_bundle(
        primary,
        {"online_experimental_v2_260801": specialist},
        bundle,
    )

    checkpoint = extract_expert(bundle, "classic_generic_2022", extracted)

    assert extracted.is_file()
    assert checkpoint["config"]["profiles"] == ["classic_generic_2022"]
    assert torch.all(checkpoint["model"]["missing_source"] == 1.0)
    assert checkpoint["summary"]["source_bundle_sha256"]


def test_extract_expert_rejects_unknown_default_route(tmp_path: Path) -> None:
    primary = tmp_path / "primary.pt"
    bundle = tmp_path / "bundle.pt"
    write_checkpoint(primary, 1.0, profiles=["classic_generic_2022"])
    build_bundle(primary, {}, bundle)

    with pytest.raises(ValueError, match="no default route"):
        extract_expert(bundle, "online_default_v1", tmp_path / "missing.pt")
