from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

try:
    from .build_bundle import BUNDLE_KIND, SUPPORTED_BUNDLE_VERSIONS
except ImportError:
    from build_bundle import BUNDLE_KIND, SUPPORTED_BUNDLE_VERSIONS


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def extract_expert(bundle_path: Path, profile: str, output_path: Path) -> dict[str, object]:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if bundle.get("kind") != BUNDLE_KIND:
        raise ValueError("source is not a routed policy bundle")
    if bundle.get("bundle_version") not in SUPPORTED_BUNDLE_VERSIONS:
        raise ValueError("policy bundle version is not supported")
    expert = bundle["routes"].get(profile)
    if expert is None:
        raise ValueError(f"policy bundle has no default route for profile: {profile}")
    profiles = [
        routed_profile
        for routed_profile, routed_expert in bundle["routes"].items()
        if routed_expert == expert
    ]
    config = dict(bundle["config"])
    config["profile"] = None
    config["profiles"] = profiles
    checkpoint = {
        "model": bundle["experts"][expert],
        "checkpoint_version": bundle["checkpoint_version"],
        "observation_version": bundle["observation_version"],
        "rule_features": bundle["rule_features"],
        "config": config,
        "summary": {
            "algorithm": "extracted_immutable_bundle_expert",
            "source_bundle": str(bundle_path),
            "source_bundle_sha256": digest(bundle_path),
            "source_expert": expert,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output_path)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("profile")
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    checkpoint = extract_expert(arguments.bundle, arguments.profile, arguments.output)
    print(
        f"wrote {arguments.output} from {checkpoint['summary']['source_expert']} "
        f"for {len(checkpoint['config']['profiles'])} profiles"
    )


if __name__ == "__main__":
    main()
