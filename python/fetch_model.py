from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "model-registry" / "models.json"


@dataclass(frozen=True)
class ModelArtifact:
    identifier: str
    asset: str
    url: str
    sha256: str
    size_bytes: int


def registry() -> list[ModelArtifact]:
    payload = json.loads(REGISTRY.read_text())
    return [
        ModelArtifact(
            identifier=model["id"],
            asset=model["asset"],
            url=model["url"],
            sha256=model["sha256"],
            size_bytes=model["size_bytes"],
        )
        for model in payload["models"]
    ]


def fetch(identifier: str, destination: Path | None) -> Path:
    artifacts = {artifact.identifier: artifact for artifact in registry()}
    artifact = artifacts.get(identifier)
    if artifact is None:
        raise ValueError(f"unknown model: {identifier}")
    target = destination or ROOT / "models" / artifact.asset
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(artifact.url) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
            digest.update(block)
            size += len(block)
    if size != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
        temporary.unlink()
        raise ValueError("downloaded model failed size or SHA-256 verification")
    temporary.replace(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "identifier", nargs="?", default="universal-distilled-ppo-2026-08-29"
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    print(fetch(arguments.identifier, arguments.output))


if __name__ == "__main__":
    main()
