import re

from python.fetch_model import registry


def test_model_registry_has_unique_verified_artifacts() -> None:
    artifacts = registry()
    assert artifacts
    assert len({artifact.identifier for artifact in artifacts}) == len(artifacts)
    for artifact in artifacts:
        assert artifact.size_bytes > 0
        assert re.fullmatch(r"[0-9a-f]{64}", artifact.sha256)
        assert artifact.url.startswith("https://github.com/chelokot/antiyoy/releases/")
