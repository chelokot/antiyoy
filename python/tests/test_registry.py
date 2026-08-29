import re
from pathlib import Path

from python.fetch_model import registry


def test_model_registry_has_unique_verified_artifacts() -> None:
    artifacts = registry()
    assert artifacts
    assert len({artifact.identifier for artifact in artifacts}) == len(artifacts)
    for artifact in artifacts:
        assert artifact.size_bytes > 0
        assert re.fullmatch(r"[0-9a-f]{64}", artifact.sha256)
        assert artifact.url.startswith("https://github.com/chelokot/antiyoy/releases/")


def test_neural_arena_page_has_board_and_legal_action_controls() -> None:
    page = Path("python/policy_arena.html").read_text()

    assert "Antiyoy Neural Policy Arena" in page
    assert 'id="board"' in page
    assert 'id="actions"' in page
    assert "/api/action" in page
    assert "HUMAN CYAN · BETA AMBER" in page
