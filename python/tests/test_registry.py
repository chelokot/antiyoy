import json
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


def test_registry_defaults_to_the_universal_two_to_eight_player_bundle() -> None:
    payload = json.loads(Path("model-registry/models.json").read_text())
    current = payload["models"][0]

    assert current["id"] == "universal-routed-2to8p-engine-v6-2026-08-31"
    assert current["status"] == "beta"
    assert current["experts"] == 41
    assert current["profile_routes"] == 7
    assert current["context_routes"] == 49
    assert current["seat_context_routes"] == 21
    assert current["domain_routes"] == 10
    engine_v6_arena = current["engine_v6_fixed_duel_vs_search_2048"]
    assert engine_v6_arena["games"] == 336
    assert engine_v6_arena["record"] == "336-0-0"
    assert engine_v6_arena["truncations"] == 0
    assert engine_v6_arena["profile_records"]["online_default_v1"] == "48-0-0"
    assert sum(model["status"] == "beta" for model in payload["models"]) == 1


def test_registry_records_the_frozen_policy_puct_specialist() -> None:
    payload = json.loads(Path("model-registry/models.json").read_text())
    specialist = next(
        model
        for model in payload["models"]
        if model["id"] == "classic-generic-duel-value-v3-2026-08-31"
    )

    assert specialist["policy_parameters_frozen"] is True
    assert specialist["value_calibration"]["heldout_sign_accuracy_after"] > 0.92
    puct = specialist["policy_guided_puct_vs_direct_policy"]
    assert puct["games"] == 512
    assert puct["record"] == "280-0-232"
    assert puct["truncations"] == 0
    assert puct["relative_elo"] > 32


def test_neural_arena_page_has_board_and_legal_action_controls() -> None:
    page = Path("python/policy_arena.html").read_text()

    assert "Antiyoy Neural Policy Arena" in page
    assert 'id="board"' in page
    assert 'id="actions"' in page
    assert "/api/action" in page
    assert "HUMAN CYAN · BETA AMBER" in page
