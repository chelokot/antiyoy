import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from python.evaluate_matrix import gate_failures, parse_matrix, summarize_domains


def test_release_matrix_is_valid_and_covers_multiplayer() -> None:
    matrix = parse_matrix(Path("benchmarks/configs/universal-cross-domain-v1.json"))

    assert len(matrix.profiles) == 7
    assert {domain.players for domain in matrix.domains} == {2, 3, 4}
    assert {domain.generator for domain in matrix.domains} == {
        "procedural_v1",
        "symmetric_duel_v1",
    }


def test_large_multiplayer_matrix_rotates_every_seat() -> None:
    matrix = parse_matrix(
        Path("benchmarks/configs/universal-cross-domain-5to8p-v1.json")
    )

    assert [domain.players for domain in matrix.domains] == [5, 6, 7, 8]
    assert all(domain.games_per_seed == domain.players for domain in matrix.domains)
    assert all(len(domain.seeds) == 2 for domain in matrix.domains)


def test_matrix_rejects_incomplete_seat_rotation(tmp_path) -> None:
    payload = json.loads(
        Path("benchmarks/configs/universal-cross-domain-v1.json").read_text()
    )
    payload["domains"][3]["games_per_seed"] = 8
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="positive multiple of players"):
        parse_matrix(path)


def test_summary_and_gates_select_cross_domain_weakness() -> None:
    domains = [
        {
            "id": "two-player",
            "aggregate": {
                "games": 8,
                "wins": 6,
                "draws": 0,
                "losses": 2,
                "truncations": 0,
                "baseline_truncations": 0,
                "relative_elo": 190.0,
            },
            "players": 2,
            "weakest_seat": {
                "seat": 1,
                "score": 0.5,
                "score_delta": 0.0,
                "elo_delta": 0.0,
            },
        },
        {
            "id": "four-player",
            "aggregate": {
                "games": 8,
                "wins": 1,
                "draws": 0,
                "losses": 7,
                "truncations": 1,
                "baseline_truncations": 0,
                "relative_elo": -150.0,
            },
            "players": 4,
            "weakest_seat": {
                "seat": 3,
                "score": 0.0,
                "score_delta": -0.25,
                "elo_delta": -300.0,
            },
        },
    ]

    summary = summarize_domains(domains)
    failures = gate_failures(
        summary,
        parse_matrix(Path("benchmarks/configs/universal-cross-domain-v1.json")).gates,
    )

    assert summary["weakest_domain"]["id"] == "four-player"
    assert summary["weakest_slice"]["domain"] == "four-player"
    assert len(failures) == 3
