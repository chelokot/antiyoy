# antiyoy

A deterministic, configurable hex strategy engine built for reinforcement
learning, large-scale self-play, reproducible tournaments, and the browser.

The project is organized around one invariant: the interactive game, headless
simulation, training workers, replay viewer, and multiplayer server all execute
the same Rust rules engine.

[Open the live WebAssembly arena](https://antiyoy-arena-lab.chelokot.chatgpt.site)

## Direction

- Versioned compatibility profiles for Classic, Slay, Online, Duel, and
  Experimental v1/v2 economies and construction rules.
- Compact deterministic state and actions with stable binary replays.
- Batched headless environments and Python bindings for RL training.
- Browser rendering through WebAssembly with no duplicate game logic.
- Reproducible leagues, calibrated Elo ratings, and downloadable agents.
- Configurable maps, economy, movement, vegetation, diplomacy, and objectives.

## Workspace

| Package | Responsibility |
| --- | --- |
| `antiyoy-core` | Deterministic rules, state transitions, map generation, observations |
| `antiyoy-agents` | Baseline policies sharing the authoritative legal-action stream |
| `antiyoy-rl` | Parallel environments, tensor observations, action features, rewards |
| `antiyoy-python` | GIL-releasing NumPy bindings built with PyO3 and Maturin |
| `antiyoy-eval` | Symmetric matches, tournaments, adjudication, and ratings |
| `antiyoy-protocol` | Versioned replay and network messages |
| `antiyoy-server` | Authoritative HTTP/WebSocket match rooms for humans and bots |
| `antiyoy-cli` | Headless games, validation, tournaments, and benchmarks |
| `antiyoy-wasm` | Browser-safe bindings over the core engine |

## RL throughput

The `rl-bench` path excludes replay serialization and exercises the workload a
trainer sees: state transitions, exact legal actions, dense reward components,
and a complete structure-of-arrays observation after every vector step.

```bash
RAYON_NUM_THREADS=8 cargo run --release -p antiyoy-cli -- \
  rl-bench --environments 256 --transitions 500000 --json
```

The versioned tensor contract is documented in [`docs/rl.md`](docs/rl.md).
Reproducible machine results live under [`benchmarks/`](benchmarks/).

## Auditable leagues

The league runner persists a versioned JSON leaderboard and one compact binary
replay per rated match. A result enters Elo only after the engine replays every
action and verifies the final state and declared outcome. Exact duplicate
matches are rejected.

```bash
cargo run --release -p antiyoy-cli -- league \
  --games 100 --rules online-duel-v1 \
  --state runs/duel/league.json --replay-dir runs/duel/replays --json
```

Any stored game can be checked independently with
`cargo run --release -p antiyoy-cli -- verify <replay>`.
The live WebAssembly arena accepts the same `.antiyoy` file and provides
play/pause, single-step, backward seek, and clickable per-hex state inspection.

## Multiplayer rooms

The multiplayer service owns each game, checks an optimistic revision and a
256-bit human seat credential, applies actions through the core, advances native
bots, broadcasts versioned snapshots, and exposes the exact verified replay.
Independent room locks prevent a bot in one match from blocking other matches.

```bash
cargo run --release -p antiyoy-server -- \
  --host 127.0.0.1 --port 8080 \
  --maximum-rooms 1024 --maximum-cells 4096 \
  --maximum-action-limit 10000 --update-capacity 32
```

Create a room with `POST /v1/matches`, inspect it with
`GET /v1/matches/{match_id}`, download its binary replay from
`GET /v1/matches/{match_id}/replay`, or connect to
`GET /v1/matches/{match_id}/watch`. A connection begins as a read-only spectator;
a human sends the versioned `ClientMessage::Authenticate` frame with its seat
and returned token before submitting actions. Network structures carry
`NETWORK_SCHEMA_VERSION` and every update includes a deterministic state digest.
`DELETE /v1/matches/{match_id}?seat=…` with the seat token in an Authorization
Bearer header closes and removes a room. All memory-amplifying limits are
explicit server configuration, and room capacity is checked atomically.

Python trainers can install the native environment in an active virtual
environment:

```bash
cd python
maturin develop --release
pytest -q tests
```

## Downloadable policies

The model registry contains immutable release assets with size and SHA-256
verification. Fetch the current experimental universal policy with:

```bash
python python/fetch_model.py
PYTHONPATH=python python python/evaluate.py \
  models/universal-distilled-ppo-2026-08-29.pt \
  --games 128 --baseline greedy --profile online_duel_v1
```

This checkpoint is not assigned an absolute Elo. Its held-out relative ratings
against the native greedy baseline are recorded per profile under
[`benchmarks/`](benchmarks/); Experimental v2 remains below that baseline.

## Status

The engine is under active construction. The compatibility matrix in
[`docs/rules.md`](docs/rules.md) is the source of truth for implemented rules.

## Intellectual property

This is an unofficial implementation. It does not include artwork, audio, or
source code from the original game. Antiyoy is by Yiotro; this project is not
affiliated with or endorsed by Yiotro.

The original code in this repository is licensed under MIT.
