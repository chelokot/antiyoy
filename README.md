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

Benchmark procedural four-player domain randomization separately:

```bash
RAYON_NUM_THREADS=8 cargo run --release -p antiyoy-cli -- \
  rl-bench --map procedural --width 31 --height 21 --players 4 \
  --environments 256 --transitions 500000 --json
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
Its live map panel also creates reproducible two-to-eight-player
`procedural_v1` scenarios entirely in WASM, with editable dimensions, seed, and
land density. Human mode derives every destination action from the core's legal
mask and advances all bot opponents until control returns to the player.

## Multiplayer rooms

The multiplayer service owns each game, checks an optimistic revision and a
256-bit human seat credential, applies actions through the core, advances native
bots, broadcasts versioned snapshots, and exposes the exact verified replay.
Independent room locks prevent a bot in one match from blocking other matches.

```bash
cargo run --release -p antiyoy-server -- \
  --host 127.0.0.1 --port 8080 \
  --maximum-rooms 1024 --maximum-cells 4096 \
  --maximum-action-limit 10000 --update-capacity 32 \
  --data-directory server-data
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
Every accepted transition is atomically persisted before it is acknowledged.
On restart, the service verifies each replay, reconstructs the authoritative
game, and replays native bot decisions to restore their exact RNG state. Only
token hashes are written to disk.
Terminal rooms enter the server's Elo ledger only after replay verification;
`GET /v1/league` returns its versioned standings and match ledger. Room files
and `league.json` use fsync-plus-rename atomic replacement.
Match snapshots expose `NotFinished`, `Pending`, `Recorded`, or `Duplicate`
rating status; an identical replay cannot inflate Elo twice.

Python trainers can install the native environment in an active virtual
environment:

```bash
cd python
maturin develop --release
pytest -q tests
```

`ProceduralConfig` and `VectorEnv.procedural` generate deterministic connected
maps with configurable land, players, starts, money, trees, neutral structures,
and graves. Training with `python train.py --procedural` regenerates the whole
scenario on every episode seed and records every worker's exact generator
config in the checkpoint. The original `procedural_v1` generator is intended
for domain randomization and is explicitly separate from upstream map-seed
compatibility.

Versioned scenario objectives cover domination, alliances, survival, target
elimination, exact economy thresholds, and ensuring a specified player's
victory. RL termination reports objective satisfaction separately from the
actual winner; the complete contract is in
[`docs/objectives.md`](docs/objectives.md).

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
