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
