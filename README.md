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

## Deterministic search agent

The native whole-turn agent combines bounded beam search with a complete greedy
turn as its fallback candidate. It caches the chosen action sequence only while
the authoritative state matches exactly, so interactive and tournament play
remain deterministic. Search size is an explicit compute knob:

```bash
cargo run --release -p antiyoy-cli -- compare \
  --pairs 16 --width 11 --height 9 --rules online-duel-v1 \
  --first search --second greedy --search-nodes 4096 --json
```

The held-out mirrored evaluation under [`benchmarks/`](benchmarks/) covers all
seven compatibility profiles and reports paired relative Elo rather than an
unsubstantiated absolute rating. Browser search remains selectable at 64, 256,
or 2048 nodes, and rated placement always uses the fixed 2048-node agent.

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
mask and offers both measured search opponents and the exported neural beta.
The neural route consumes the authoritative Rust observation and legal-action
list in ONNX Runtime Web. The arena loads only the two seat routes relevant to
the current match. Both Classic Generic seats now use independently gated
zero-search Soft-PUCT students, so the side selector can place the human first
or second without falling back to the source policy. Other profiles retain
their accepted routed experts. Rated placement uses the Classic Generic arena, alternates the human
between both seats, assigns a fresh deterministic seed to every attempt, and
stores a device-local Elo relative to the fixed 2048-node search opponent.
This local calibration is explicitly separate from the verified multiplayer
league.

## Multiplayer rooms

The multiplayer service owns each two-to-eight-player game, checks an optimistic
revision and a 256-bit human seat credential, applies actions through the core,
advances native bots, broadcasts versioned snapshots, and exposes the exact
verified replay. Rooms can use the symmetric two-player arena or the complete
typed `procedural_v1` generator configuration. Independent room locks prevent a
bot in one match from blocking other matches.

```bash
cargo run --release -p antiyoy-server -- \
  --host 127.0.0.1 --port 8080 \
  --maximum-rooms 1024 --maximum-cells 4096 \
  --maximum-action-limit 10000 --update-capacity 32 \
  --search-nodes 2048 \
  --data-directory server-data
```

Create a room with `POST /v1/matches`, inspect it with
`GET /v1/matches/{match_id}`, download its binary replay from
`GET /v1/matches/{match_id}/replay`, or connect to
`GET /v1/matches/{match_id}/watch`. A connection begins as a read-only spectator;
a human sends the versioned `ClientMessage::Authenticate` frame with its seat
and returned token before submitting actions. Every seat can independently be
human, open for an atomic invite claim, random, greedy, or a bounded whole-turn search agent.
An open room remains waiting with an empty legal mask until every guest has claimed
their seat; the browser invitation contains no credential. Network structures carry
`NETWORK_SCHEMA_VERSION`; every update includes the exact rules profile, scenario,
and deterministic state digest. The browser can create symmetric or procedural
rooms for two to eight invited humans across all seven built-in rules profiles.
`DELETE /v1/matches/{match_id}?seat=…` with the seat token in an Authorization
Bearer header closes and removes a room. All memory-amplifying limits are
explicit server configuration, and room capacity is checked atomically.
Every accepted transition is atomically persisted before it is acknowledged.
On restart, the service verifies each replay, reconstructs the authoritative
game, and replays native bot decisions to restore their exact RNG state. Only
token hashes are written to disk.
Terminal rooms enter the server's Elo ledger only after replay verification;
`GET /v1/league` returns its versioned standings and match ledger, and the
browser exposes both in a compact Server League panel. Full-range `u64` seeds
remain decimal strings across JSON while legacy numeric league files still
load. A one-click rated challenge fills the configured two-to-eight-player map
with authoritative search opponents, rotates the human seat between successful
attempts, and refreshes the verified Elo result in place. Room files
and `league.json` use fsync-plus-rename atomic replacement.
Match snapshots expose `NotFinished`, `Pending`, `Recorded`, or `Duplicate`
rating status; an identical replay cannot inflate Elo twice.
Multiplayer Elo computes all pairwise expectations from the pre-match ratings,
normalizes each pair by `players − 1`, and applies every delta simultaneously.
The update is zero-sum, preserves the ordinary two-player formula, and counts
one game per participant. The exact schema and request examples are documented
in [`docs/multiplayer.md`](docs/multiplayer.md).

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
verification. Fetch the current universal beta policy with:

```bash
python python/fetch_model.py
PYTHONPATH=python python python/evaluate.py \
  models/universal-routed-2to8p-engine-v6-2026-08-31.pt \
  --games 32 --baseline search --search-nodes 2048 \
  --profile online_experimental_v2_260801 \
  --width 11 --height 9 --action-limit 1000
```

The current routed bundle contains 41 immutable experts and selects them through
profile, generator, player-count, seat, and exact-domain routes covering two to
eight players. On a fresh engine-v6 fixed-duel gate it scored 336 wins and no
losses in 336 paired games against native search-2048, with every rules profile
tested from both seats and no action-limit adjudications. The tensor-identical
control bundle scored 312–24 on the same arena; three accepted seat-specific
search-distilled specialists added +685 relative Elo inside that pool. The
inherited 5–8-player procedural routes remain much weaker: their latest matrix
is 55 wins in 364 games against search-256, with 132 action-limit truncations.
Neither result is an absolute human Elo or a claim of universal superhuman play.
Exact results and training provenance live under
[`benchmarks/`](benchmarks/).

The first policy-guided PUCT loop is measured separately from that search-2048
pool. A frozen-policy value specialist trained on sparse plausible deviations
improved held-out value sign accuracy from 72.7% to 92.1%. At eight PUCT nodes
and a root value weight of one, it scored 280–232 against the tensor-identical
direct policy on 512 fresh paired games: an observed +32.67 relative Elo with no
truncations. This result applies only to the classic Generic 11×9 duel route.
Canonical PUCT with the original uncalibrated value head lost its 16-game scout
0–16 and was rejected. Full accepted and rejected evidence is in
[`benchmarks/2026-08-31-policy-guided-puct-amplification-rocm.json`](benchmarks/2026-08-31-policy-guided-puct-amplification-rocm.json).

The first completed PUCT amplify→distill loop exports the tree's full soft root
distribution instead of copying only its argmax. An action-head-only student
keeps the encoder, rules context, and calibrated value path bit-identical. The
accepted bundle routes that student only to seat 0 and preserves the frozen
source policy for seat 1. It scored 281–231 against the source on 512 additional
fresh paired games: +34.04 observed relative Elo, a 50.55–59.14% Wilson score
interval, and zero truncations. Seat 0 added 25 wins while the unchanged seat 1
reproduced its source result exactly. Four hard-target variants were rejected;
the complete record is in
[`benchmarks/2026-08-31-puct-soft-distillation-rocm.json`](benchmarks/2026-08-31-puct-soft-distillation-rocm.json).

The second loop targets only the previously weak seat 1 route and adds a strong
retention constraint. On a disjoint 512-map fixed-seat gate it won 470 games
against the unchanged source opponent; the source policy won only 48 games from
seat 1 on those same maps. That is a +0.824 score delta and +419.54 observed
relative Elo within this deliberately seat-specific pool, with a 89.10–93.87%
Wilson score interval and no truncations. The combined v2 bundle preserves the
accepted seat 0 specialist and routes this new student only to seat 1. It does
not remove the underlying first-move advantage: the seat 0 specialist still won
every direct cross-seat game in the aggregate no-regression gate. Full training,
rejected-candidate, holdout, routing, and browser-parity evidence is in
[`benchmarks/2026-08-31-puct-seat1-soft-distillation-rocm.json`](benchmarks/2026-08-31-puct-seat1-soft-distillation-rocm.json).

Reproduce the accepted direct-policy comparison with the small specialist:

```bash
python python/fetch_model.py classic-generic-duel-value-v3-2026-08-31
PYTHONPATH=python python python/evaluate.py \
  models/classic-generic-duel-value-v3-2026-08-31.pt \
  --games 64 --seed 700000 --device cuda --baseline policy \
  --model-agent puct --puct-nodes 8 --puct-root-value-weight 1 \
  --profile classic_generic_2022 --width 11 --height 9 --action-limit 1000
```

Reproduce the zero-search distilled-policy comparison:

```bash
python python/fetch_model.py classic-generic-duel-value-v3-2026-08-31
python python/fetch_model.py classic-generic-duel-puct-distilled-v2-2026-08-31
PYTHONPATH=python python python/evaluate.py \
  models/classic-generic-duel-puct-distilled-v2-2026-08-31.pt \
  --games 64 --seed 920000 --device cuda --baseline policy \
  --baseline-checkpoint models/classic-generic-duel-value-v3-2026-08-31.pt \
  --profile classic_generic_2022 --width 11 --height 9 --action-limit 1000
```

Play against the neural beta in a local browser (cyan moves first):

```bash
python python/play_policy.py --profile online_experimental_v2_260801
```

The launcher fetches and SHA-verifies the registry default when it is missing,
binds only to `127.0.0.1`, opens an interactive hex board, and runs every amber
reply through the checkpoint. Pass `--checkpoint`, `--seed`, `--device`, or
`--no-browser` to override the defaults.

Policy bundles remain one verified artifact while routing immutable rules
profiles to compatible experts. The evaluator and browser arena expose the
selected route; legal actions, state transitions, and observations still come
from the single authoritative Rust engine.

Export and verify the fixed browser route on a machine with the training extras:

```bash
PYTHONPATH=python python python/export_browser_policy.py \
  models/universal-routed-2to8p-engine-v6-2026-08-31.pt \
  web/public/browser-primary.onnx
PYTHONPATH=python python python/export_browser_policy.py \
  models/universal-routed-2to8p-engine-v6-2026-08-31.pt \
  web/public/browser-experimental-v2.onnx \
  --profile online_experimental_v2_260801
PYTHONPATH=python python python/export_browser_policy.py \
  models/classic-generic-duel-puct-distilled-v2-2026-08-31.pt \
  web/public/browser-classic-generic-puct-seat0-v2.onnx \
  --profile classic_generic_2022 --seat 0
PYTHONPATH=python python python/verify_browser_policy.py \
  models/classic-generic-duel-puct-distilled-v2-2026-08-31.pt \
  web/public/browser-classic-generic-puct-seat0-v2.onnx \
  --profile classic_generic_2022 --seat 0
PYTHONPATH=python python python/export_browser_policy.py \
  models/classic-generic-duel-puct-distilled-v2-2026-08-31.pt \
  web/public/browser-classic-generic-puct-seat1-v2.onnx \
  --profile classic_generic_2022 --seat 1
PYTHONPATH=python python python/verify_browser_policy.py \
  models/classic-generic-duel-puct-distilled-v2-2026-08-31.pt \
  web/public/browser-classic-generic-puct-seat1-v2.onnx \
  --profile classic_generic_2022 --seat 1
```

Run the verifier once per routed profile. It drives an entire game through both
PyTorch and ONNX, requires the same argmax action at every state, and reports
the maximum numeric error and observed legal-action-set range.

## Status

The engine is under active construction. The compatibility matrix in
[`docs/rules.md`](docs/rules.md) is the source of truth for implemented rules.

## Intellectual property

This is an unofficial implementation. Antiyoy is by Yiotro; this project is not
affiliated with or endorsed by Yiotro. The browser client uses a small set of
original Antiyoy game-piece images under Yiotro's non-commercial terms. Those
files are identified separately in `web/public/game-pieces/README.md` and are
not covered by this repository's MIT license.

The original code in this repository is licensed under MIT.
