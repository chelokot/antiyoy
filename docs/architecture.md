# Architecture

The simulation core owns all authoritative state. Other packages may serialize,
render, batch, or transport that state, but must never reproduce rule logic.

## Determinism contract

Given a rules hash, map, initial seed, and action stream, every supported target
must produce the same state hashes. Randomness is explicit state. Iteration with
semantic consequences has a stable order. Replays store intent, not snapshots,
and periodic hashes detect divergence.

## Performance contract

The hot state is contiguous and index-addressed. Hex adjacency is precomputed.
Legality can be emitted as a dense action mask without allocation after warmup.
The scalar environment is the correctness oracle; batching and search reuse it
until profiling proves a specialized representation is necessary.

Legal actions preserve their canonical order while using one bounded traversal
per ready unit. Vector environments step independent games in parallel and emit
one flattened structure-of-arrays observation with offset tables for variable
maps, provinces, and legal-action counts.

## Boundaries

- `antiyoy-core` has no wall clock, filesystem, sockets, or UI dependencies.
- `antiyoy-protocol` versions every persisted and network-visible structure.
- Bindings expose owned buffers and bulk operations instead of per-hex calls.
- Training artifacts are content-addressed and never committed to Git.
- Multiplayer rooms authenticate human seats, serialize actions with an
  optimistic revision, and broadcast the same `GameView` consumed by WebAssembly.
- Each authoritative room has an independent lock; unrelated matches continue
  while a bot computes a move or a replay snapshot is materialized.
- Persistent rooms store a versioned request, replay, bot-controller position,
  and hashes of human credentials. Restoration verifies replay hashes and
  replays bot choices before accepting new actions.
- Terminal rooms are admitted to the atomic league ledger only through its
  replay verifier; pending and duplicate rating outcomes remain explicit in
  network snapshots.
