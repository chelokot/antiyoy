# Benchmarks

Checked-in results are immutable measurements tied to an exact commit, command,
machine, thread budget, and deterministic checksum. The `rl-bench` workload
includes transitions, canonical legal-action regeneration, raw reward
components, resets, and a complete observation after each vector step. It does
not include replay hashing, model inference, or accelerator transfer.

Run the same workload with:

```bash
cargo build --release -p antiyoy-cli
RAYON_NUM_THREADS=8 target/release/antiyoy rl-bench \
  --environments 256 --transitions 1000000 --json
```

Throughput comparisons are meaningful only when the commit, environment count,
map dimensions, observation version, thread budget, and action-selection stream
match. The checksum detects transition or selection drift; it is not a
cryptographic state digest.

Training smoke tests are also checked in as metadata-only records. They include
the model hash and held-out results but never commit checkpoints. A smoke test
validates the learning path; it is not a release candidate or a calibrated
leaderboard entry.
