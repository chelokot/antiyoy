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

The same runner accepts the complete `procedural_v1` map configuration. For
example, a four-player connected-map workload is:

```bash
RAYON_NUM_THREADS=8 target/release/antiyoy rl-bench \
  --map procedural --width 31 --height 21 --players 4 \
  --land-density-per-million 650000 \
  --starting-province-size 5 --starting-money 10 \
  --tree-density-per-million 150000 \
  --neutral-tower-density-per-million 20000 \
  --neutral-capital-density-per-million 10000 \
  --grave-density-per-million 15000 \
  --environments 256 --transitions 1000000 --json
```

`setup_seconds` isolates batch and map generation from steady-state transition
throughput. Every result records generator name, dimensions, player count, and
exact playable hex count; do not compare records with different values.

Throughput comparisons are meaningful only when the commit, generator config,
environment count, observation version, thread budget, and action-selection
stream match. The checksum detects transition or selection drift; it is not a
cryptographic state digest.

Training smoke tests are also checked in as metadata-only records. They include
the model hash and held-out results but never commit checkpoints. A smoke test
validates the learning path; it is not a release candidate or a calibrated
leaderboard entry.

Universal curriculum records report each compatibility profile separately.
Aggregate training loss is never used to select a checkpoint: held-out mirrored
games against a named baseline are authoritative, including regressions and
profiles on which a candidate remains weak.

Policy evaluation schema v2 uses adjacent games with the same map seed and
opposite model seats. It reports each seat separately as well as the aggregate.
Earlier schema-v1 policy records alternated seats across different seeds; they
are preserved as historical measurements but are not paired ratings.

Search-teacher records separate target generation from authoritative stepping
and reset time. Reproduce them with `python/benchmark_teacher.py`; the reported
throughput includes cached whole-turn plans, legal-index projection, and every
configured compatibility profile.

Cross-domain routed-bundle records use the checked-in matrix configuration and
name every held-out selection window. Exact-domain specialists are accepted or
rejected per profile and seat before the complete matrix is rerun. The record
keeps unsuccessful candidates as ablation evidence, reports unchanged domains,
and preserves failed release gates instead of presenting an aggregate gain as
a universal policy improvement.

Roll-in ablations use a small selection window only to choose which candidate
earns a larger confirmation run. A scout improvement is never a promotion gate:
the candidate must preserve the gain across fresh seeds and every seat. Records
retain higher-accuracy checkpoints that fail this outcome test so imitation
metrics cannot be mistaken for strategic strength.

Fixed-opponent terminal-credit ablations route the candidate only into the
trained profile, generator, player count, seat, and exact domain digest. The
control and overlay must use identical seeds; outcomes for every unchanged seat
must match exactly. A candidate that loses the target-seat scout is rejected
without a confirmation run, regardless of its on-policy training win rate.

The 5–8 player matrix gives every seat one game on each of two held-out seeds
for every compatibility profile. Its increasing board sizes and action limits
are part of the domain identity. Use a search-node override only for explicitly
labelled scouting runs; release records retain the configured 2048-node budget.
