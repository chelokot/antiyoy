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
configured compatibility profile. Pass `--replan-each-action` to measure
Markovian labels that discard the cached plan before every atomic action.

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

Counterfactual terminal runs pair each sampled learner game with a frozen,
greedy policy on the identical seed and report both records. Exact-seat scouts
may avoid replaying unchanged routes, but a tie still fails selection and any
actual gain must subsequently survive the all-seat release suite.

The 5–8 player matrix gives every seat one game on each of two held-out seeds
for every compatibility profile. Its increasing board sizes and action limits
are part of the domain identity. Use a search-node override only for explicitly
labelled scouting runs; release records retain the configured 2048-node budget.

Browser export records require action-level parity across a complete game, not
only a successful ONNX conversion. They name the routed expert, model and
runtime hashes, dynamic legal-action range, maximum numeric error, and a
WebAssembly-runtime smoke transition through the authoritative Rust engine.

Policy-guided PUCT records use the same checkpoint as both amplified agent and
direct-policy opponent. Node count, leaf batches, root visits, root selection,
and value-calibration provenance are part of the arena identity. A zero-weight
policy/value blend must reproduce direct policy exactly before any non-zero
weight is measured. Calibration games, tuning scouts, and accepted holdouts use
disjoint seed windows; failed value datasets and search budgets remain in the
record instead of being discarded after a successful run.

PUCT distillation records must compare a student checkpoint against the exact
frozen source checkpoint rather than using policy self-play as a proxy. Soft
root targets, trainable parameter scope, retention KL, routing, selection seeds,
and final held-out seeds are part of the protocol. A routed specialist passes
only when every unchanged seat reproduces its source result and the aggregate
gain survives a disjoint confirmation; rejected hard-target variants remain in
the same report.

Method comparisons reuse every map seed and model seat for candidate and
baseline self-play. Reports count improved, regressed, and unchanged map
outcomes and apply an exact two-sided sign test only to discordant maps. Suites
pool those raw counts before recomputing significance; per-window p-values are
never averaged. Aggregate Elo without this matched-map evidence is diagnostic,
not sufficient for promotion.

Replayable action-Q records bind every dataset to the complete source-model
hash and retain each sampled state's episode seed, seat, round, fingerprint,
and contiguous action prefix. Sampled prefixes must reproduce the exact Rust
state before training. Offline action-pair accuracy is diagnostic only: shared
and exact-seat heads still require fresh, paired full-game gates in every
collected map domain, and a cross-domain tie or regression is not promoted.

Full action-slate records preserve every legal alternative at each sampled
root, including source logits, PUCT probabilities, Q-values, and visit counts.
The replay ledger stores each episode once and fingerprints every sampled Rust
state. Split games rather than individual roots, leave unvisited alternatives
at the source policy, and require a paired outcome scout before spending fresh
confirmation seeds. Lower held-out listwise KL is diagnostic, not evidence of
strategic improvement.
