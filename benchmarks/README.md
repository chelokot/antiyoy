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

Audit procedural starting positions before interpreting multiplayer win rates:

```bash
cargo run --release -p antiyoy-cli -- seat-audit \
  --map procedural --width 19 --height 15 --players 5 \
  --maps 64 --seed 4500000 --action-limit 2400 --rotate-starts --json
```

The same deterministic greedy agent controls every seat. Each map is replayed
once per cyclic relabelling of the starting provinces. `wins_by_seat` therefore
measures the remaining seat effect, including turn order and first-round economy
balancing, while `wins_by_original_start` holds the map geometry fixed and
measures starting-position differences.
Truncated games are adjudicated and counted separately. This is a fairness
diagnostic, not an Elo estimate. For the initial Voronoi-region sizes on the
unrotated maps, run `python -m python.benchmark_seat_balance`; its per-seed
records can be compared with the Rust audit by using the same generator inputs.

Generator schema 2 is an opt-in experiment (`--generator-schema-version 2` in
`seat-audit`, `rl-bench`, and `benchmark_seat_balance`). For a given seed, it
preserves schema-1 land, capital locations, and neutral objects, then cyclically
assigns the generated starting positions to player IDs by `seed % players`.
This can remove the persistent link between a strong or weak generated
position and one seat over a seed window; it does not make each individual map
symmetric. Schema 1 remains the default, so existing replays and model domains
do not silently change. Compare both versions on fresh, identical seed
windows before using schema 2 for training or a rated queue.

Compare native agents on the same procedural maps with every candidate seat:

```bash
cargo run --release -p antiyoy-cli -- multi-compare \
  --map procedural --generator-schema-version 2 \
  --width 19 --height 15 --players 5 \
  --maps 16 --seed 5000000 --action-limit 2400 \
  --candidate search --baseline greedy --search-nodes 256 --json
```

Each seed first receives one all-baseline reference game. The candidate then
replaces each seat in turn while the other seats retain the same baseline.
The report retains seed/seat outcomes, paired better/worse/same counts,
truncations, and per-seat wins. Reference wins are replicated across seats for
the paired comparison, but the baseline game is simulated only once per seed.
These correlated games are a method comparison, not independent Elo matches.

Diagnose whether the first whole-turn search decision on a search-controlled
trajectory deserves its static score improvement:

```bash
cargo run --release -p antiyoy-cli -- turn-credit \
  --map procedural --generator-schema-version 2 \
  --width 19 --height 15 --players 5 \
  --maps 16 --seed 5200300 --rollin-seat 0 --target-round 100 \
  --search-nodes 256 --rollout-limit 2400 --json
```

The selected seat uses search until its first completed turn that differs from
the greedy alternative; other seats use greedy. From that identical pre-turn
state, both complete turns are continued with the same greedy policy to a
terminal winner. `positions` may be smaller than `maps` because some search
trajectories never diverge before the target round. `truncated` continuations
are adjudications at the action limit, not observed terminal wins. This
counterfactual diagnoses local turn credit under a fixed continuation policy;
it is neither an independent game evaluation nor a proof that choosing the
same turn repeatedly improves the search agent. Better/worse/same refer to the
sampled seat's win/draw/loss outcome, so a different opponent winning can
still count as same.
Each branch also records `reply_score`, the unchanged static evaluator's score
when the searched seat next gets a turn, after greedy opponent replies. It is
absent if the game ends or that seat is eliminated first.

Pass `--alternative-search-nodes 64,1024` to compare additional complete-turn
plans from the same sampled state. Exact duplicate end states share one greedy
continuation, while every alternative keeps its own action sequence and search
budget. `distinct_end_states` reports how much diversity the slate actually
contains. This is a label-coverage probe, not an agent that can see the future.

An offline minimum-score-gain gate can be audited from these records by keeping
only search turns whose `search.static_score - greedy.static_score` reaches a
prechosen margin. Choose the margin on one seed window and validate it on a
fresh window; group correlated seat results by map. This does not replay the
trajectory of an actual gated agent, so a positive result would still require
complete matched games before promotion.

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

Action-level counterfactual scouts fork the exact live Rust state, change one
legal action, and let the frozen policy play the remainder. Horizon-limited
territory is diagnostic only: incomplete branches are censored, and their
unknown winner is never recorded as a draw. A candidate decision rule must be
compared against the direct policy in complete, paired games on fresh seeds
before it can be called an amplifier.

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

Phase-spread slate pilots decouple rollout steps from expensive PUCT labels.
They retain intervening actions in the exact replay, report completed games
and sampled episode-step coverage, and compare search budgets on identical
policy rollouts. A change in search-decision frequency is a teacher diagnostic,
not evidence of stronger play.

Multiplayer value-calibration scouts report game-disjoint prediction metrics
and fresh matched-map outcomes separately. A lower holdout error or a positive
finite-sample Elo estimate does not promote the value head when only a few
paired maps change result; the full discordant count and exact sign test remain
visible.
