# Python environment

The `antiyoy_rl` package is a thin NumPy boundary over the authoritative Rust
batch environment. It releases the Python GIL during parallel stepping and
returns structure-of-arrays buffers instead of Python objects per hex or action.

## Development install

```bash
python -m venv .venv
.venv/bin/python -m pip install maturin==1.15.0 pytest numpy
cd python
../.venv/bin/maturin develop --release
../.venv/bin/pytest -q tests
```

## Minimal loop

```python
import numpy as np

from antiyoy_rl import ProceduralConfig, ScenarioObjective, VectorEnv

generator = ProceduralConfig(
    width=31,
    height=21,
    players=4,
    seed=1,
    land_density_per_million=650_000,
)
objective = ScenarioObjective.survive_through_round(player=0, round=100)
environment = VectorEnv.procedural(256, generator, objective=objective)
observation = environment.observe()
action_indices = np.zeros(environment.environments, dtype=np.uint64)
result = environment.step(action_indices)
```

Each action index is local to its environment's half-open range in
`observation["action_offsets"]`. Select a legal action, step the complete batch,
and reset every environment whose terminal or truncated value is one.

The default generator is replay-compatible schema 1. Pass
`ProceduralConfig(..., schema_version=2)` or train with
`train.py --procedural --generator-schema-version 2` to rotate which player ID
inherits each generated starting position across seeds. The training summary
records `procedural_v2`; it does not silently relabel old checkpoints or
eliminate asymmetry inside one map.

## Exact counterfactual branches

`VectorEnv.fork(np.array([source_index, ...], dtype=np.uint64))` copies live Rust
states without replaying their action histories. Indices may repeat or be
reordered. Each copy retains its rules, procedural generator, objective, fog
mode, legal actions, episode step, and action limit, then advances independently
of the source. This is useful for evaluating alternative legal actions from the
same position and for collecting state-action training targets.

`antiyoy_rl.counterfactual.rollout_candidates` applies each candidate action
to a fork and continues with the same frozen routed policy. With no horizon it
returns exact terminal or adjudicated results under that continuation. A finite
`horizon` returns territory at that many future atomic actions and explicitly
marks unfinished branches as censored; their winner is `None`, not a draw.
These deterministic continuations are conditional counterfactuals, not an
estimate of performance against arbitrary opponents.

From the repository root, `python -m python.benchmark_counterfactual CHECKPOINT`
samples policy positions and reports all candidate results. For example, add
`--generator procedural_v1 --players 5 --environments 4 --updates 320
--label-stride 8 --candidate-count 3 --training-seat 4
--rollout-horizon 96 --device cpu`. The report contains episode seeds and full
action prefixes so each sampled position can be replayed. To test whether
lookahead actually improves games, use
`python -m python.evaluate_counterfactual CHECKPOINT` with the same domain
options, `--environments 64 --model-seat 4 --candidate-count 3
--rollout-horizon 96 --intervention-round 0`. It changes one decision per game
and compares complete outcomes against the direct policy on identical seeds.
Only the paired outcome test, not territory at the horizon, is a strength gate.

## Policy training

`UniversalPolicy` is a rules-conditioned hex convolutional actor-critic. It
scores the current variable legal-action set from source, target, action, and
global board embeddings instead of allocating a mostly invalid fixed policy
head. Each occupied cell carries its owner's relative turn distance and
active-player relation. Global context includes the player count and logarithmic
round, so multiplayer opponents no longer collapse into one interchangeable
category. Diplomacy-enabled observations also add relation/proposal context and
player-targeted diplomatic actions. The trainer uses clipped PPO with generalized
advantage estimation. Both
bootstrapping and GAE flip perspective whenever the active player changes, and
consume raw reward components supplied by Rust.

```bash
python train.py --environments 64 --updates 1000 --device cuda \
  --procedural --width 31 --height 21 --players 4 \
  --profiles classic_generic_2022 classic_slay_2022 online_duel_v1 \
    online_experimental_v2_260801 \
  --imitation-updates 500 --imitation-teacher search \
  --imitation-rollin policy --search-nodes 2048 \
  --rollout-steps 16 --epochs 2 \
  --checkpoint ../models/universal-ppo.pt
```

The profile list is cycled across the vector batch and each environment keeps
its rules through resets. Procedural reset seeds rebuild the entire connected
map and starting position; the exact generator configs are stored in the
checkpoint. Use `--land-density-schedule-per-million 600000 650000 700000` to
cycle heterogeneous map densities across workers while keeping their assignments
stable through deterministic resets. `--players-schedule 5 6 7 8` applies the
same contract to player counts. `--map-size-schedule 19x15 21x15 23x17 25x17`
cycles board dimensions. Schedules advance by worker index, so equally sized
player and map schedules preserve intentional pairs such as five players on
19×15 through every deterministic reset. All domain schedules can be combined
in one batch.
Add `--fog` to train from the active player's exact
visibility projection; full-state mode is the faster default for centralized
self-play. Add `--diplomacy --initial-relation neutral` to expose bilateral
offers, declarations of war, alliance propagation, and their exact action mask.
Use `--objective-json` with a serialized `ScenarioObjective` to train campaign
curricula; terminal results expose whether the condition was actually satisfied.

Teacher distillation is an optional curriculum, not an evaluation shortcut.
`--imitation-teacher greedy` teaches a cheap tactical prior; `search` uses the
bounded whole-turn agent. Rust keeps one search agent per environment, reuses
the exact cached turn plan after each selected action, releases the GIL, and
computes independent environments through Rayon. Node, beam, branch, and turn
depth budgets are explicit CLI parameters and are stored in the checkpoint.
Held-out mirrored matches still determine whether the resulting model exceeds
its teacher. Evaluation requires an even game count, repeats every map seed for
both model seats, and emits both seat slices so first-player advantage cannot be
hidden by an aggregate score.

`--imitation-rollin teacher` follows the expert trajectory and is the default
behavioral-cloning curriculum. `--imitation-rollin policy` performs online
DAgger-style recovery training: search labels every current state, while the
policy's own greedy action advances the environment. This exposes compounding
errors and loops that are absent from clean teacher trajectories.
`--imitation-search-replan` discards the search agent's hidden cached turn plan
before labeling every state. The resulting label is a deterministic function of
the observation instead of prior teacher history, which is required when the
student has no recurrent plan state. It costs one complete search per atomic
action and is therefore intended for targeted distillation runs.
`--imitation-symmetry-augmentation` rotates alternating batch observations by
180 degrees while preserving each legal action index. It prevents the hex CNN
from specializing its spatial filters to one side of a symmetric arena.
`--imitation-reference-weight W` keeps a frozen copy of the initialized policy
and adds `W × KL(reference || candidate)` to the teacher loss. This provides an
explicit stability objective when fine-tuning weak profiles would otherwise
erase strong behavior from the published checkpoint.
Repeat `--imitation-slice-weight PROFILE:SEAT:WEIGHT` to emphasize a measured
worst-case slice in both teacher and retention losses. Unspecified slices keep
weight one; malformed, duplicate, unscheduled, or out-of-range slices fail
before the environment or model is allocated.
Repeat `--imitation-action-weight ACTION_KIND:WEIGHT` to counter a measured
teacher-label imbalance across `end_turn`, `move`, `recruit`, `build`,
`plant_tree`, and `diplomacy`. The action weight multiplies the profile-seat
weight and is normalized across the batch, so rare strategic actions can be
emphasized without changing environment sampling or the authoritative mask.
Repeat `--imitation-policy-rollin-slice PROFILE:SEAT` for asymmetric DAgger.
The selected seats advance with the candidate's action while every opponent
seat advances with the search label. This trains recovery against an exact
teacher opening without replacing the opponent with another copy of the policy.
Set `--updates 0` for an imitation-only run without a PPO phase; at least one
imitation update is then required.
Use `--imitation-reset-interval N` to reset every training environment onto a
fresh deterministic procedural seed after each `N` teacher updates. This
increases map diversity without increasing the transition budget; zero keeps
natural episode boundaries only. The checkpoint summary records the exact
number of environment resets.

For long-horizon best-response training, set `--fixed-opponent greedy` or
`--fixed-opponent search` with `--learner-seat SEAT`. Every batch worker runs a
complete game in which the policy controls only that seat and the named frozen
agent controls every opponent. The trainer retains the behavior log-probability
and value for each learner decision, then applies clipped PPO from the terminal
win, loss, or adjudicated result. This avoids treating action-level imitation
accuracy as strategic strength. `--opponent-minibatch` bounds replay memory per
optimizer step, and `--opponent-reference-weight W` adds a frozen-policy KL
anchor while fine-tuning. Each update consumes one complete episode per worker;
the checkpoint reports games, outcomes, environment transitions, and optimizer
steps separately. Held-out all-seat evaluation remains the promotion gate.

`--opponent-counterfactual-baseline` splits an even environment batch into
matched pairs. The first game samples the learner while the second runs the
frozen initialization policy greedily on the identical map against identical
opponents. PPO receives half the difference between their terminal outcomes,
so maps where both policies win or both lose contribute zero credit instead of
gradient noise. The checkpoint reports learner and frozen-baseline records
separately. This doubles simulation per learner episode and is intended for
sparse multiplayer outcomes.

```bash
python train.py --environments 24 --updates 8 --device cuda \
  --procedural --width 21 --height 15 --players 6 --action-limit 2800 \
  --fixed-opponent search --learner-seat 2 --search-nodes 256 \
  --opponent-counterfactual-baseline \
  --opponent-minibatch 256 --opponent-reference-weight 0.25 \
  --initialize ../models/generic-6p.pt \
  --checkpoint ../models/generic-6p-seat2-best-response.pt
```

Measure target-generation cost before a large run:

```bash
RAYON_NUM_THREADS=8 python benchmark_teacher.py \
  --environments 64 --transitions 20000 --search-nodes 2048
```

Use `--resume CHECKPOINT --checkpoint NEW_CHECKPOINT` to continue a run with
the exact model and optimizer state while changing the profile schedule. The
trainer rejects incompatible checkpoint, observation, and rule-feature
versions before allocating a rollout.

For interruptible shared-GPU jobs, `--checkpoint-every N` atomically replaces
one hidden recovery checkpoint beside the requested final path every `N`
imitation or PPO updates. Successful completion removes it, so checkpointing
cannot accumulate files. Recovery restores exact weights and optimizer state;
the resumed command intentionally starts a new environment batch and curriculum
segment.

Use `--initialize CHECKPOINT` for weights-only transfer from a compatible
checkpoint while creating a fresh optimizer. This accepts the published
checkpoint-v4/observation-v6 alpha through the same zero-column diplomacy
migration used by evaluation. `--initialize` and `--resume` are mutually
exclusive so a warm start cannot be mistaken for an exact continuation.

Evaluation also accepts the published checkpoint-v4/observation-v6 alpha. Its
weights are expanded with zeroed diplomacy columns so disabled-diplomacy
profiles preserve the legacy policy exactly; optimizer resume remains strict.
Missing multiplayer-context weights are likewise initialized to zero, preserving
published policy outputs exactly while allowing new runs to learn them.

This trainer is an executable baseline for validating the complete ROCm path;
checkpoint strength must be established by held-out mirrored tournaments before
publishing an Elo number.

```bash
python evaluate.py ../models/universal-ppo.pt --games 64 --baseline greedy \
  --profile online_duel_v1 --width 11 --height 9 --action-limit 1000
```

Use `--baseline search --search-nodes 2048` for the stronger deterministic
teacher. Search beam, branch, and maximum turn depth are independently
configurable and emitted in the result.

Use `--baseline reply_search` with `--search-nodes 256`,
`--reply-search-nodes 64`, and `--reply-slate-size 8` to test the opt-in
whole-turn opponent-response agent against a checkpoint. The reference is a
self-match of that same search agent;
evaluate disjoint procedural map seeds with both model seats. A positive
search-vs-search result alone does not establish that this search is a stronger
neural-policy teacher.

Once a search teacher has beaten the frozen source on a disjoint all-seat
arena, it can supply policy-rollin imitation labels without changing the
environment or the source checkpoint:

```bash
python train.py --updates 0 --imitation-updates 512 --environments 32 \
  --imitation-reset-interval 64 --imitation-teacher reply_search \
  --imitation-rollin policy --search-nodes 256 \
  --reply-search-nodes 64 --reply-slate-size 8 \
  --initialize ../models/routed-v6.pt \
  --initialize-profile classic_generic_2022 \
  --initialize-generator procedural_v1 --initialize-players 2 \
  --profile classic_generic_2022 --procedural \
  --generator-schema-version 2 --width 11 --height 9 --players 2 \
  --device cpu --checkpoint ../models/reply-distilled.pt
```

The teacher retains its exact planned turn while the student follows its own
actions; divergence invalidates the plan and triggers a new search. Imitation
accuracy is only a training diagnostic. A student becomes a stronger model
only after a fresh paired full-game comparison against the frozen source.

Policy-guided PUCT is a separate model-side amplifier. Rust owns the cloned
search trees, exact legal transitions, virtual visits, and deterministic
backup; Python batches pending leaves into a single policy/value inference.
The active routed expert supplies policy priors at every leaf. Value routing is
independently configurable because multiplayer search may need the root seat's
utility while an opponent supplies the legal-action policy.

```bash
python evaluate.py ../models/universal-routed.pt --games 64 \
  --baseline search --search-nodes 2048 --model-agent puct \
  --puct-nodes 64 --puct-leaf-batch-size 512 --device cuda
```

The result records decisions, evaluated leaves, inference batches, exact tree
nodes, root visits, and maximum reached depth. Compare it with the same seed,
seat, baseline, and model using `--model-agent policy`; changing only the PUCT
budget isolates the value of search from the value of the checkpoint.

For a two-player full-information search ablation, `--puct-value-source heuristic`
keeps the model's legal-action priors but replaces neural leaf values with the
native territory/economy position score transformed by
`tanh(score / --puct-heuristic-scale)`. The default scale is 2048. This mode
rejects fog and multiplayer because the native full-state scorer would either
leak hidden information or cease to be a two-player zero-sum value. It tests
the search algorithm independently from value-head calibration; it is not a
claim that PUCT is stronger.

Canonical PUCT chooses the most visited root action. For a distilled policy
whose value head is still being calibrated, use the continuous policy/value
root blend instead:

```bash
python evaluate.py ../models/universal-routed.pt --games 64 \
  --baseline policy --model-agent puct --puct-nodes 64 \
  --puct-root-value-weight 0.25 --device cuda
```

The score is `log(policy prior) + weight × searched mean value`. Weight zero is
an exact direct-policy endpoint; increasing it measures how much authority the
search value receives without conflating the result with a different model.

For the direct amplification match, use the checkpoint itself as the opponent:

```bash
python evaluate.py ../models/universal-routed.pt --games 64 \
  --baseline policy --model-agent puct --puct-nodes 64 --device cuda
```

Every opponent seat then uses the same routed checkpoint without search. The
paired seat rotation and policy-vs-policy reference run isolate the Elo added by
PUCT rather than mixing it with a different heuristic opponent.

The historical scalar backup is exact for two-player zero-sum games but only an
approximation with three or more players: each leaf value belongs to the player
whose turn it is. Use `--puct-value-perspective root` for a consistent
multiplayer paranoid search. Policy priors still come from the actual active
seat's routed expert, while a separate value pass re-encodes the position from
the root seat and routes its value expert. The tree then maximizes root utility
on root turns and assumes every opponent minimizes it. The perspective mode is
recorded and must match across comparisons.

`--puct-opponent-horizon leaf` instead searches only the root player's current
turn. A state reached after `EndTurn` is evaluated once from the root
perspective but opponent actions are not expanded. This matches the horizon of
the native whole-turn planner and avoids inventing a coalition of every other
player. The default `search` horizon preserves full adversarial tree expansion;
the horizon is emitted alongside every PUCT result.

`--puct-objective maxn` removes that coalition approximation while retaining
full opponent expansion. Every leaf is evaluated from each of its two-to-eight
player perspectives in one combined inference batch. Rust backs up the complete
utility vector and each node maximizes the component belonging to its actual
active player. The existing scalar value head and checkpoint format are reused;
the active player's already-computed value is not evaluated twice. `scalar`
remains the compatibility default, and the objective is recorded in evaluation
and distillation reports.

PUCT depends on a calibrated value head. Search-teacher imitation updates policy
logits but deliberately do not train values, so a newly distilled expert must
pass value calibration before its PUCT result is treated as amplification. The
calibrator labels held-out direct-policy self-play states from the active
player's perspective and freezes every policy and trunk parameter:

```bash
python calibrate_value.py ../models/universal-routed.pt \
  ../models/classic-generic-duel-value.pt --profile classic_generic_2022 \
  --games 128 --validation-games 32 --seed 600000 --device cuda \
  --exploration-probability 0.005 --exploration-top-k 4
```

The emitted specialist checkpoint includes before/after train and held-out
MAE, RMSE, sign accuracy, and correlation, plus source and output SHA-256
digests. Controlled exploration replaces 0.5% of actions with a sampled rank
2–4 policy action. This preserves mostly on-policy outcome labels while exposing
the value head to plausible off-policy successors; higher exploration rates
must pass a separate outcome gate. Procedural calibration accepts the same generator
contract as evaluation, including `--generator procedural_v1` or
`procedural_v2`, `--players 2..8`,
map dimensions, density controls, and action limit. It routes every active seat
through the source bundle. Add `--training-seat` to collect loss and held-out
metrics only for one player's states and emit that seat's exact routed expert;
without it, every calibrated seat must share one expert. Overlay the resulting
checkpoint only on that calibrated context, preserving
every inherited route:

For multiplayer value heads, `--target-mode zero_sum` assigns the winner `+1`
and each loser `-1 / (players - 1)`. The per-game target sum is therefore zero,
instead of making four of five targets `-1` in a five-player game. The legacy
`binary` target remains the default for exact reproduction; compare target modes
on disjoint value and outcome gates before routing either checkpoint.

`--loss-mode ranking` optimizes pairwise logistic separation between winning
and losing states in each batch, with a quarter-weight MSE term anchoring value
magnitude. This objective is intended for PUCT, where ordering successor values
can matter more than absolute calibration error. `mse` remains the default;
ranking checkpoints still require the same held-out value report and direct
outcome gate.

```bash
python build_bundle.py ../models/universal-routed.pt \
  ../models/universal-routed-value.pt --overlay \
  --context-route classic_generic_2022:symmetric_duel_v1:2=\
../models/classic-generic-duel-value.pt
```

For seat-rotated procedural duels, use `--generator procedural_v2 --players 2`.
When an existing checkpoint routes its procedural expert under `procedural_v1`,
also pass `--route-generator procedural_v1` to calibration and distillation,
and `--generator-schema-version 2 --route-generator procedural_v1` to
evaluation. The generated maps use schema 2 throughout; only expert selection
reuses the old route. Both the environment and route domain hashes are recorded
in the training reports.

Then repeat the direct `--baseline policy` arena on fresh seeds. Accept the
overlay only when held-out value metrics improve and PUCT beats the unchanged
direct checkpoint; otherwise the result remains a recorded rejected attempt.

Distill an accepted PUCT teacher into a zero-search policy with the calibrated
checkpoint frozen as both evaluator and retention reference:

```bash
python distill_puct.py ../models/classic-generic-duel-value.pt \
  ../models/classic-generic-duel-puct-distilled.pt \
  --environments 64 --updates 1000 --seed 800000 --device cuda \
  --puct-nodes 8 --puct-root-value-weight 1 --retention-weight 1
```

Only the student's action head is trainable. The board encoder, rules context,
and calibrated value path must remain bit-identical to the teacher or the run
fails. By default the imitation loss matches the complete soft root distribution
emitted by the Rust tree: for the blended root this is the normalized
`prior × exp(weight × searched mean value)` target. It preserves uncertainty and
the magnitude of small search corrections instead of turning every target into
a hard argmax. KL retention covers the complete batch. The checkpoint records
search cost, target KL, labeled examples, direct-policy disagreement,
imitation accuracy, KL retention, completed games, and all deterministic seeds.
Add `--training-seat 0` or `--training-seat 1` to optimize only positions where
that player is active. The teacher still routes every search leaf through the
source expert for its actual seat, while retention and imitation losses count
only the requested specialist states. The report separates all visited states,
actual labeled examples, and optimizer updates. Without this option, both seats
must resolve to one shared source expert so the legacy all-seat protocol remains
exactly reproducible.

The same command supports procedural multiplayer curricula. Select the exact
player-count route with `--generator procedural_v1 --players 5`, provide the
generator dimensions and densities, and use `--training-seat` for the specialist
being improved. Add both `--puct-value-perspective root` and
`--puct-opponent-horizon leaf` when the accepted teacher used root-perspective
whole-turn search. Procedural resets regenerate topology from every recorded
seed.
Teacher roll-in preserves each opponent's routed expert; student roll-in
substitutes the student only for its target seat. Two-to-eight-player runs thus
train and report one explicit route without silently replacing other seats.

For offline counterfactual work, capture every legal root action rather than
only the selected search move:

```bash
python collect_action_slate.py ../models/universal-routed-value.pt \
  ../datasets/action-slates.pt --generator procedural_v1 --players 5 \
  --environments 64 --updates 3200 --label-stride 8 --rollin student \
  --seed 2500000 --device cuda \
  --width 19 --height 15 --action-limit 2400 --puct-nodes 32

python train_action_slate.py ../models/universal-routed-value.pt \
  ../models/action-slate-seat-0.pt ../datasets/action-slates.pt \
  --device cuda --training-seat 0 --advantage-scale 0.5 \
  --visit-prior 4 --retention-weight 4 --action-residual-hidden 64
```

The versioned slate artifact stores exact action-head inputs, source logits,
search probabilities, mean Q-values, and visit counts. Visits distinguish an
unmeasured action from a measured action whose value is zero. State replay is
normalized by episode: one action log is stored per seed, while each labeled
state references an exact step and state fingerprint. Sampled states must
replay bit-for-bit through the Rust engine before the dataset is saved.
`--label-stride 8` advances the frozen policy through seven unlabelled steps
between PUCT roots, retaining every intervening action in the replay. It
spreads the same 400 labels per environment across 3,200 steps, including
later rounds and episode resets. Sparse labels require student roll-in; the
report records the stride, labelled updates, completed games, and sampled
episode-step coverage.

The conservative target starts from the source logits and adds only
visit-confidence-weighted, within-state centered Q advantages. Unvisited
actions retain their source score, and a full-slate KL term holds the learned
distribution close to the source on every state, including states with no
measured preference. Training and validation split by complete episode.
Offline KL or ranking accuracy never promotes a model; compose each trained
seat checkpoint into an exact route and run fresh paired full-game gates.

With `--action-residual-hidden`, training updates only a
zero-initialized scorer that combines source, target, and global features with
their pairwise products. The frozen source policy scores exactly the same
actions before optimization; older checkpoints still use their original head.

To diagnose a procedural-duel amplifier before distilling it, collect only
PUCT-versus-direct disagreements and replay both legal actions:

```bash
python collect_action_q.py ../models/procedural-duel-value-bundle.pt \
  ../datasets/duel-action-pairs.pt --generator procedural_v2 \
  --route-generator procedural_v1 --players 2 --width 11 --height 9 \
  --environments 16 --updates 256 --rollin student --puct-nodes 8 --device cpu
python audit_action_q.py ../datasets/duel-action-pairs.pt \
  ../models/procedural-duel-value-bundle.pt --max-examples 64 --device cpu
```

The collector stores exact local action indices, episode prefixes, and state
fingerprints. The audit forks each replayed state, applies both actions, and
reports terminal or adjudicated outcomes under the same frozen direct-policy
continuation. It reports censored branches separately if a horizon is set.
These conditional pair labels diagnose search-value ranking; they are not an
online strength rating and should not be promoted without fresh paired games.
Use `--single-disagreement` with `evaluate.py --baseline policy --model-agent puct`
to search until the first action that differs from the frozen direct policy,
then finish each game entirely with that direct policy. The result records how
many games received an intervention and the searched-root count. Compare this
diagnostic with full-game PUCT on disjoint matched maps to distinguish the
first search correction from errors introduced by repeated replanning. It is
not a separate trained agent or a substitute for an all-seat strength gate.

Compare the cheap student against the exact frozen source on disjoint seeds:

```bash
python evaluate.py ../models/classic-generic-duel-puct-distilled.pt \
  --baseline policy \
  --baseline-checkpoint ../models/classic-generic-duel-value.pt \
  --games 256 --seed 900000 --device cuda \
  --profile classic_generic_2022 --width 11 --height 9 --action-limit 1000
```

This separates a genuine distilled-policy gain from the online search gain and
from checkpoint drift. A candidate is promoted only after a disjoint paired
arena; imitation accuracy is diagnostic rather than a release criterion.

Use `--model-seat SEAT` for an exact-seat scout. Unlike the all-seat rotation,
every game then uses a distinct seed with the policy fixed to that seat, and
the evaluator calibrates the result against baseline self-play on the same
seed window. This reduces a six-player specialist scout to one sixth of the
games without evaluating unrelated routes. It is a selection tool, not a
release gate: a promoted bundle still requires the held-out all-seat suite.
The result includes `elo_delta`, measured against the equal-player win
expectation, and `baseline_adjusted_elo_delta`, the edge-corrected
log-odds improvement over the source policy at the same seat. Method comparisons
must use the latter or the raw `score_delta`; a weak starting seat can improve
while its equal-player-relative estimate remains negative. Neither number is a
pool-independent Elo rating.
`paired_method_comparison` additionally compares candidate and baseline outcome
on every identical seed and seat. `paired_map_comparison` first groups all seat
rotations of one procedural map, then reports improved, regressed, and unchanged
independent maps with an exact two-sided sign test. The map-clustered bootstrap
adds 95% intervals for `score_delta` and pool-relative Elo; overlapping zero is
not evidence of a successful amplification. Suite reports pool matched-map
counts across seed windows and recompute the sign test, including separate
totals for every seat.
Every result also preserves the ordered `game_seeds`, `model_seats`, and
candidate `winners` vectors. Evaluations with different search settings can
therefore be compared map by map when these schedules and the arena
configuration are identical.

Use `--procedural --generator-schema-version 2` to evaluate the opt-in
rotated-seat map domain. The generator name becomes `procedural_v2` and gets a
distinct domain digest; schema 1 remains the default and keeps its existing
routes. A bundle without a v2 context or domain route falls back to its
profile-level expert. To transfer an existing v1 exact-seat bundle without
changing its expert selection, add `--route-generator procedural_v1`; the
report records both the v2 environment domain and the distinct policy route
domain. Inspect `selected_experts` before comparing models.
`evaluate_suite.py` accepts the same two flags and records the environment
generator and policy route separately across its all-seat gate.
The same seed has the same land and capital locations under both versions, but
assigns those starting positions to different seats. The generator fairness
audit is in `benchmarks/2026-09-23-rotated-seat-generator-cpu.json`; it is not
an Elo or a model-strength measurement.

Arena dimensions and action limits otherwise inherit the training checkpoint.
Cross-checkpoint comparisons must pass the same explicit `--width`, `--height`,
and `--action-limit`; all three values are emitted in every result.

`evaluate_suite.py` gates the weakest `profile × seat` slice aggregated across
all requested seed windows and also writes the weakest individual seed-window
slice for diagnostics.
For every map window it also runs baseline self-play. The reported
`baseline_score` and `score_delta` calibrate each seat against the actual turn
order advantage instead of assuming that every seat wins exactly `1/N` games.
Use `--minimum-seat-score-delta` together with `--minimum-aggregate-score` as a
release gate so a one-sided specialization cannot pass on its average alone.

`build_bundle.py PRIMARY OUTPUT --route PROFILE=CHECKPOINT` creates one atomic
policy artifact with explicit deterministic profile routes. It verifies source
checkpoint, observation, rule-feature, hidden-width, and layer compatibility;
the bundle stores every source SHA-256 and evaluator output names the selected
expert. This isolates genuinely conflicting rulesets without duplicating game
state or legality code.

More specific `--context-route PROFILE:GENERATOR:PLAYERS=CHECKPOINT` and
`--seat-context-route PROFILE:GENERATOR:PLAYERS:SEAT=CHECKPOINT` selectors
override the profile route. Exact seat routes make multiplayer specialists
possible while the original two-player expert remains immutable. Bundle
versions 1 through 4 remain loadable.

`--domain-route PROFILE:GENERATOR:PLAYERS:SEAT:DOMAIN=CHECKPOINT` is the most
specific selector. `DOMAIN` is the SHA-256 key emitted beside the complete
seed-free `domain_descriptor` by every evaluation result. The descriptor binds
dimensions, player count, action limit, visibility/diplomacy mode, and every
procedural generator parameter, so an expert verified on density 700 cannot be
silently selected on density 650 or a different object distribution. Seeds are
deliberately excluded: held-out maps in one exact domain share a route.

Pass `--overlay` when `PRIMARY` is an existing routed bundle. The builder
replaces only the supplied selectors, verifies every new checkpoint against the
base architecture, deduplicates identical source hashes, and removes experts
that no route references. The output records the exact base bundle hash, making
iterative specialist acceptance reproducible without repeating a long route
manifest or retaining superseded weights.

Multiplayer evaluation deserializes a checkpoint once per profile/seed window
and instantiates each distinct routed expert once. Seat and domain selection no
longer reread the complete bundle for every player.

When refining an accepted expert, `train.py --initialize BUNDLE
--initialize-profile PROFILE --initialize-generator procedural_v1
--initialize-players 6` starts from the exact context route instead of the
bundle's profile fallback. Add `--initialize-seat` and `--initialize-domain`
for the more specific route levels.

Evaluation alternates model seats, uses held-out seeds, and reports the raw
win/draw/loss score plus an Elo difference against the named baseline. Terminal
draws and action-limit truncations are reported separately, along with action
kind histograms for both competitors, so a stalling policy cannot hide behind
an aggregate draw rate. The release matrix gates the truncation-rate delta over
baseline self-play, so rulesets that naturally adjudicate long stalemates are
not confused with a model that stalls more often. It does not assign an
absolute leaderboard rating to an uncalibrated baseline.

Run the regression suite across every versioned profile and multiple seed
windows with one command:

```bash
python evaluate_suite.py ../models/universal-routed-2to8p-engine-v6-2026-08-31.pt \
  --seeds 100000 130000 --games 32 --baseline search --search-nodes 2048 \
  --minimum-aggregate-score 0.5 --output ../runs/seed-sweep.json
```

The suite hashes the checkpoint, records the fixed arena and search budget,
keeps per-profile/per-seed action and timeout diagnostics, and exits nonzero
when the optional aggregate regression gate fails. Its JSON output is atomic.

`evaluate_matrix.py CHECKPOINT
../benchmarks/configs/universal-cross-domain-v1.json` applies the same
self-play-calibrated seat gate across symmetric and procedural map domains,
different sizes, densities, and player counts.

## Play against a checkpoint

After installing the training extra and native extension, launch the local
neural arena with:

```bash
python play_policy.py --profile online_experimental_v2_260801
```

The default beta is downloaded through `fetch_model.py` with exact size and
SHA-256 verification. The stdlib HTTP server binds to loopback, opens a browser,
renders every observable hex and legal command, rejects stale revisions, and
runs complete opponent turns under a lock. Use `--no-browser` for a remote SSH
forward or automated smoke test.
