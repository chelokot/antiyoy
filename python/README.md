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

## Policy training

`UniversalPolicy` is a rules-conditioned hex convolutional actor-critic. It
scores the current variable legal-action set from source, target, action, and
global board embeddings instead of allocating a mostly invalid fixed policy
head. Diplomacy-enabled observations add relation/proposal context and
player-targeted diplomatic actions. The trainer uses clipped PPO with
generalized advantage estimation. Both
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

```bash
python train.py --environments 24 --updates 8 --device cuda \
  --procedural --width 21 --height 15 --players 6 --action-limit 2800 \
  --fixed-opponent search --learner-seat 2 --search-nodes 256 \
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
python evaluate_suite.py ../models/universal-routed-search-dagger-2026-08-30.pt \
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
