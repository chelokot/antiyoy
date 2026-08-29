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
checkpoint. Add `--fog` to train from the active player's exact
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
its teacher.

`--imitation-rollin teacher` follows the expert trajectory and is the default
behavioral-cloning curriculum. `--imitation-rollin policy` performs online
DAgger-style recovery training: search labels every current state, while the
policy's own greedy action advances the environment. This exposes compounding
errors and loops that are absent from clean teacher trajectories.

Measure target-generation cost before a large run:

```bash
RAYON_NUM_THREADS=8 python benchmark_teacher.py \
  --environments 64 --transitions 20000 --search-nodes 2048
```

Use `--resume CHECKPOINT --checkpoint NEW_CHECKPOINT` to continue a run with
the exact model and optimizer state while changing the profile schedule. The
trainer rejects incompatible checkpoint, observation, and rule-feature
versions before allocating a rollout.

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

Evaluation alternates model seats, uses held-out seeds, and reports the raw
win/draw/loss score plus an Elo difference against the named baseline. Terminal
draws and action-limit truncations are reported separately, along with action
kind histograms for both competitors, so a stalling policy cannot hide behind
an aggregate draw rate. It does not assign an absolute leaderboard rating to
an uncalibrated baseline.
