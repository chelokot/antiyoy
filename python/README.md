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

from antiyoy_rl import VectorEnv

environment = VectorEnv(256, width=11, height=9, seed=1)
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
head. The trainer uses clipped PPO with generalized advantage estimation. Both
bootstrapping and GAE flip perspective whenever the active player changes, and
consume raw reward components supplied by Rust.

```bash
python train.py --environments 64 --updates 1000 --device cuda \
  --profiles classic_generic_2022 classic_slay_2022 online_duel_v1 \
    online_experimental_v2_260801 \
  --imitation-updates 500 --rollout-steps 16 --epochs 2 \
  --checkpoint ../models/universal-ppo.pt
```

The profile list is cycled across the vector batch and each environment keeps
its rules through resets. Add `--fog` to train from the active player's exact
visibility projection; full-state mode is the faster default for centralized
self-play.

Greedy distillation is an optional curriculum, not an evaluation shortcut. It
teaches a common tactical prior from the native baseline before PPO self-play;
held-out mirrored matches still determine whether the resulting model actually
matches or exceeds that baseline.

Use `--resume CHECKPOINT --checkpoint NEW_CHECKPOINT` to continue a run with
the exact model and optimizer state while changing the profile schedule. The
trainer rejects incompatible checkpoint, observation, and rule-feature
versions before allocating a rollout.

This trainer is an executable baseline for validating the complete ROCm path;
checkpoint strength must be established by held-out mirrored tournaments before
publishing an Elo number.

```bash
python evaluate.py ../models/universal-ppo.pt --games 64 --baseline greedy \
  --profile online_duel_v1
```

Evaluation alternates model seats, uses held-out seeds, and reports the raw
win/draw/loss score plus an Elo difference against the named baseline. It does
not assign an absolute leaderboard rating to an uncalibrated baseline.
