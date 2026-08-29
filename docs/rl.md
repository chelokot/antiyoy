# Reinforcement-learning contract

`antiyoy-rl` is the allocation-reusing bridge between the correctness-first
engine and policy training. A batch may contain different map dimensions,
province counts, and immutable rules profiles. Independent environments are
stepped in parallel while results retain input order.

## Observation version 7

`BatchObservation` is a flat structure of arrays. `cell_offsets`,
`province_offsets`, and `action_offsets` have `environment_count + 1` entries;
the half-open range at indices `i..i+1` selects one environment without copying.
The batch carries one complete immutable `Rules` value per environment,
including every economic, combat, vegetation, and lifecycle parameter seen by
a universal policy.

Version 7 allows Classic, Slay, Online, Duel, Experimental, and custom
configurations in one batch. It also conditions on the selected full-state or
fog visibility mask and adds the complete directed diplomacy matrix. Checkpoints
store the observation version, feature width, and exact training rules. The
evaluator migrates the published version-6 alpha weights without changing their
outputs when diplomacy is disabled; trainer resume requires an exact contract.

Per-environment arrays contain width, height, active player, and round. Per-cell
arrays contain the playable mask, absolute owner (`255` is neutral), object
code, unit strength, readiness, effective defense, active-player visibility,
and province ID (`65535` means no province). Full-state mode marks every
playable cell visible and avoids fog traversal. Fog mode emits the exact active
player mask for decentralized or player-equivalent observations while the
authoritative arrays remain available for centralized critics.
Province arrays contain owner, exact signed treasury and profit, capital hex, and
size. The representation is lossless for authoritative game state used by a
policy and does not normalize or clamp configurable economic values.
`relation_offsets` partitions square per-environment matrices; an environment
with diplomacy disabled has an empty range and incurs no matrix copy. `relations` uses
war/neutral/friend/alliance codes `0..3`; directed `proposals` uses the same
codes and `255` for no pending offer. `player_counts` defines each matrix side.

Object codes are stable within this observation version:

| Code | Object |
| ---: | --- |
| 0 | Empty |
| 1 | Capital |
| 2 | Farm |
| 3 | Tower |
| 4 | Strong tower |
| 5 | Pine |
| 6 | Palm |
| 7 | Grave |

## Variable legal-action space

Policies score only legal actions produced by the core. Each action feature has
a kind, source, target, and one parameter. Missing hexes use `65535`.

| Kind | Source | Target | Parameter |
| --- | --- | --- | --- |
| End turn | missing | missing | 0 |
| Move | unit hex | destination | 0 |
| Recruit | province capital | destination | strength 1–4 |
| Build | missing | destination | farm 0, tower 1, strong tower 2 |
| Plant tree | missing | destination | 0 |
| Diplomacy | missing | target player ID | declare 0, propose neutral 1, friend 2, alliance 3, accept 4, reject 5 |

An action chosen at offset `action_offsets[i] + k` is stepped as local action
index `k`. This removes a huge mostly-invalid global action tensor and lets one
model operate across map sizes.

## Rewards and episode boundaries

Every step returns raw components from the acting player's perspective:

- terminal outcome in `{-1, 0, 1}`;
- owned-territory delta;
- exact treasury delta summed over that player's provinces;
- total unit-strength delta.

`objective_satisfied` distinguishes meeting a configured campaign condition
from merely reaching a terminal failure. `winner` remains the actual player who
won the game or objective. A versioned scenario objective may end an episode
before core domination; see [`objectives.md`](objectives.md).

The trainer owns reward weights so experiments cannot silently change engine
semantics. `terminal` reports completion of the configured objective, with core
domination as the default. `truncated` reports only the configured action limit.
A done environment rejects further actions until an explicit deterministic
reset.

The bundled PPO trainer collects multi-step rollouts and computes GAE in the
acting player's frame. A transition that hands control to another player flips
both the bootstrapped value and recursive advantage sign. Its checkpoint
contract records the observation version, rule feature width, training rules,
optimizer, and every hyperparameter needed to reproduce the run.

## Determinism

Parallel stepping is deterministic because games share no mutable state, output
order is indexed, and all engine randomness lives inside each game. Tests run
equal seeds and equal action indices through parallel environments and compare
the complete resulting `Game` values after every transition.

## Procedural domain randomization

`GeneratorConfig` schema 1 defines the original `procedural_v1` generator. It
controls dimensions, player count, exact land density, connected starting
province size and treasury, plus categorical densities for trees, neutral
towers, neutral capitals, and graves. The generator grows one connected land
mass, chooses farthest-point seeds, partitions it with a deterministic graph
Voronoi pass, and assigns every player one connected start. A fixed config and
seed produce the same complete `Scenario` on every target.

This generator is designed for balanced RL curricula; it is not claimed to
reproduce a Classic or Online map seed. Compatibility rules and map-generation
identity are separately versioned. Replays store the fully generated scenario,
so verification never depends on regenerating a map.

Procedural `BatchEnv` instances retain one config per environment. Consecutive
environments start at consecutive seeds, and `reset_with_seed` regenerates the
topology, objects, starts, and treasuries atomically. Python exposes the same
contract through `ProceduralConfig`, `VectorEnv.procedural`, and
`VectorEnv.procedural_mixed`. `generator_jsons()` records the exact per-worker
configs in checkpoints.

## Search teacher

`VectorEnv.search_actions()` returns one legal local action index per
environment. Each worker owns a persistent deterministic whole-turn agent, so
executing its selected action reuses the remaining plan on the next call when
the state still matches exactly. Configuration changes rebuild all teacher
agents atomically. Search runs without the GIL and parallelizes over independent
environments with Rayon; node, beam, branch, and maximum turn-action limits are
caller controlled.

Evaluation workers may pass an `active_mask` NumPy `uint8` vector. Masked
arenas return action index zero without running search, removing the long-tail
cost after their result has already been recorded.
