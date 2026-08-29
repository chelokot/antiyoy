# Reinforcement-learning contract

`antiyoy-rl` is the allocation-reusing bridge between the correctness-first
engine and policy training. A batch may contain different map dimensions and
province counts under one immutable rules profile. Independent environments are
stepped in parallel while results retain input order.

## Observation version 4

`BatchObservation` is a flat structure of arrays. `cell_offsets`,
`province_offsets`, and `action_offsets` have `environment_count + 1` entries;
the half-open range at indices `i..i+1` selects one environment without copying.
The batch also carries the complete immutable `Rules` value, including every
economic, combat, and vegetation parameter seen by a universal policy.

Version 4 conditions on construction readiness, Duel foreign placement,
province/turn lifecycle, and the Classic versus Online vegetation algorithm.
Checkpoints store the observation version, feature width, and exact serialized
rules; evaluation rejects a model whose contract does not match the native
engine.

Per-environment arrays contain width, height, active player, and round. Per-cell
arrays contain the playable mask, absolute owner (`255` is neutral), object
code, unit strength, readiness, effective defense, and province ID (`65535`
means no province).
Province arrays contain owner, exact signed treasury and profit, capital hex, and
size. The representation is lossless for authoritative game state used by a
policy and does not normalize or clamp configurable economic values.

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

An action chosen at offset `action_offsets[i] + k` is stepped as local action
index `k`. This removes a huge mostly-invalid global action tensor and lets one
model operate across map sizes.

## Rewards and episode boundaries

Every step returns raw components from the acting player's perspective:

- terminal outcome in `{-1, 0, 1}`;
- owned-territory delta;
- exact treasury delta summed over that player's provinces;
- total unit-strength delta.

The trainer owns reward weights so experiments cannot silently change engine
semantics. `terminal` reports a rules victory. `truncated` reports only the
configured action limit. A done environment rejects further actions until an
explicit deterministic reset.

## Determinism

Parallel stepping is deterministic because games share no mutable state, output
order is indexed, and all engine randomness lives inside each game. Tests run
equal seeds and equal action indices through parallel environments and compare
the complete resulting `Game` values after every transition.
