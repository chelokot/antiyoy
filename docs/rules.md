# Rules compatibility

This document records observed behavior separately from configurable extensions.
Each checked item must have a focused engine test.

## Classic generic profile

- [ ] Connected same-player hexes form an independently funded province.
- [ ] A province needs at least two hexes and receives one capital.
- [ ] Base prices: unit level `10 × level`, farm `12 + 2 × existing farms`,
      tower `15`, strong tower `35`, planted tree `10`.
- [ ] Clear hex income is `1`; a farm hex earns `5`; trees earn `0`.
- [ ] Unit upkeep is `2, 6, 18, 36`; tower upkeep is `1, 6`.
- [ ] Unit levels `1..4` merge additively up to level four.
- [ ] Units move at most four friendly hexes and may enter one attack hex.
- [ ] Attack strength must exceed adjacent/self defense; level four overrides
      this comparison in generic rules.
- [ ] Capital/tower/strong tower defense is `1/2/3` on self and neighbors.
- [ ] Captures split and merge provinces deterministically; money follows the
      largest surviving fragment and sums when friendly provinces merge.
- [ ] Negative money becomes zero and all units in that province starve.
- [ ] Isolated units starve at the start of their faction's turn.
- [ ] Trees expand once per round before player zero starts.

## Classic slay profile

- [ ] Trees are more aggressive, farms and strong towers are disabled, basic
      towers have no upkeep, every clear non-tree hex earns one, and unit-four
      upkeep is `54`.
- [ ] Strength must strictly exceed defense for every unit level.

## Extensions

Diplomacy, fog of war, custom objectives, simultaneous/vectorized stepping, and
league metadata are versioned optional modules. Compatibility profiles keep
their default values aligned with the original game while custom profiles may
change every numeric rule.
