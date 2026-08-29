# Rules compatibility

This document records observed behavior separately from configurable extensions.
Each checked item must have a focused engine test.

## Classic generic profile

- [x] Connected same-player hexes form an independently funded province.
- [x] A province needs at least two hexes and receives one capital.
- [x] Base prices: unit level `10 × level`, farm `12 + 2 × existing farms`,
      tower `15`, strong tower `35`, planted tree `10`.
- [x] Clear hex income is `1`; a farm hex earns `5`; trees earn `0`.
- [x] Unit upkeep is `2, 6, 18, 36`; tower upkeep is `1, 6`.
- [x] Unit levels `1..4` merge additively up to level four.
- [x] Units move at most four friendly hexes and may enter one attack hex.
- [x] Attack strength must exceed adjacent/self defense; level four overrides
      this comparison in generic rules.
- [x] Capital/tower/strong tower defense is `1/2/3` on self and neighbors.
- [x] Captures split and merge provinces deterministically; money follows the
      largest surviving fragment and sums when friendly provinces merge.
- [x] Negative money becomes zero and all units in that province starve.
- [x] Isolated units starve at the start of their faction's turn.
- [x] Trees expand once per round before player zero starts.

## Classic slay profile

- [x] Trees are more aggressive, farms and strong towers are disabled, basic
      towers have no upkeep, every clear non-tree hex earns one, and unit-four
      upkeep is `54`.
- [x] Strength must strictly exceed defense for every unit level.

## Online compatibility profiles

| Profile | Income clear/farm | Farm price | Unit-four upkeep | New unit ready | Foreign recruit zone |
| --- | ---: | ---: | ---: | --- | --- |
| `online_default_v1` | `1 / 5` | `12 + 2F` | `36` | Owned empty hex | Province boundary |
| `online_classic_v1` | `1 / disabled` | disabled | `54` | Owned empty hex | Province boundary |
| `online_duel_v1` | `1 / 5` | `12 + 2F` | `36` | Never | Adjacent to own capital/farm |
| `online_experimental_v1` | `1 / 5` | `12 + 2F` | `36` | Never | Adjacent to own capital/farm |
| `online_experimental_v2_260801` | `0 / 7` | `8 + 2F` | `36` | Never | Adjacent to own capital/farm |

`F` is the number of farms already present in the connected province. The
numeric profiles, knight override, construction readiness, and Duel placement
restriction are implemented and regression-tested. Online province split
selection, turn timing, vegetation, diplomacy, and fog remain unchecked until
their differential fixtures are committed.

## Extensions

Diplomacy, fog of war, custom objectives, and league metadata are versioned
optional modules. Simultaneous vectorized stepping is implemented around the
same sequential core. Custom profiles may change every numeric rule.
