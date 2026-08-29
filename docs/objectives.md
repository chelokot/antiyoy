# Scenario objectives

Objective schema 1 supplies campaign and curriculum termination independently
of immutable rules and replayed board state. The default is `Domination`, which
matches normal engine victory. Additional conditions are:

| Condition | Completion |
| --- | --- |
| `DiplomaticVictory` | The player owns a province and every surviving owner is allied with them. |
| `SurviveThroughRound` | The player remains alive after the configured round, or wins earlier. |
| `DestroyPlayer` | The acting beneficiary remains alive after the target loses all provinces. |
| `ReachEconomy` | The player's selected aggregate metric reaches the signed threshold. |
| `EnsurePlayerVictory` | The core game ends with the configured player as winner. |

`ReachEconomy` distinguishes gross income, net profit, and treasury. Values are
summed across all connected provinces without normalization or clamping.
Colored singleton cells are not provinces and therefore do not keep a player
alive, matching normal Antiyoy victory semantics.

Evaluation returns `Active` or `Complete { satisfied, winner }`. `satisfied`
describes the campaign condition; `winner` records the actual game or objective
winner. A failed campaign can therefore retain the opponent who won instead of
collapsing all failures into an ambiguous draw.

Objectives do not alter `Game` serialization or legacy replay digests. Rust RL
batches store one immutable objective per environment and can finish an episode
before core domination. Python exposes typed `ScenarioObjective` factories and
accepts an objective in symmetric, procedural, and mixed-profile constructors.
The trainer's `--objective-json` value and every resulting checkpoint preserve
the exact versioned configuration.
