# Authoritative multiplayer

Network schema 4 supports two to eight independently controlled seats. A room
uses either a symmetric duel or the exact `GeneratorConfig` consumed by the
core. The server rejects mismatched seat/player counts, duplicate names, maps
above its configured cell limit, and action limits above its configured cap
before allocating a room.

Create a four-player procedural room with `POST /v1/matches`:

```json
{
  "schema_version": 4,
  "rules_profile": "OnlineDefaultV1",
  "scenario": {
    "Procedural": {
      "schema_version": 1,
      "width": 17,
      "height": 13,
      "players": 4,
      "seed": 700,
      "land_density_per_million": 650000,
      "starting_province_size": 5,
      "starting_money": 10,
      "tree_density_per_million": 150000,
      "neutral_tower_density_per_million": 20000,
      "neutral_capital_density_per_million": 10000,
      "grave_density_per_million": 15000
    }
  },
  "seats": [
    { "name": "andrii", "kind": "Human" },
    { "name": "search-a", "kind": "Search" },
    { "name": "greedy-a", "kind": "Greedy" },
    { "name": "random-a", "kind": "Random" }
  ],
  "action_limit": 2000
}
```

The response includes a token only for each human seat. Tokens are shown once;
only their BLAKE3 hashes are persisted. A snapshot includes the scenario, all
public seat descriptors, the current revision, rating state, deterministic
digest, complete game view, and authoritative legal actions.

Connect to `GET /v1/matches/{match_id}/watch` as a spectator. Authenticate a
human seat before submitting an action:

```json
{
  "Authenticate": {
    "schema_version": 4,
    "seat": 0,
    "token": "returned-seat-token"
  }
}
```

Submit an action with the last observed revision:

```json
{
  "Submit": {
    "schema_version": 4,
    "revision": 12,
    "action": "EndTurn"
  }
}
```

The server serializes the accepted human action, advances consecutive bot seats
until another human owns the turn or the match ends, atomically persists the
room, then broadcasts one snapshot. A restart verifies the replay and replays
every deterministic bot decision to restore search plans and random streams.

## Multiplayer Elo

Only replay-verified terminal or adjudicated matches enter the league. For `N`
participants, every unordered pair receives the ordinary Elo expected score
from ratings captured before the match. The winner scores `1` against every
other participant; non-winning pairs score `0.5`; a match without a winner is
all draws. Each pair delta uses `K / (N − 1)`, and all deltas are applied
simultaneously.

This keeps the update zero-sum, makes the two-player case exactly ordinary Elo,
and increments every participant's game count once. Existing schema-1
two-player league JSON upgrades without rating drift. Existing schema-1 room
files are decoded through their old fixed-seat structure and rewritten to the
variable-seat storage schema after replay verification.
