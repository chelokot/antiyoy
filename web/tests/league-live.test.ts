import assert from "node:assert/strict";
import test from "node:test";

import {
  NETWORK_SCHEMA_VERSION,
  fetchLeague,
  leagueStandings,
} from "../app/multiplayer-client";

const endpoint = process.env.ANTIYOY_MULTIPLAYER_ENDPOINT;
const maximumSeed = "18446744073709551615";

test("a completed server match enters the verified league exactly once", {
  skip: endpoint === undefined,
  timeout: 30_000,
}, async () => {
  assert.ok(endpoint);
  const initial = await fetchLeague(endpoint);
  const request = {
    schema_version: NETWORK_SCHEMA_VERSION,
    rules_profile: "OnlineDuelV1",
    scenario: { SymmetricDuel: { width: 7, height: 5, seed: maximumSeed } },
    seats: [
      { name: "league-e2e-greedy", kind: "Greedy" },
      { name: "league-e2e-random", kind: "Random" },
    ],
    action_limit: 5,
  };

  const firstResponse = await fetch(`${endpoint}/v1/matches`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
  });
  assert.equal(firstResponse.status, 200);
  const first = await firstResponse.json() as {
    snapshot: { status: string; rating_status: string };
  };
  assert.equal(first.snapshot.status, "ActionLimit");
  assert.equal(first.snapshot.rating_status, "Recorded");

  const rated = await fetchLeague(endpoint);
  assert.equal(rated.matches.length, initial.matches.length + 1);
  const recorded = rated.matches.at(-1);
  assert.ok(recorded);
  assert.equal(recorded.seed, maximumSeed);
  assert.deepEqual(recorded.agents, ["league-e2e-greedy", "league-e2e-random"]);
  assert.equal(recorded.outcome.actions, 5);
  assert.equal(recorded.outcome.termination, "ActionLimit");
  assert.equal(recorded.final_digest.length, 32);
  const standings = leagueStandings(rated)
    .filter((standing) => recorded.agents.includes(standing.name));
  assert.equal(standings.length, 2);
  assert.ok(standings.every((standing) => standing.rating.games === 1));
  assert.ok(Math.abs(standings.reduce((total, standing) => total + standing.rating.elo, 0) - 2_000) < 1e-9);

  const duplicateResponse = await fetch(`${endpoint}/v1/matches`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
  });
  assert.equal(duplicateResponse.status, 200);
  const duplicate = await duplicateResponse.json() as {
    snapshot: { rating_status: string };
  };
  assert.equal(duplicate.snapshot.rating_status, "Duplicate");
  assert.equal((await fetchLeague(endpoint)).matches.length, rated.matches.length);
});
