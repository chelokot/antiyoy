import assert from "node:assert/strict";
import test from "node:test";

import {
  MultiplayerConnection,
  RULES_PROFILE_WIRE,
  claimOpenSeat,
  createInviteUrl,
  createJoinableDuel,
  createJoinableMatch,
  createOpenSeatInvites,
  fetchLeague,
  leagueStandings,
  parseInvite,
  roomConfigFromSnapshot,
  type MatchSnapshot,
} from "../app/multiplayer-client";

const MATCH_ID = "0123456789abcdef0123456789abcdef";

test("maps every browser rules profile to the exact Rust wire variant", () => {
  assert.deepEqual(RULES_PROFILE_WIRE, {
    classic_generic_2022: "ClassicGeneric",
    classic_slay_2022: "ClassicSlay",
    online_default_v1: "OnlineDefaultV1",
    online_classic_v1: "OnlineClassicV1",
    online_duel_v1: "OnlineDuelV1",
    online_experimental_v1: "OnlineExperimentalV1",
    online_experimental_v2_260801: "OnlineExperimentalV2_260801",
  });
});

function snapshot(
  status: MatchSnapshot["status"] = "Running",
  overrides: Partial<MatchSnapshot> = {},
): MatchSnapshot {
  return {
    schema_version: 6,
    match_id: MATCH_ID,
    revision: 4,
    status,
    rating_status: "NotFinished",
    actions_played: 4,
    digest: Array.from({ length: 32 }, () => 0),
    rules_profile: "OnlineDuelV1",
    scenario: { SymmetricDuel: { width: 11, height: 9, seed: "47" } },
    seats: [
      { name: "host", kind: "Human" },
      { name: status === "Waiting" ? "open-seat-2" : "guest", kind: status === "Waiting" ? "Open" : "Human" },
    ],
    game: {
      width: 11,
      height: 9,
      round: 1,
      active_player: 0,
      terminal: false,
      winner: null,
      cells: [],
      provinces: [],
      relations: [],
      legal_actions: status === "Waiting" ? [] : ["EndTurn"],
    },
    ...overrides,
  };
}

test("creates a waiting duel and keeps the seat token out of its invite URL", async () => {
  const originalFetch = globalThis.fetch;
  let requestBody = "";
  globalThis.fetch = async (_input, init) => {
    requestBody = String(init?.body);
    return Response.json({
      snapshot: snapshot("Waiting"),
      credentials: [{ seat: 0, name: "host", token: "host-secret" }],
    });
  };
  try {
    const session = await createJoinableDuel("https://antiyoy.test/", "host", "47");
    const invite = createInviteUrl("https://arena.test/", session.snapshot.match_id, 1);
    assert.equal(session.endpoint, "https://antiyoy.test");
    assert.equal(parseInvite(new URL(invite).hash)?.matchId, MATCH_ID);
    assert.equal(parseInvite(new URL(invite).hash)?.seat, 1);
    assert.doesNotMatch(invite, /host-secret/);
    assert.deepEqual(JSON.parse(requestBody), {
      schema_version: 6,
      rules_profile: "OnlineDuelV1",
      scenario: { SymmetricDuel: { width: 11, height: 9, seed: "47" } },
      seats: [
        { name: "host", kind: "Human" },
        { name: "open-seat-2", kind: "Open" },
      ],
      action_limit: 10_000,
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("creates a four-player procedural room with one token-free invite per open seat", async () => {
  const originalFetch = globalThis.fetch;
  let requestBody = "";
  const seats: MatchSnapshot["seats"] = [
    { name: "host", kind: "Human" },
    { name: "open-seat-2", kind: "Open" },
    { name: "open-seat-3", kind: "Open" },
    { name: "open-seat-4", kind: "Open" },
  ];
  const proceduralSnapshot = snapshot("Waiting", {
    rules_profile: "OnlineExperimentalV2_260801",
    scenario: {
      Procedural: {
        schema_version: 1,
        width: 17,
        height: 13,
        players: 4,
        seed: "18446744073709551615",
        land_density_per_million: 600_000,
        starting_province_size: 5,
        starting_money: 10,
        tree_density_per_million: 150_000,
        neutral_tower_density_per_million: 20_000,
        neutral_capital_density_per_million: 10_000,
        grave_density_per_million: 15_000,
      },
    },
    seats,
  });
  globalThis.fetch = async (_input, init) => {
    requestBody = String(init?.body);
    return Response.json({
      snapshot: proceduralSnapshot,
      credentials: [{ seat: 0, name: "host", token: "host-secret" }],
    });
  };
  try {
    const session = await createJoinableMatch("https://antiyoy.test", "host", {
      map: "procedural",
      profile: "online_experimental_v2_260801",
      width: 17,
      height: 13,
      players: 4,
      seed: "18446744073709551615",
      landDensity: 600_000,
    });
    const request = JSON.parse(requestBody) as {
      schema_version: number;
      rules_profile: string;
      scenario: MatchSnapshot["scenario"];
      seats: MatchSnapshot["seats"];
    };
    assert.equal(request.schema_version, 6);
    assert.equal(request.rules_profile, "OnlineExperimentalV2_260801");
    assert.deepEqual(request.scenario, proceduralSnapshot.scenario);
    assert.deepEqual(request.seats, seats);
    const invites = createOpenSeatInvites("https://arena.test/", session.snapshot);
    assert.deepEqual(invites.map((invite) => invite.seat), [1, 2, 3]);
    assert.equal(new Set(invites.map((invite) => invite.url)).size, 3);
    assert.ok(invites.every((invite) => !invite.url.includes("host-secret")));
    assert.deepEqual(roomConfigFromSnapshot(session.snapshot), {
      map: "procedural",
      profile: "online_experimental_v2_260801",
      width: 17,
      height: 13,
      players: 4,
      seed: "18446744073709551615",
      landDensity: 600_000,
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("claims the invited seat through the versioned endpoint", async () => {
  const originalFetch = globalThis.fetch;
  let requestUrl = "";
  globalThis.fetch = async (input) => {
    requestUrl = String(input);
    return Response.json({
      snapshot: snapshot(),
      credential: { seat: 1, name: "guest", token: "guest-secret" },
    });
  };
  try {
    const session = await claimOpenSeat("https://antiyoy.test", MATCH_ID, 1, "guest");
    assert.equal(session.credential.token, "guest-secret");
    assert.equal(
      requestUrl,
      `https://antiyoy.test/v1/matches/${MATCH_ID}/seats/1/claim`,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("loads exact authoritative standings and a full-u64 verified ledger", async () => {
  const originalFetch = globalThis.fetch;
  let requestUrl = "";
  let requestMethod = "";
  const league = {
    schema_version: 2,
    elo: { k_factor: 32 },
    participants: {
      host: { rating: { elo: 1016, games: 1 }, wins: 1, draws: 0, losses: 0 },
      guest: { rating: { elo: 984, games: 1 }, wins: 0, draws: 0, losses: 1 },
    },
    matches: [{
      id: "ab".repeat(32),
      agents: ["host", "guest"],
      player_count: 2,
      seed: "18446744073709551615",
      outcome: { winner: 0, actions: 5, termination: "ActionLimit" },
      final_digest: Array.from({ length: 32 }, (_, byte) => byte),
    }],
  };
  globalThis.fetch = async (input, init) => {
    requestUrl = String(input);
    requestMethod = init?.method ?? "";
    return Response.json(league);
  };
  try {
    const loaded = await fetchLeague("https://antiyoy.test/");
    assert.equal(requestUrl, "https://antiyoy.test/v1/league");
    assert.equal(requestMethod, "GET");
    assert.equal(loaded.matches[0]?.seed, "18446744073709551615");
    assert.deepEqual(leagueStandings(loaded).map((standing) => ({
      rank: standing.rank,
      name: standing.name,
      elo: standing.rating.elo,
      record: [standing.wins, standing.draws, standing.losses],
    })), [
      { rank: 1, name: "host", elo: 1016, record: [1, 0, 0] },
      { rank: 2, name: "guest", elo: 984, record: [0, 0, 1] },
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("rejects lossy or internally inconsistent league payloads", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => Response.json({
    schema_version: 2,
    elo: { k_factor: 32 },
    participants: {
      host: { rating: { elo: 1000, games: 1 }, wins: 0, draws: 1, losses: 0 },
      guest: { rating: { elo: 1000, games: 1 }, wins: 0, draws: 1, losses: 0 },
    },
    matches: [{
      id: "cd".repeat(32),
      agents: ["host", "guest"],
      player_count: 2,
      seed: Number.MAX_SAFE_INTEGER,
      outcome: { winner: null, actions: 5, termination: "ActionLimit" },
      final_digest: Array.from({ length: 32 }, () => 0),
    }],
  });
  try {
    await assert.rejects(
      fetchLeague("https://antiyoy.test"),
      /incompatible league snapshot/,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("authenticates over WebSocket and submits the exact authoritative action", () => {
  class MockSocket extends EventTarget {
    readonly sent: string[] = [];
    readyState: number = WebSocket.OPEN;

    send(message: string): void {
      this.sent.push(message);
    }

    close(): void {
      this.readyState = WebSocket.CLOSED;
    }
  }

  const socket = new MockSocket();
  const snapshots: MatchSnapshot[] = [];
  const statuses: string[] = [];
  const errors: string[] = [];
  const connection = new MultiplayerConnection(
    "https://antiyoy.test",
    { seat: 1, name: "guest", token: "guest-secret" },
    MATCH_ID,
    {
      onSnapshot: (next) => snapshots.push(next),
      onStatus: (status) => statuses.push(status),
      onError: (message) => errors.push(message),
    },
    () => socket as unknown as WebSocket,
  );
  connection.connect();
  socket.dispatchEvent(new Event("open"));
  socket.dispatchEvent(new MessageEvent("message", {
    data: JSON.stringify({ Snapshot: snapshot() }),
  }));
  socket.dispatchEvent(new MessageEvent("message", {
    data: JSON.stringify({ Authenticated: { seat: 1 } }),
  }));
  connection.submit("EndTurn", 4);

  assert.deepEqual(JSON.parse(socket.sent[0]), {
    Authenticate: { schema_version: 6, seat: 1, token: "guest-secret" },
  });
  assert.deepEqual(JSON.parse(socket.sent[1]), {
    Submit: { schema_version: 6, revision: 4, action: "EndTurn" },
  });
  assert.equal(snapshots.length, 1);
  assert.deepEqual(statuses, ["connecting", "authenticated"]);
  assert.deepEqual(errors, []);
  connection.disconnect();
});
