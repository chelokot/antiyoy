import assert from "node:assert/strict";
import test from "node:test";

import {
  MultiplayerConnection,
  claimOpenSeat,
  createInviteUrl,
  createJoinableDuel,
  parseInvite,
  type MatchSnapshot,
} from "../app/multiplayer-client";

const MATCH_ID = "0123456789abcdef0123456789abcdef";

function snapshot(status: MatchSnapshot["status"] = "Running"): MatchSnapshot {
  return {
    schema_version: 5,
    match_id: MATCH_ID,
    revision: 4,
    status,
    rating_status: "NotFinished",
    actions_played: 4,
    scenario: { SymmetricDuel: { width: 11, height: 9, seed: 47 } },
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
      schema_version: 5,
      rules_profile: "OnlineDuelV1",
      scenario: { SymmetricDuel: { width: 11, height: 9, seed: 47 } },
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

test("authenticates over WebSocket and submits the exact authoritative action", () => {
  class MockSocket extends EventTarget {
    readonly sent: string[] = [];
    readyState = WebSocket.OPEN;

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
    Authenticate: { schema_version: 5, seat: 1, token: "guest-secret" },
  });
  assert.deepEqual(JSON.parse(socket.sent[1]), {
    Submit: { schema_version: 5, revision: 4, action: "EndTurn" },
  });
  assert.equal(snapshots.length, 1);
  assert.deepEqual(statuses, ["connecting", "authenticated"]);
  assert.deepEqual(errors, []);
  connection.disconnect();
});
