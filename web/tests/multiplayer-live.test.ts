import assert from "node:assert/strict";
import test from "node:test";

import {
  NETWORK_SCHEMA_VERSION,
  claimOpenSeat,
  createJoinableDuel,
  type MatchSnapshot,
  type OnlineSession,
} from "../app/multiplayer-client";

const endpoint = process.env.ANTIYOY_MULTIPLAYER_ENDPOINT;

type ConnectedProbe = {
  socket: WebSocket;
  snapshots: MatchSnapshot[];
};

test("two independent clients complete an authoritative round", {
  skip: endpoint === undefined,
  timeout: 20_000,
}, async () => {
  assert.ok(endpoint);
  const host = await createJoinableDuel(endpoint, "e2e-host", "9001");
  let guest: OnlineSession | null = null;
  let hostProbe: ConnectedProbe | null = null;
  let guestProbe: ConnectedProbe | null = null;
  try {
    assert.equal(host.snapshot.status, "Waiting");
    guest = await claimOpenSeat(endpoint, host.snapshot.match_id, 1, "e2e-guest");
    assert.equal(guest.snapshot.status, "Running");
    [hostProbe, guestProbe] = await Promise.all([connect(host), connect(guest)]);

    submitEndTurn(hostProbe.socket, guest.snapshot.revision);
    const [hostSawGuestTurn, guestSawGuestTurn] = await Promise.all([
      waitForSnapshot(hostProbe, (snapshot) => snapshot.revision === 1),
      waitForSnapshot(guestProbe, (snapshot) => snapshot.revision === 1),
    ]);
    assert.equal(hostSawGuestTurn.game.active_player, 1);
    assert.deepEqual(hostSawGuestTurn, guestSawGuestTurn);

    submitEndTurn(guestProbe.socket, guestSawGuestTurn.revision);
    const [hostSawRound, guestSawRound] = await Promise.all([
      waitForSnapshot(hostProbe, (snapshot) => snapshot.revision === 2),
      waitForSnapshot(guestProbe, (snapshot) => snapshot.revision === 2),
    ]);
    assert.equal(hostSawRound.game.active_player, 0);
    assert.equal(hostSawRound.game.round, 2);
    assert.deepEqual(hostSawRound, guestSawRound);

    const authoritative = await fetch(`${endpoint}/v1/matches/${host.snapshot.match_id}`);
    assert.equal(authoritative.status, 200);
    assert.deepEqual(await authoritative.json(), hostSawRound);
    const replay = await fetch(`${endpoint}/v1/matches/${host.snapshot.match_id}/replay`);
    assert.equal(replay.status, 200);
    assert.equal(replay.headers.get("content-type"), "application/vnd.antiyoy.replay");
    assert.ok((await replay.arrayBuffer()).byteLength > 100);

    const preflight = await fetch(`${endpoint}/v1/matches`, {
      method: "OPTIONS",
      headers: {
        origin: "http://localhost:3000",
        "access-control-request-method": "POST",
      },
    });
    assert.equal(preflight.status, 200);
    assert.equal(preflight.headers.get("access-control-allow-origin"), "http://localhost:3000");
  } finally {
    hostProbe?.socket.close();
    guestProbe?.socket.close();
    await fetch(`${endpoint}/v1/matches/${host.snapshot.match_id}?seat=0`, {
      method: "DELETE",
      headers: { authorization: `Bearer ${host.credential.token}` },
    });
  }
});

async function connect(session: OnlineSession): Promise<ConnectedProbe> {
  const url = new URL(`${session.endpoint}/v1/matches/${session.snapshot.match_id}/watch`);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(url);
  const probe: ConnectedProbe = { socket, snapshots: [] };
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("WebSocket authentication timed out")), 5_000);
    socket.addEventListener("open", () => {
      socket.send(JSON.stringify({
        Authenticate: {
          schema_version: NETWORK_SCHEMA_VERSION,
          seat: session.credential.seat,
          token: session.credential.token,
        },
      }));
    });
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(String(event.data)) as {
        Snapshot?: MatchSnapshot;
        Authenticated?: { seat: number };
        Error?: { code: string; message: string };
      };
      if (message.Snapshot !== undefined) {
        probe.snapshots.push(message.Snapshot);
      }
      if (message.Authenticated !== undefined) {
        clearTimeout(timeout);
        assert.equal(message.Authenticated.seat, session.credential.seat);
        resolve();
      }
      if (message.Error !== undefined) {
        clearTimeout(timeout);
        reject(new Error(`${message.Error.code}: ${message.Error.message}`));
      }
    });
    socket.addEventListener("error", () => {
      clearTimeout(timeout);
      reject(new Error("WebSocket connection failed"));
    });
  });
  return probe;
}

function submitEndTurn(socket: WebSocket, revision: number): void {
  socket.send(JSON.stringify({
    Submit: {
      schema_version: NETWORK_SCHEMA_VERSION,
      revision,
      action: "EndTurn",
    },
  }));
}

async function waitForSnapshot(
  probe: ConnectedProbe,
  predicate: (snapshot: MatchSnapshot) => boolean,
): Promise<MatchSnapshot> {
  const existing = probe.snapshots.find(predicate);
  if (existing !== undefined) {
    return existing;
  }
  return new Promise<MatchSnapshot>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Snapshot broadcast timed out")), 5_000);
    const listener = (event: MessageEvent) => {
      const message = JSON.parse(String(event.data)) as { Snapshot?: MatchSnapshot };
      if (message.Snapshot !== undefined && predicate(message.Snapshot)) {
        clearTimeout(timeout);
        probe.socket.removeEventListener("message", listener);
        resolve(message.Snapshot);
      }
    };
    probe.socket.addEventListener("message", listener);
  });
}
