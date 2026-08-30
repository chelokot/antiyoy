import assert from "node:assert/strict";
import test from "node:test";

import {
  NETWORK_SCHEMA_VERSION,
  claimOpenSeat,
  createJoinableDuel,
  createJoinableMatch,
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

test("four invited players complete a procedural authoritative round", {
  skip: endpoint === undefined,
  timeout: 30_000,
}, async () => {
  assert.ok(endpoint);
  const host = await createJoinableMatch(endpoint, "four-host", {
    map: "procedural",
    profile: "online_default_v1",
    width: 17,
    height: 13,
    players: 4,
    seed: "18446744073709551615",
    landDensity: 650_000,
  });
  const sessions: OnlineSession[] = [host];
  const probes: ConnectedProbe[] = [];
  try {
    assert.equal(host.snapshot.status, "Waiting");
    for (let seat = 1; seat < 4; seat += 1) {
      const joined = await claimOpenSeat(
        endpoint,
        host.snapshot.match_id,
        seat,
        `four-guest-${seat + 1}`,
      );
      sessions.push(joined);
      assert.equal(joined.snapshot.status, seat === 3 ? "Running" : "Waiting");
      assert.equal(joined.snapshot.game.legal_actions.length > 0, seat === 3);
    }
    assert.deepEqual(sessions[3]?.snapshot.scenario, {
      Procedural: {
        schema_version: 1,
        width: 17,
        height: 13,
        players: 4,
        seed: "18446744073709551615",
        land_density_per_million: 650_000,
        starting_province_size: 5,
        starting_money: 10,
        tree_density_per_million: 150_000,
        neutral_tower_density_per_million: 20_000,
        neutral_capital_density_per_million: 10_000,
        grave_density_per_million: 15_000,
      },
    });
    assert.equal(sessions[3]?.snapshot.rules_profile, "OnlineDefaultV1");
    probes.push(...await Promise.all(sessions.map(connect)));

    let authoritative = sessions[3]?.snapshot;
    assert.ok(authoritative);
    for (let seat = 0; seat < 4; seat += 1) {
      assert.equal(authoritative.game.active_player, seat);
      const activeProbe = probes[seat];
      assert.ok(activeProbe);
      submitEndTurn(activeProbe.socket, authoritative.revision);
      const nextRevision = authoritative.revision + 1;
      const broadcasts = await Promise.all(
        probes.map((probe) => waitForSnapshot(
          probe,
          (snapshot) => snapshot.revision === nextRevision,
        )),
      );
      const firstBroadcast = broadcasts[0];
      assert.ok(firstBroadcast);
      broadcasts.forEach((snapshot) => assert.deepEqual(snapshot, firstBroadcast));
      authoritative = firstBroadcast;
    }
    assert.equal(authoritative.game.active_player, 0);
    assert.equal(authoritative.game.round, 2);
    assert.equal(authoritative.revision, 4);
  } finally {
    probes.forEach((probe) => probe.socket.close());
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
