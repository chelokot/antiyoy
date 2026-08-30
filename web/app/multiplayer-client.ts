import type { CoreAction, StateView } from "./game-types";

export const NETWORK_SCHEMA_VERSION = 5;
export const DEFAULT_MULTIPLAYER_ENDPOINT = "https://antiyoy.test";

export type MatchStatus = "Waiting" | "Running" | "Victory" | "ActionLimit";
export type ConnectionStatus = "connecting" | "authenticated" | "disconnected";
export type SeatKind = "Human" | "Greedy" | "Random" | "Search" | "Open";

export type SeatCredential = {
  seat: number;
  name: string;
  token: string;
};

export type MatchSnapshot = {
  schema_version: number;
  match_id: string;
  revision: number;
  status: MatchStatus;
  rating_status: "NotFinished" | "Pending" | "Recorded" | "Duplicate";
  actions_played: number;
  scenario: unknown;
  seats: Array<{ name: string; kind: SeatKind }>;
  game: StateView;
};

export type OnlineSession = {
  endpoint: string;
  credential: SeatCredential;
  snapshot: MatchSnapshot;
};

type CreateMatchResponse = {
  snapshot: MatchSnapshot;
  credentials: SeatCredential[];
};

type ClaimSeatResponse = {
  snapshot: MatchSnapshot;
  credential: SeatCredential;
};

type ServerMessage =
  | { Snapshot: MatchSnapshot }
  | { Authenticated: { seat: number } }
  | { Error: { code: string; message: string } };

type SocketCallbacks = {
  onSnapshot: (snapshot: MatchSnapshot) => void;
  onStatus: (status: ConnectionStatus) => void;
  onError: (message: string) => void;
};

type WebSocketFactory = (url: string) => WebSocket;

export class MultiplayerApiError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "MultiplayerApiError";
    this.code = code;
  }
}

export class MultiplayerConnection {
  private socket: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempt = 0;
  private stopped = false;
  private authenticated = false;

  constructor(
    private readonly endpoint: string,
    private readonly credential: SeatCredential,
    private readonly matchId: string,
    private readonly callbacks: SocketCallbacks,
    private readonly socketFactory: WebSocketFactory = (url) => new WebSocket(url),
  ) {}

  connect(): void {
    this.stopped = false;
    this.openSocket();
  }

  submit(action: CoreAction, revision: number): void {
    if (this.socket?.readyState !== WebSocket.OPEN || !this.authenticated) {
      throw new Error("Multiplayer connection is not authenticated");
    }
    this.socket.send(JSON.stringify({
      Submit: {
        schema_version: NETWORK_SCHEMA_VERSION,
        revision,
        action,
      },
    }));
  }

  disconnect(): void {
    this.stopped = true;
    this.authenticated = false;
    if (this.reconnectTimer !== null) {
      globalThis.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.socket?.close(1000, "client disconnect");
    this.socket = null;
    this.callbacks.onStatus("disconnected");
  }

  private openSocket(): void {
    this.callbacks.onStatus("connecting");
    const socket = this.socketFactory(watchUrl(this.endpoint, this.matchId));
    this.socket = socket;
    socket.addEventListener("open", () => {
      this.reconnectAttempt = 0;
      socket.send(JSON.stringify({
        Authenticate: {
          schema_version: NETWORK_SCHEMA_VERSION,
          seat: this.credential.seat,
          token: this.credential.token,
        },
      }));
    });
    socket.addEventListener("message", (event) => {
      const message = parseServerMessage(String(event.data));
      if ("Snapshot" in message) {
        this.callbacks.onSnapshot(message.Snapshot);
      } else if ("Authenticated" in message) {
        if (message.Authenticated.seat !== this.credential.seat) {
          this.callbacks.onError("Server authenticated the wrong seat");
          socket.close(1002, "seat mismatch");
          return;
        }
        this.authenticated = true;
        this.callbacks.onStatus("authenticated");
      } else {
        this.callbacks.onError(`${message.Error.code}: ${message.Error.message}`);
      }
    });
    socket.addEventListener("error", () => {
      this.callbacks.onError("Multiplayer socket failed");
    });
    socket.addEventListener("close", () => {
      if (this.socket !== socket) {
        return;
      }
      this.socket = null;
      this.authenticated = false;
      this.callbacks.onStatus("disconnected");
      if (!this.stopped) {
        const delay = Math.min(5_000, 500 * 2 ** this.reconnectAttempt);
        this.reconnectAttempt += 1;
        this.reconnectTimer = globalThis.setTimeout(() => this.openSocket(), delay);
      }
    });
  }
}

export async function createJoinableDuel(
  endpoint: string,
  hostName: string,
  seed: string,
): Promise<OnlineSession> {
  if (!/^\d+$/.test(seed)) {
    throw new Error("Seed must be an unsigned integer");
  }
  const response = await requestJson<CreateMatchResponse>(endpoint, "/v1/matches", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      schema_version: NETWORK_SCHEMA_VERSION,
      rules_profile: "OnlineDuelV1",
      scenario: { SymmetricDuel: { width: 11, height: 9, seed: Number(seed) } },
      seats: [
        { name: hostName, kind: "Human" },
        { name: "open-seat-2", kind: "Open" },
      ],
      action_limit: 10_000,
    }),
  });
  assertSnapshot(response.snapshot);
  const credential = response.credentials[0];
  if (response.credentials.length !== 1 || credential === undefined || credential.seat !== 0) {
    throw new Error("Server returned invalid host credentials");
  }
  return { endpoint: normalizeEndpoint(endpoint), credential, snapshot: response.snapshot };
}

export async function claimOpenSeat(
  endpoint: string,
  matchId: string,
  seat: number,
  name: string,
): Promise<OnlineSession> {
  const response = await requestJson<ClaimSeatResponse>(
    endpoint,
    `/v1/matches/${encodeURIComponent(validateMatchId(matchId))}/seats/${seat}/claim`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ schema_version: NETWORK_SCHEMA_VERSION, name }),
    },
  );
  assertSnapshot(response.snapshot);
  if (response.credential.seat !== seat || response.credential.name !== name) {
    throw new Error("Server returned invalid guest credentials");
  }
  return { endpoint: normalizeEndpoint(endpoint), credential: response.credential, snapshot: response.snapshot };
}

export function createInviteUrl(currentUrl: string, matchId: string, seat: number): string {
  const url = new URL(currentUrl);
  url.hash = new URLSearchParams({ room: validateMatchId(matchId), seat: seat.toString() }).toString();
  return url.toString();
}

export function parseInvite(hash: string): { matchId: string; seat: number } | null {
  const params = new URLSearchParams(hash.startsWith("#") ? hash.slice(1) : hash);
  const room = params.get("room");
  const seat = Number(params.get("seat"));
  if (room === null || !/^[0-9a-f]{32}$/.test(room) || !Number.isInteger(seat) || seat < 0 || seat > 7) {
    return null;
  }
  return { matchId: room, seat };
}

function parseServerMessage(serialized: string): ServerMessage {
  const message = JSON.parse(serialized) as ServerMessage;
  if ("Snapshot" in message) {
    assertSnapshot(message.Snapshot);
  }
  return message;
}

function assertSnapshot(snapshot: MatchSnapshot): void {
  if (
    snapshot.schema_version !== NETWORK_SCHEMA_VERSION
    || !/^[0-9a-f]{32}$/.test(snapshot.match_id)
    || !Number.isInteger(snapshot.revision)
    || !Array.isArray(snapshot.game.legal_actions)
    || !Array.isArray(snapshot.game.cells)
  ) {
    throw new Error("Server returned an incompatible multiplayer snapshot");
  }
}

async function requestJson<ResponseBody>(
  endpoint: string,
  path: string,
  init: RequestInit,
): Promise<ResponseBody> {
  const response = await fetch(`${normalizeEndpoint(endpoint)}${path}`, init);
  const payload = await response.json() as ResponseBody | { code: string; message: string };
  if (!response.ok) {
    const error = payload as { code: string; message: string };
    throw new MultiplayerApiError(error.code, error.message);
  }
  return payload as ResponseBody;
}

function normalizeEndpoint(endpoint: string): string {
  const url = new URL(endpoint);
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    throw new Error("Multiplayer endpoint must use HTTP or HTTPS");
  }
  url.pathname = url.pathname.replace(/\/$/, "");
  url.search = "";
  url.hash = "";
  return url.toString().replace(/\/$/, "");
}

function watchUrl(endpoint: string, matchId: string): string {
  const url = new URL(`${normalizeEndpoint(endpoint)}/v1/matches/${validateMatchId(matchId)}/watch`);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

function validateMatchId(matchId: string): string {
  if (!/^[0-9a-f]{32}$/.test(matchId)) {
    throw new Error("Room code must contain 32 lowercase hexadecimal characters");
  }
  return matchId;
}
