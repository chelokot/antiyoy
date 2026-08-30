import type { CoreAction, StateView } from "./game-types";

export const NETWORK_SCHEMA_VERSION = 6;
export const DEFAULT_MULTIPLAYER_ENDPOINT = "https://antiyoy.test";

export const RULES_PROFILE_WIRE = {
  classic_generic_2022: "ClassicGeneric",
  classic_slay_2022: "ClassicSlay",
  online_default_v1: "OnlineDefaultV1",
  online_classic_v1: "OnlineClassicV1",
  online_duel_v1: "OnlineDuelV1",
  online_experimental_v1: "OnlineExperimentalV1",
  online_experimental_v2_260801: "OnlineExperimentalV2_260801",
} as const;

export type RulesProfileId = keyof typeof RULES_PROFILE_WIRE;
export type WireRulesProfile = typeof RULES_PROFILE_WIRE[RulesProfileId];
export type OnlineRoomConfig = {
  map: "duel" | "procedural";
  profile: RulesProfileId;
  width: number;
  height: number;
  players: number;
  seed: string;
  landDensity: number;
};

export type MatchStatus = "Waiting" | "Running" | "Victory" | "ActionLimit";
export type ConnectionStatus = "connecting" | "authenticated" | "disconnected";
export type SeatKind = "Human" | "Greedy" | "Random" | "Search" | "Open";

export type SeatCredential = {
  seat: number;
  name: string;
  token: string;
};

type GeneratorConfig = {
  schema_version: 1;
  width: number;
  height: number;
  players: number;
  seed: string;
  land_density_per_million: number;
  starting_province_size: 5;
  starting_money: 10;
  tree_density_per_million: 150000;
  neutral_tower_density_per_million: 20000;
  neutral_capital_density_per_million: 10000;
  grave_density_per_million: 15000;
};

export type MatchScenario =
  | { SymmetricDuel: { width: number; height: number; seed: string } }
  | { Procedural: GeneratorConfig };

export type MatchSnapshot = {
  schema_version: number;
  match_id: string;
  revision: number;
  status: MatchStatus;
  rating_status: "NotFinished" | "Pending" | "Recorded" | "Duplicate";
  actions_played: number;
  digest: number[];
  rules_profile: WireRulesProfile;
  scenario: MatchScenario;
  seats: Array<{ name: string; kind: SeatKind }>;
  game: StateView;
};

export type SeatInvite = {
  seat: number;
  url: string;
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

type CreateMatchRequest = {
  schema_version: number;
  rules_profile: WireRulesProfile;
  scenario: MatchScenario;
  seats: Array<{ name: string; kind: SeatKind }>;
  action_limit: number;
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

export async function createJoinableMatch(
  endpoint: string,
  hostName: string,
  config: OnlineRoomConfig,
): Promise<OnlineSession> {
  const request = createMatchRequest(hostName, config);
  const response = await requestJson<CreateMatchResponse>(endpoint, "/v1/matches", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
  });
  assertSnapshot(response.snapshot);
  const credential = response.credentials[0];
  if (response.credentials.length !== 1 || credential === undefined || credential.seat !== 0) {
    throw new Error("Server returned invalid host credentials");
  }
  return { endpoint: normalizeEndpoint(endpoint), credential, snapshot: response.snapshot };
}

export async function createJoinableDuel(
  endpoint: string,
  hostName: string,
  seed: string,
): Promise<OnlineSession> {
  return createJoinableMatch(endpoint, hostName, {
    map: "duel",
    profile: "online_duel_v1",
    width: 11,
    height: 9,
    players: 2,
    seed,
    landDensity: 650_000,
  });
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

export function createOpenSeatInvites(currentUrl: string, snapshot: MatchSnapshot): SeatInvite[] {
  return snapshot.seats.flatMap((seat, index) => seat.kind === "Open"
    ? [{ seat: index, url: createInviteUrl(currentUrl, snapshot.match_id, index) }]
    : []);
}

export function roomConfigFromSnapshot(snapshot: MatchSnapshot): OnlineRoomConfig {
  const profile = profileIdFromWire(snapshot.rules_profile);
  if ("SymmetricDuel" in snapshot.scenario) {
    return {
      map: "duel",
      profile,
      width: snapshot.scenario.SymmetricDuel.width,
      height: snapshot.scenario.SymmetricDuel.height,
      players: 2,
      seed: snapshot.scenario.SymmetricDuel.seed.toString(),
      landDensity: 650_000,
    };
  }
  const config = snapshot.scenario.Procedural;
  return {
    map: "procedural",
    profile,
    width: config.width,
    height: config.height,
    players: config.players,
    seed: config.seed.toString(),
    landDensity: config.land_density_per_million,
  };
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

function createMatchRequest(hostName: string, config: OnlineRoomConfig): CreateMatchRequest {
  if (hostName.length === 0 || hostName.length > 64) {
    throw new Error("Player name must contain between 1 and 64 characters");
  }
  if (!/^\d+$/.test(config.seed)) {
    throw new Error("Seed must be an unsigned integer");
  }
  const seed = BigInt(config.seed);
  if (seed > 18_446_744_073_709_551_615n) {
    throw new Error("Seed exceeds the u64 range");
  }
  if (!Number.isInteger(config.width) || config.width < 5 || config.width > 41) {
    throw new Error("Width must be an integer between 5 and 41");
  }
  if (!Number.isInteger(config.height) || config.height < 2 || config.height > 31) {
    throw new Error("Height must be an integer between 2 and 31");
  }
  if (!Number.isInteger(config.players) || config.players < 2 || config.players > 8) {
    throw new Error("Player count must be an integer between 2 and 8");
  }
  if (config.map === "duel" && config.players !== 2) {
    throw new Error("Symmetric duel requires exactly two players");
  }
  if (
    !Number.isInteger(config.landDensity)
    || config.landDensity < 200_000
    || config.landDensity > 1_000_000
  ) {
    throw new Error("Land density must be between 200000 and 1000000 ppm");
  }
  const scenario: MatchScenario = config.map === "duel"
    ? { SymmetricDuel: { width: config.width, height: config.height, seed: seed.toString() } }
    : {
        Procedural: {
          schema_version: 1,
          width: config.width,
          height: config.height,
          players: config.players,
          seed: seed.toString(),
          land_density_per_million: config.landDensity,
          starting_province_size: 5,
          starting_money: 10,
          tree_density_per_million: 150_000,
          neutral_tower_density_per_million: 20_000,
          neutral_capital_density_per_million: 10_000,
          grave_density_per_million: 15_000,
        },
      };
  return {
    schema_version: NETWORK_SCHEMA_VERSION,
    rules_profile: RULES_PROFILE_WIRE[config.profile],
    scenario,
    seats: Array.from({ length: config.players }, (_, seat) => {
      const kind: SeatKind = seat === 0 ? "Human" : "Open";
      return { name: seat === 0 ? hostName : `open-seat-${seat + 1}`, kind };
    }),
    action_limit: 10_000,
  };
}

function profileIdFromWire(profile: WireRulesProfile): RulesProfileId {
  const entry = Object.entries(RULES_PROFILE_WIRE)
    .find(([, candidate]) => candidate === profile);
  if (entry === undefined) {
    throw new Error("Server returned an unknown rules profile");
  }
  return entry[0] as RulesProfileId;
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
    || !Array.isArray(snapshot.digest)
    || snapshot.digest.length !== 32
    || snapshot.seats.length < 2
    || snapshot.seats.length > 8
    || !Array.isArray(snapshot.game.legal_actions)
    || !Array.isArray(snapshot.game.cells)
  ) {
    throw new Error("Server returned an incompatible multiplayer snapshot");
  }
  const config = roomConfigFromSnapshot(snapshot);
  if (config.players !== snapshot.seats.length) {
    throw new Error("Server returned a mismatched multiplayer scenario");
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
