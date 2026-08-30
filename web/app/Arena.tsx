"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import type {
  WasmGame as WasmGameType,
  WasmReplay as WasmReplayType,
} from "@/lib/antiyoy-wasm/antiyoy_wasm";
import { RoutedBrowserPolicy, type PolicyDecision } from "./browser-policy";
import type { CellView, CoreAction, StateView } from "./game-types";
import {
  DEFAULT_MULTIPLAYER_ENDPOINT,
  MultiplayerConnection,
  claimOpenSeat,
  createJoinableMatch,
  createOpenSeatInvites,
  createRatedBotChallenge,
  deleteOnlineMatch,
  fetchLeague,
  leagueStandings,
  parseInvite,
  roomConfigFromSnapshot,
  type ConnectionStatus,
  type LeagueMatch,
  type LeagueSnapshot,
  type LeagueStanding,
  type OnlineRoomConfig,
  type OnlineSession,
  type RulesProfileId,
} from "./multiplayer-client";

type WasmModule = typeof import("@/lib/antiyoy-wasm/antiyoy_wasm");

type ReplayMetadata = {
  seed: string;
  frames: number;
  engine_version: number;
  format_version: number;
  rules_profile: string;
};

type LiveConfig = OnlineRoomConfig;
type RulesProfileName = RulesProfileId;
type BotStrengthName = typeof BOT_STRENGTH_NAMES[number];
type BotOpponentName = "neural" | BotStrengthName;
type BrowserPolicyKey = keyof typeof BROWSER_POLICY_MODELS;
type PolicyStatus = "loading" | "ready" | "error";
type LeagueStatus = "idle" | "loading" | "ready" | "error";

type PlacementRating = {
  version: 1;
  elo: number;
  games: number;
  wins: number;
  draws: number;
  losses: number;
  attempts: number;
};

const WIDTH = 11;
const HEIGHT = 9;
const SEED = 47n;
const PLACEMENT_STORAGE_KEY = "antiyoy-arena-placement-v1";
const MULTIPLAYER_ENDPOINT_STORAGE_KEY = "antiyoy-arena-multiplayer-endpoint-v1";
const MULTIPLAYER_NAME_STORAGE_KEY = "antiyoy-arena-multiplayer-name-v1";
const SERVER_CHALLENGE_ATTEMPT_STORAGE_KEY = "antiyoy-arena-server-challenge-attempt-v1";
const PLACEMENT_SEED_BASE = 9_140_003n;
const PLACEMENT_SEED_STEP = 104_729n;
const SEARCH_ELO = 1_000;
const RATED_SEARCH_NODES = 2_048;
const INITIAL_PLACEMENT: PlacementRating = {
  version: 1,
  elo: 1_000,
  games: 0,
  wins: 0,
  draws: 0,
  losses: 0,
  attempts: 0,
};
const DEFAULT_CONFIG: LiveConfig = {
  map: "duel",
  profile: "classic_generic_2022",
  width: WIDTH,
  height: HEIGHT,
  players: 2,
  seed: SEED.toString(),
  landDensity: 650_000,
};
const RULES_PROFILES = [
  { id: "classic_generic_2022", label: "Classic Generic" },
  { id: "classic_slay_2022", label: "Classic Slay" },
  { id: "online_default_v1", label: "Online Default" },
  { id: "online_classic_v1", label: "Online Classic" },
  { id: "online_duel_v1", label: "Online Duel" },
  { id: "online_experimental_v1", label: "Experimental v1" },
  { id: "online_experimental_v2_260801", label: "Experimental v2" },
] as const;
const BOT_STRENGTHS = {
  quick: { label: "Quick · 64", nodes: 64 },
  strong: { label: "Strong · 256", nodes: 256 },
  brutal: { label: "Brutal · full turn", nodes: 2_048 },
} as const;
const BOT_STRENGTH_NAMES = ["quick", "strong", "brutal"] as const;
const BROWSER_POLICY_MODELS = {
  primary: "/browser-primary.onnx",
  experimentalV2: "/browser-experimental-v2.onnx",
  onlineDefaultSeat0: "/browser-online-default-seat0-v6.onnx",
  onlineDuelSeat0: "/browser-online-duel-seat0-v6.onnx",
  onlineExperimentalV1Seat0: "/browser-online-experimental-v1-seat0-v6.onnx",
} as const;
const PLAYER_NAMES = ["CYAN", "AMBER", "VIOLET", "CORAL", "LIME", "BLUE", "PINK", "SILVER"] as const;
const MODEL_URL = "https://github.com/chelokot/antiyoy/releases/tag/model-v0.5.0-beta.1";
const MODEL_RESULTS = [
  ["Classic · Generic + Slay", "96–0"],
  ["Online Default", "48–0"],
  ["Online Classic", "48–0"],
  ["Duel + Experimental v1", "96–0"],
  ["Experimental v2", "48–0"],
] as const;

function parseState(serialized: string): StateView {
  return JSON.parse(serialized) as StateView;
}

function playerLabel(player: number): string {
  return PLAYER_NAMES[player] ?? `PLAYER ${player + 1}`;
}

function centerPlayableCell(state: StateView): number {
  const center = (state.cells.length - 1) / 2;
  const playable = state.cells.filter((cell) => cell.playable);
  return playable.reduce(
    (closest, cell) => Math.abs(cell.id - center) < Math.abs(closest - center) ? cell.id : closest,
    playable[0]?.id ?? 0,
  );
}

function createGame(bindings: WasmModule, config: LiveConfig): WasmGameType {
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
  if (!Number.isInteger(config.landDensity) || config.landDensity < 200_000 || config.landDensity > 1_000_000) {
    throw new Error("Land density must be between 200000 and 1000000 ppm");
  }
  if (config.map === "procedural") {
    return bindings.WasmGame.procedural_with_profile(
      config.width,
      config.height,
      config.players,
      seed,
      config.landDensity,
      config.profile,
    );
  }
  return bindings.WasmGame.with_profile(
    config.width,
    config.height,
    seed,
    config.profile,
  );
}

function advanceBotsUntilHuman(
  instance: WasmGameType,
  initial: StateView,
  humanSeat: number,
  searchNodes: number,
): { state: StateView; actions: number } {
  let state = initial;
  let actions = 0;
  while (!state.terminal && state.active_player !== humanSeat && actions < 2_000) {
    state = parseState(instance.step_search_with_budget(searchNodes));
    actions += 1;
  }
  if (!state.terminal && state.active_player !== humanSeat) {
    throw new Error("Bot response exceeded 2000 actions");
  }
  return { state, actions };
}

function loadPlacementRating(): PlacementRating {
  const serialized = window.localStorage.getItem(PLACEMENT_STORAGE_KEY);
  if (serialized === null) {
    return INITIAL_PLACEMENT;
  }
  try {
    const rating = JSON.parse(serialized) as PlacementRating;
    if (
      rating.version === 1
      && Number.isFinite(rating.elo)
      && [rating.games, rating.wins, rating.draws, rating.losses, rating.attempts]
        .every((value) => Number.isInteger(value) && value >= 0)
      && rating.games === rating.wins + rating.draws + rating.losses
      && rating.attempts >= rating.games
    ) {
      return rating;
    }
  } catch {
    return INITIAL_PLACEMENT;
  }
  return INITIAL_PLACEMENT;
}

function recordPlacementResult(
  rating: PlacementRating,
  score: 0 | 0.5 | 1,
): PlacementRating {
  const expected = 1 / (1 + 10 ** ((SEARCH_ELO - rating.elo) / 400));
  const kFactor = rating.games < 10 ? 40 : 20;
  return {
    ...rating,
    elo: rating.elo + kFactor * (score - expected),
    games: rating.games + 1,
    wins: rating.wins + Number(score === 1),
    draws: rating.draws + Number(score === 0.5),
    losses: rating.losses + Number(score === 0),
  };
}

function pieceLabel(cell: CellView): string {
  if (cell.strength > 0) {
    return `UNIT ${cell.strength}`;
  }
  return cell.object.toUpperCase();
}

function pieceGlyph(cell: CellView): string {
  if (cell.strength > 0) {
    return ["", "Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ"][cell.strength];
  }
  return {
    Capital: "♛",
    Farm: "⌂",
    Tower: "♜",
    StrongTower: "⬟",
    Pine: "♠",
    Palm: "♣",
    Grave: "†",
  }[cell.object] ?? "";
}

function actionTarget(action: CoreAction): number | null {
  if (typeof action === "string") {
    return null;
  }
  if ("Move" in action) {
    return action.Move.target;
  }
  if ("Recruit" in action) {
    return action.Recruit.target;
  }
  if ("Build" in action) {
    return action.Build.target;
  }
  if ("PlantTree" in action) {
    return action.PlantTree.target;
  }
  return null;
}

function hexCoordinates(hex: number, width: number): string {
  return `${hex % width},${Math.floor(hex / width)}`;
}

function actionLabel(action: CoreAction, width: number): string {
  if (action === "EndTurn") {
    return "End turn";
  }
  if ("Move" in action) {
    return `Move ${hexCoordinates(action.Move.source, width)} → here`;
  }
  if ("Recruit" in action) {
    return `Recruit unit ${action.Recruit.strength}`;
  }
  if ("Build" in action) {
    return `Build ${action.Build.structure.replace(/([A-Z])/g, " $1").trim().toLowerCase()}`;
  }
  if ("PlantTree" in action) {
    return "Plant tree";
  }
  return `${action.Diplomacy.command.replace(/([A-Z])/g, " $1").trim()} player ${action.Diplomacy.target + 1}`;
}

function matchOpponentLabel(
  replayMetadata: ReplayMetadata | null,
  placementMode: boolean,
  humanMode: boolean,
  opponent: BotOpponentName,
): string {
  if (replayMetadata !== null) {
    return "recorded trace";
  }
  if (placementMode) {
    return "rated search · 2048 nodes";
  }
  if (humanMode) {
    return opponent === "neural"
      ? "routed neural · ONNX"
      : `${BOT_STRENGTHS[opponent].label.toLowerCase()} search`;
  }
  return opponent === "neural" ? "routed neural self-play" : "greedy vs full-turn search";
}

function supportsNeuralPolicy(config: LiveConfig): boolean {
  return config.map === "duel"
    && config.width === WIDTH
    && config.height === HEIGHT
    && config.players === 2;
}

function policyKeyForProfile(profile: RulesProfileName, seat = 1): BrowserPolicyKey {
  if (seat === 0) {
    if (profile === "online_default_v1") {
      return "onlineDefaultSeat0";
    }
    if (profile === "online_duel_v1") {
      return "onlineDuelSeat0";
    }
    if (profile === "online_experimental_v1") {
      return "onlineExperimentalV1Seat0";
    }
  }
  return profile === "online_experimental_v2_260801" ? "experimentalV2" : "primary";
}

function rulesProfileLabel(profile: RulesProfileName): string {
  return RULES_PROFILES.find((candidate) => candidate.id === profile)?.label ?? profile;
}

export default function Arena() {
  const wasmModule = useRef<WasmModule | null>(null);
  const game = useRef<WasmGameType | null>(null);
  const replay = useRef<WasmReplayType | null>(null);
  const browserPolicies = useRef<Partial<Record<BrowserPolicyKey, RoutedBrowserPolicy>>>({});
  const multiplayerConnection = useRef<MultiplayerConnection | null>(null);
  const leagueRequest = useRef(0);
  const botResponseInFlight = useRef(false);
  const placementRecorded = useRef(false);
  const boardViewport = useRef<HTMLDivElement | null>(null);
  const boardContent = useRef<HTMLDivElement | null>(null);
  const [state, setState] = useState<StateView | null>(null);
  const [selectedId, setSelectedId] = useState(Math.floor((WIDTH * HEIGHT) / 2));
  const [playing, setPlaying] = useState(false);
  const [actions, setActions] = useState(0);
  const [engineVersion, setEngineVersion] = useState<number | null>(null);
  const [replayMetadata, setReplayMetadata] = useState<ReplayMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [humanMode, setHumanMode] = useState(true);
  const [humanSeat, setHumanSeat] = useState(0);
  const [placementMode, setPlacementMode] = useState(false);
  const [placement, setPlacement] = useState<PlacementRating>(INITIAL_PLACEMENT);
  const [draftConfig, setDraftConfig] = useState<LiveConfig>(DEFAULT_CONFIG);
  const [activeConfig, setActiveConfig] = useState<LiveConfig>(DEFAULT_CONFIG);
  const [boardScale, setBoardScale] = useState(1);
  const [openPanel, setOpenPanel] = useState<"left" | "right" | null>(null);
  const [botOpponent, setBotOpponent] = useState<BotOpponentName>("neural");
  const [policyStatuses, setPolicyStatuses] = useState<Record<BrowserPolicyKey, PolicyStatus>>({
    primary: "loading",
    experimentalV2: "loading",
    onlineDefaultSeat0: "loading",
    onlineDuelSeat0: "loading",
    onlineExperimentalV1Seat0: "loading",
  });
  const [botThinking, setBotThinking] = useState(false);
  const [policyDecision, setPolicyDecision] = useState<PolicyDecision | null>(null);
  const [onlineSession, setOnlineSession] = useState<OnlineSession | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("disconnected");
  const [onlineEndpoint, setOnlineEndpoint] = useState(DEFAULT_MULTIPLAYER_ENDPOINT);
  const [onlineName, setOnlineName] = useState("player");
  const [joinCode, setJoinCode] = useState("");
  const [joinSeat, setJoinSeat] = useState(1);
  const [inviteBaseUrl, setInviteBaseUrl] = useState("");
  const [copiedInviteSeat, setCopiedInviteSeat] = useState<number | null>(null);
  const [onlineBusy, setOnlineBusy] = useState(false);
  const [league, setLeague] = useState<LeagueSnapshot | null>(null);
  const [leagueStatus, setLeagueStatus] = useState<LeagueStatus>("idle");
  const [leagueError, setLeagueError] = useState<string | null>(null);
  const [leagueEndpoint, setLeagueEndpoint] = useState<string | null>(null);
  const neuralPolicySeat = humanMode ? (humanSeat + 1) % 2 : state?.active_player ?? 0;
  const activePolicyKey = policyKeyForProfile(activeConfig.profile, neuralPolicySeat);
  const policyStatus = policyStatuses[activePolicyKey];
  const activePolicyKeyRef = useRef(policyKeyForProfile(DEFAULT_CONFIG.profile));

  const refreshLeague = useCallback(async () => {
    const request = leagueRequest.current + 1;
    leagueRequest.current = request;
    setLeagueEndpoint(onlineEndpoint);
    setLeagueStatus("loading");
    setLeagueError(null);
    try {
      const snapshot = await fetchLeague(onlineEndpoint);
      if (leagueRequest.current !== request) {
        return;
      }
      setLeague(snapshot);
      setLeagueStatus("ready");
    } catch (reason: unknown) {
      if (leagueRequest.current !== request) {
        return;
      }
      setLeague(null);
      setLeagueStatus("error");
      setLeagueError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [onlineEndpoint]);

  const beginOnlineSession = useCallback((session: OnlineSession) => {
    multiplayerConnection.current?.disconnect();
    replay.current?.free();
    replay.current = null;
    setReplayMetadata(null);
    setPlacementMode(false);
    setHumanMode(true);
    setHumanSeat(session.credential.seat);
    setPlaying(false);
    setBotThinking(false);
    setOnlineSession(session);
    setState(session.snapshot.game);
    setActions(session.snapshot.actions_played);
    setSelectedId(centerPlayableCell(session.snapshot.game));
    setActiveConfig(roomConfigFromSnapshot(session.snapshot));
    const connection = new MultiplayerConnection(
      session.endpoint,
      session.credential,
      session.snapshot.match_id,
      {
        onSnapshot: (snapshot) => {
          setOnlineSession((current) => current === null ? null : { ...current, snapshot });
          setState(snapshot.game);
          setActions(snapshot.actions_played);
          botResponseInFlight.current = false;
          setBotThinking(false);
          setError(null);
          if (snapshot.rating_status === "Recorded") {
            void refreshLeague();
          }
        },
        onStatus: setConnectionStatus,
        onError: (message) => {
          botResponseInFlight.current = false;
          setBotThinking(false);
          setError(message);
        },
      },
    );
    multiplayerConnection.current = connection;
    connection.connect();
  }, [refreshLeague]);

  const disconnectOnline = useCallback(() => {
    multiplayerConnection.current?.disconnect();
    multiplayerConnection.current = null;
    setOnlineSession(null);
    setConnectionStatus("disconnected");
    setCopiedInviteSeat(null);
    setBotThinking(false);
  }, []);

  useEffect(() => {
    activePolicyKeyRef.current = activePolicyKey;
  }, [activePolicyKey]);

  useEffect(() => {
    let disposed = false;
    const load = async (key: BrowserPolicyKey) => {
      try {
        const policy = await RoutedBrowserPolicy.load(BROWSER_POLICY_MODELS[key]);
        if (disposed) {
          policy.release();
          return;
        }
        browserPolicies.current[key] = policy;
        setPolicyStatuses((current) => ({ ...current, [key]: "ready" }));
      } catch {
        if (!disposed) {
          setPolicyStatuses((current) => ({ ...current, [key]: "error" }));
          if (activePolicyKeyRef.current === key) {
            setBotOpponent((current) => current === "neural" ? "brutal" : current);
          }
        }
      }
    };
    void load("primary")
      .then(() => load("experimentalV2"))
      .then(() => load("onlineDefaultSeat0"))
      .then(() => load("onlineDuelSeat0"))
      .then(() => load("onlineExperimentalV1Seat0"));
    return () => {
      disposed = true;
      Object.values(browserPolicies.current).forEach((policy) => policy.release());
      browserPolicies.current = {};
    };
  }, []);

  useEffect(() => {
    const closePanel = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpenPanel(null);
      }
    };
    window.addEventListener("keydown", closePanel);
    return () => window.removeEventListener("keydown", closePanel);
  }, []);

  useEffect(() => {
    let disposed = false;
    queueMicrotask(() => {
      if (disposed) {
        return;
      }
      setInviteBaseUrl(window.location.href);
      const savedEndpoint = window.localStorage.getItem(MULTIPLAYER_ENDPOINT_STORAGE_KEY);
      if (savedEndpoint !== null) {
        setOnlineEndpoint(savedEndpoint);
      }
      const savedName = window.localStorage.getItem(MULTIPLAYER_NAME_STORAGE_KEY);
      if (savedName !== null && savedName.length > 0 && savedName.length <= 64) {
        setOnlineName(savedName);
      }
      const invite = parseInvite(window.location.hash);
      if (invite !== null) {
        setJoinCode(invite.matchId);
        setJoinSeat(invite.seat);
        setOpenPanel("left");
      }
    });
    return () => {
      disposed = true;
      multiplayerConnection.current?.disconnect();
      multiplayerConnection.current = null;
    };
  }, []);

  useEffect(() => {
    let disposed = false;
    void import("@/lib/antiyoy-wasm/antiyoy_wasm").then(async (module) => {
      await module.default({
        module_or_path: new URL("/antiyoy_wasm_bg.wasm", window.location.origin),
      });
      if (disposed) {
        return;
      }
      setPlacement(loadPlacementRating());
      wasmModule.current = module;
      const instance = createGame(module, DEFAULT_CONFIG);
      game.current = instance;
      setEngineVersion(module.engine_version());
      const initialState = parseState(instance.state_json());
      setState(initialState);
      setSelectedId(centerPlayableCell(initialState));
    }).catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => {
      disposed = true;
      replay.current?.free();
      replay.current = null;
      game.current?.free();
      game.current = null;
      wasmModule.current = null;
    };
  }, []);

  useEffect(() => {
    const viewport = boardViewport.current;
    const content = boardContent.current;
    if (viewport === null || content === null) {
      return;
    }
    const fit = () => {
      const availableWidth = Math.max(1, viewport.clientWidth - 24);
      const availableHeight = Math.max(1, viewport.clientHeight - 24);
      const naturalWidth = Math.max(1, content.offsetWidth);
      const naturalHeight = Math.max(1, content.offsetHeight);
      setBoardScale(Math.min(1.25, availableWidth / naturalWidth, availableHeight / naturalHeight));
    };
    const observer = new ResizeObserver(fit);
    observer.observe(viewport);
    observer.observe(content);
    fit();
    return () => observer.disconnect();
  }, [state?.width, state?.height]);

  const createOnlineRoom = useCallback(async () => {
    setOnlineBusy(true);
    try {
      const session = await createJoinableMatch(onlineEndpoint, onlineName, draftConfig);
      window.localStorage.setItem(MULTIPLAYER_ENDPOINT_STORAGE_KEY, session.endpoint);
      window.localStorage.setItem(MULTIPLAYER_NAME_STORAGE_KEY, onlineName);
      setOnlineEndpoint(session.endpoint);
      setInviteBaseUrl(window.location.href);
      setCopiedInviteSeat(null);
      beginOnlineSession(session);
      setOpenPanel(null);
      setError(null);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setOnlineBusy(false);
    }
  }, [beginOnlineSession, draftConfig, onlineEndpoint, onlineName]);

  const startRatedChallenge = useCallback(async () => {
    setOnlineBusy(true);
    try {
      const storedAttempt = Number(window.localStorage.getItem(SERVER_CHALLENGE_ATTEMPT_STORAGE_KEY) ?? "0");
      const attempt = Number.isSafeInteger(storedAttempt) && storedAttempt >= 0 ? storedAttempt : 0;
      const humanSeat = attempt % draftConfig.players;
      const session = await createRatedBotChallenge(
        onlineEndpoint,
        onlineName,
        draftConfig,
        humanSeat,
      );
      window.localStorage.setItem(MULTIPLAYER_ENDPOINT_STORAGE_KEY, session.endpoint);
      window.localStorage.setItem(MULTIPLAYER_NAME_STORAGE_KEY, onlineName);
      window.localStorage.setItem(SERVER_CHALLENGE_ATTEMPT_STORAGE_KEY, (attempt + 1).toString());
      setOnlineEndpoint(session.endpoint);
      setInviteBaseUrl(window.location.href);
      setCopiedInviteSeat(null);
      beginOnlineSession(session);
      setOpenPanel(null);
      setError(null);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setOnlineBusy(false);
    }
  }, [beginOnlineSession, draftConfig, onlineEndpoint, onlineName]);

  const joinOnlineRoom = useCallback(async () => {
    setOnlineBusy(true);
    try {
      const session = await claimOpenSeat(onlineEndpoint, joinCode, joinSeat, onlineName);
      window.localStorage.setItem(MULTIPLAYER_ENDPOINT_STORAGE_KEY, session.endpoint);
      window.localStorage.setItem(MULTIPLAYER_NAME_STORAGE_KEY, onlineName);
      window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
      setOnlineEndpoint(session.endpoint);
      setInviteBaseUrl(window.location.href);
      setCopiedInviteSeat(null);
      beginOnlineSession(session);
      setOpenPanel(null);
      setError(null);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setOnlineBusy(false);
    }
  }, [beginOnlineSession, joinCode, joinSeat, onlineEndpoint, onlineName]);

  const copyInvite = useCallback(async (seat: number, inviteUrl: string) => {
    try {
      await navigator.clipboard.writeText(inviteUrl);
      setCopiedInviteSeat(seat);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, []);

  const generate = useCallback(() => {
    const bindings = wasmModule.current;
    if (bindings === null) {
      return;
    }
    let candidate: WasmGameType | null = null;
    try {
      disconnectOnline();
      candidate = createGame(bindings, draftConfig);
      const next = parseState(candidate.state_json());
      replay.current?.free();
      replay.current = null;
      game.current?.free();
      game.current = candidate;
      candidate = null;
      setReplayMetadata(null);
      setPlacementMode(false);
      setHumanSeat(0);
      activePolicyKeyRef.current = policyKeyForProfile(draftConfig.profile);
      setActiveConfig(draftConfig);
      setState(next);
      setSelectedId(centerPlayableCell(next));
      setActions(0);
      setPolicyDecision(null);
      if (
        !supportsNeuralPolicy(draftConfig)
        || policyStatuses[policyKeyForProfile(draftConfig.profile)] === "error"
      ) {
        setBotOpponent("brutal");
      }
      setPlaying(false);
      setOpenPanel(null);
      setError(null);
    } catch (reason: unknown) {
      candidate?.free();
      setPlaying(false);
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [disconnectOnline, draftConfig, policyStatuses]);

  const startPlacement = useCallback(() => {
    const bindings = wasmModule.current;
    if (bindings === null) {
      return;
    }
    const attempt = placement.attempts;
    const seat = attempt % 2;
    const placementConfig: LiveConfig = {
      map: "duel",
      profile: "classic_generic_2022",
      width: WIDTH,
      height: HEIGHT,
      players: 2,
      seed: (PLACEMENT_SEED_BASE + BigInt(attempt) * PLACEMENT_SEED_STEP).toString(),
      landDensity: DEFAULT_CONFIG.landDensity,
    };
    let candidate: WasmGameType | null = null;
    try {
      disconnectOnline();
      candidate = createGame(bindings, placementConfig);
      const advanced = advanceBotsUntilHuman(
        candidate,
        parseState(candidate.state_json()),
        seat,
        RATED_SEARCH_NODES,
      );
      replay.current?.free();
      replay.current = null;
      game.current?.free();
      game.current = candidate;
      candidate = null;
      placementRecorded.current = false;
      const nextPlacement = { ...placement, attempts: attempt + 1 };
      window.localStorage.setItem(PLACEMENT_STORAGE_KEY, JSON.stringify(nextPlacement));
      setPlacement(nextPlacement);
      setReplayMetadata(null);
      activePolicyKeyRef.current = policyKeyForProfile(placementConfig.profile);
      setActiveConfig(placementConfig);
      setState(advanced.state);
      setSelectedId(centerPlayableCell(advanced.state));
      setActions(advanced.actions);
      setPolicyDecision(null);
      setHumanSeat(seat);
      setHumanMode(true);
      setPlacementMode(true);
      setPlaying(false);
      setOpenPanel(null);
      setError(null);
    } catch (reason: unknown) {
      candidate?.free();
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [disconnectOnline, placement]);

  const step = useCallback(async () => {
    const replayInstance = replay.current;
    if (replayInstance !== null && replayMetadata !== null) {
      const nextFrame = Math.min(actions + 1, replayMetadata.frames);
      try {
        setState(parseState(replayInstance.seek(nextFrame)));
        setActions(nextFrame);
        if (nextFrame === replayMetadata.frames) {
          setPlaying(false);
        }
      } catch (reason: unknown) {
        setPlaying(false);
        setError(reason instanceof Error ? reason.message : String(reason));
      }
      return;
    }
    const instance = game.current;
    if (instance === null) {
      return;
    }
    if (!humanMode && botOpponent === "neural") {
      if (botResponseInFlight.current || state === null) {
        return;
      }
      const policy = browserPolicies.current[
        policyKeyForProfile(activeConfig.profile, state.active_player)
      ];
      if (policy === undefined) {
        return;
      }
      botResponseInFlight.current = true;
      setBotThinking(true);
      try {
        const decision = await policy.decide(instance.policy_observation_json());
        setPolicyDecision(decision);
        const next = parseState(instance.step(decision.actionIndex));
        setState(next);
        setActions((current) => current + 1);
        if (next.terminal) {
          setPlaying(false);
        }
        setError(null);
      } catch (reason: unknown) {
        setPlaying(false);
        setError(reason instanceof Error ? reason.message : String(reason));
      } finally {
        botResponseInFlight.current = false;
        setBotThinking(false);
      }
      return;
    }
    try {
      const next = parseState(instance.step_bot());
      setState(next);
      setActions((current) => current + 1);
      if (next.terminal) {
        setPlaying(false);
      }
    } catch (reason: unknown) {
      setPlaying(false);
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [actions, activeConfig.profile, botOpponent, humanMode, replayMetadata, state]);

  useEffect(() => {
    if (!playing) {
      return;
    }
    const interval = window.setInterval(() => void step(), 180);
    return () => window.clearInterval(interval);
  }, [playing, step]);

  const reset = useCallback(() => {
    if (onlineSession !== null) {
      const session = onlineSession;
      const bindings = wasmModule.current;
      if (bindings === null) {
        return;
      }
      disconnectOnline();
      setError(null);
      void deleteOnlineMatch(session).catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : String(reason));
      });
      const instance = createGame(bindings, DEFAULT_CONFIG);
      game.current?.free();
      game.current = instance;
      const initialState = parseState(instance.state_json());
      setActiveConfig(DEFAULT_CONFIG);
      setDraftConfig(DEFAULT_CONFIG);
      setState(initialState);
      setSelectedId(centerPlayableCell(initialState));
      setActions(0);
      setPlaying(false);
      return;
    }
    if (replay.current !== null) {
      setState(parseState(replay.current.seek(0)));
      setActions(0);
      setPlaying(false);
      setError(null);
      return;
    }
    if (game.current === null) {
      return;
    }
    setPlacementMode(false);
    setHumanSeat(0);
    setState(parseState(game.current.reset()));
    setActions(0);
    setPolicyDecision(null);
    setPlaying(false);
    setError(null);
  }, [disconnectOnline, onlineSession]);

  const playHumanAction = useCallback(async (actionIndex: number) => {
    if (onlineSession !== null) {
      const action = onlineSession.snapshot.game.legal_actions[actionIndex];
      if (
        action === undefined
        || onlineSession.snapshot.status !== "Running"
        || botResponseInFlight.current
      ) {
        return;
      }
      botResponseInFlight.current = true;
      setBotThinking(true);
      try {
        const connection = multiplayerConnection.current;
        if (connection === null) {
          throw new Error("Multiplayer connection is unavailable");
        }
        connection.submit(action, onlineSession.snapshot.revision);
      } catch (reason: unknown) {
        botResponseInFlight.current = false;
        setBotThinking(false);
        setError(reason instanceof Error ? reason.message : String(reason));
      }
      return;
    }
    const instance = game.current;
    if (instance === null || replay.current !== null || botResponseInFlight.current) {
      return;
    }
    botResponseInFlight.current = true;
    setBotThinking(true);
    try {
      const afterHuman = parseState(instance.step(actionIndex));
      setState(afterHuman);
      let responseState = afterHuman;
      let responseActions = 0;
      if (!placementMode && botOpponent === "neural") {
        while (
          !responseState.terminal
          && responseState.active_player !== humanSeat
          && responseActions < 2_000
        ) {
          const policy = browserPolicies.current[
            policyKeyForProfile(activeConfig.profile, responseState.active_player)
          ];
          if (policy === undefined) {
            throw new Error("Neural policy is still loading");
          }
          const decision = await policy.decide(instance.policy_observation_json());
          setPolicyDecision(decision);
          responseState = parseState(instance.step(decision.actionIndex));
          responseActions += 1;
        }
        if (!responseState.terminal && responseState.active_player !== humanSeat) {
          throw new Error("Neural response exceeded 2000 actions");
        }
      } else {
        const searchNodes = placementMode
          ? RATED_SEARCH_NODES
          : BOT_STRENGTHS[botOpponent as BotStrengthName].nodes;
        const response = advanceBotsUntilHuman(instance, afterHuman, humanSeat, searchNodes);
        responseState = response.state;
        responseActions = response.actions;
      }
      setState(responseState);
      setActions((current) => current + responseActions + 1);
      setPlaying(false);
      setError(null);
    } catch (reason: unknown) {
      setPlaying(false);
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      botResponseInFlight.current = false;
      setBotThinking(false);
    }
  }, [activeConfig.profile, botOpponent, humanSeat, onlineSession, placementMode]);

  useEffect(() => {
    if (
      !placementMode
      || replay.current !== null
      || state === null
      || !state.terminal
      || placementRecorded.current
    ) {
      return;
    }
    placementRecorded.current = true;
    const score = state.winner === null ? 0.5 : state.winner === humanSeat ? 1 : 0;
    setPlacement((current) => {
      const next = recordPlacementResult(current, score);
      window.localStorage.setItem(PLACEMENT_STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, [humanSeat, placementMode, state]);

  const toggleHumanMode = useCallback(() => {
    disconnectOnline();
    setPlacementMode(false);
    setHumanSeat(0);
    setHumanMode((current) => !current);
    reset();
  }, [disconnectOnline, reset]);

  const seekReplay = useCallback((frame: number) => {
    if (replay.current === null) {
      return;
    }
    try {
      setState(parseState(replay.current.seek(frame)));
      setActions(frame);
      setPlaying(false);
      setError(null);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, []);

  const loadReplay = useCallback(async (file: File | undefined) => {
    const bindings = wasmModule.current;
    if (file === undefined || bindings === null) {
      return;
    }
    let candidate: WasmReplayType | null = null;
    try {
      disconnectOnline();
      candidate = new bindings.WasmReplay(new Uint8Array(await file.arrayBuffer()));
      const instance = candidate;
      const metadata = JSON.parse(instance.metadata_json()) as ReplayMetadata;
      const initialState = parseState(instance.seek(0));
      replay.current?.free();
      replay.current = instance;
      candidate = null;
      setReplayMetadata(metadata);
      setHumanMode(false);
      setPlacementMode(false);
      setHumanSeat(0);
      setState(initialState);
      setSelectedId(Math.floor(initialState.cells.length / 2));
      setActions(0);
      setPlaying(false);
      setError(null);
    } catch (reason: unknown) {
      candidate?.free();
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [disconnectOnline]);

  const restoreLive = useCallback(() => {
    disconnectOnline();
    replay.current?.free();
    replay.current = null;
    setReplayMetadata(null);
    setPlacementMode(false);
    setHumanSeat(0);
    if (game.current !== null) {
      const liveState = parseState(game.current.reset());
      setState(liveState);
      setSelectedId(centerPlayableCell(liveState));
    }
    setActions(0);
    setPlaying(false);
    setError(null);
  }, [disconnectOnline]);

  const rows = useMemo(() => {
    if (state === null) {
      return [];
    }
    return Array.from({ length: state.height }, (_, row) =>
      state.cells.slice(row * state.width, (row + 1) * state.width),
    );
  }, [state]);

  const selected = state?.cells[selectedId] ?? null;
  const province = selected?.province === null || selected === null
    ? null
    : state?.provinces.find((candidate) => candidate.id === selected.province) ?? null;
  const selectedQ = selectedId % (state?.width ?? WIDTH);
  const selectedR = Math.floor(selectedId / (state?.width ?? WIDTH));
  const playerCount = state === null ? activeConfig.players : Math.sqrt(state.relations.length);
  const territories = Array.from(
    { length: playerCount },
    (_, player) => state?.cells.filter((cell) => cell.owner === player).length ?? 0,
  );
  const controlledCells = territories.reduce((total, cells) => total + cells, 0);
  const territoryShares = territories.map((cells) => controlledCells === 0 ? 0 : (cells / controlledCells) * 100);
  const selectedActions = (state?.legal_actions ?? [])
    .map((action, index) => ({ action, index }))
    .filter(({ action }) => actionTarget(action) === selectedId);
  const globalActions = (state?.legal_actions ?? [])
    .map((action, index) => ({ action, index }))
    .filter(({ action }) => actionTarget(action) === null);
  const actionableTargets = new Set(
    (state?.legal_actions ?? []).flatMap((action) => {
      const target = actionTarget(action);
      return target === null ? [] : [target];
    }),
  );
  const humanCanAct = onlineSession === null
    ? humanMode
      && replayMetadata === null
      && state !== null
      && state.active_player === humanSeat
      && !state.terminal
      && !botThinking
      && (placementMode || botOpponent !== "neural" || policyStatus === "ready")
    : onlineSession.snapshot.status === "Running"
      && connectionStatus === "authenticated"
      && state !== null
      && state.active_player === humanSeat
      && !state.terminal
      && !botThinking;
  const opponentLabel = onlineSession === null
    ? matchOpponentLabel(replayMetadata, placementMode, humanMode, botOpponent)
    : onlineSession.snapshot.seats
      .filter((_, seat) => seat !== humanSeat)
      .map((seat) => seat.name)
      .join(", ");
  const openSeatInvites = useMemo(
    () => onlineSession === null || inviteBaseUrl.length === 0
      ? []
      : createOpenSeatInvites(inviteBaseUrl, onlineSession.snapshot),
    [inviteBaseUrl, onlineSession],
  );
  const displayedLeague = leagueEndpoint === onlineEndpoint ? league : null;
  const displayedLeagueStatus = leagueEndpoint === onlineEndpoint ? leagueStatus : "idle";
  const displayedLeagueError = leagueEndpoint === onlineEndpoint ? leagueError : null;
  const standings = useMemo(
    () => displayedLeague === null ? [] : leagueStandings(displayedLeague),
    [displayedLeague],
  );
  const roomConfigSummary = `${rulesProfileLabel(draftConfig.profile)} · ${draftConfig.map === "duel" ? "duel" : `${draftConfig.width}×${draftConfig.height} procedural`} · ${draftConfig.map === "duel" ? 2 : draftConfig.players} players`;
  const economyPlayer = humanMode ? humanSeat : state?.active_player ?? 0;
  const economyProvince = province?.owner === economyPlayer
    ? province
    : state?.provinces.find((candidate) => candidate.owner === economyPlayer) ?? null;
  const economy = economyProvince ?? { money: 0, profit: 0 };

  return (
    <main className="arena-shell">
      <div className="arena-layout">
        <aside id="match-drawer" aria-label="Match settings" className={`arena-sidebar arena-sidebar-left ${openPanel === "left" ? "panel-open" : ""}`}>
          <div className="panel-heading"><div><p className="eyebrow">CONTROL ROOM</p><p className="panel-heading-title">Match &amp; setup</p></div><button className="panel-close" type="button" aria-label="Close overview panel" onClick={() => setOpenPanel(null)}>×</button></div>
          <div className="panel-scroll">
          <details className="panel-section panel-section-hero" open>
            <summary>MATCH</summary>
            <div className="panel-section-body"><p className="eyebrow">{onlineSession !== null ? "AUTHORITATIVE MULTIPLAYER" : replayMetadata === null ? placementMode ? "RATED PLACEMENT" : humanMode ? "HUMAN VS AI" : "SELF-PLAY" : "VERIFIED REPLAY"}</p><h2 className="mt-2 text-xl font-semibold">{replayMetadata === null ? humanMode ? `you are ${playerLabel(humanSeat).toLowerCase()}` : botOpponent === "neural" ? "neural policy mirror" : "greedy vs turn-search" : "training trace"}</h2><p className="mt-1 text-sm text-[#686a65]">{onlineSession !== null ? onlineSession.snapshot.status === "Waiting" ? "waiting for the invited player" : "every move is validated by the Rust server" : replayMetadata === null ? placementMode ? "local Elo vs fixed search-2048" : humanMode ? "select a highlighted hex, then choose an action" : botOpponent === "neural" ? "the routed model plays every seat" : "deterministic whole-turn planning" : `${replayMetadata.frames} deterministic actions`}</p><div className="mt-6 space-y-4"><Metric label="RULESET" value={replayMetadata?.rules_profile ?? activeConfig.profile} /><Metric label="MAP" value={replayMetadata === null ? activeConfig.map === "procedural" ? "procedural_v1" : "symmetric_duel_v1" : "replay scenario"} /><Metric label="OPPONENT" value={opponentLabel || "open seat"} /><Metric label="SEED" value={onlineSession === null ? `${replayMetadata?.seed ?? activeConfig.seed} · reproducible` : "server-authoritative"} /><Metric label="ROUND" value={state === null ? "loading" : `${state.round} · ${playerLabel(state.active_player)} to move`} /><Metric label="LEGAL ACTIONS" value={state?.legal_actions.length.toString() ?? "…"} accent />{onlineSession === null && !placementMode && botOpponent === "neural" && <Metric label="NEURAL INFERENCE" value={policyDecision === null ? policyStatus : `${policyDecision.milliseconds.toFixed(1)} ms · ${policyDecision.legalActions} actions`} accent />}</div></div>
          </details>
          {replayMetadata === null && <details className="panel-section panel-section-online"><summary>ONLINE MULTIPLAYER · {onlineSession?.snapshot.status ?? "READY"}</summary><div className="panel-section-body map-config"><label className="config-field"><span>PLAYER NAME</span><input type="text" maxLength={64} value={onlineName} onChange={(event) => setOnlineName(event.target.value)} /></label>{onlineSession === null ? <><div className="online-config-summary"><p className="eyebrow">ROOM SETTINGS</p><p>{roomConfigSummary}</p><span>Change them in Game Config before creating the room.</span></div><button className="generate-button" type="button" disabled={onlineBusy || onlineName.length === 0} onClick={() => void createOnlineRoom()}>{onlineBusy ? "Connecting…" : `Create ${draftConfig.map === "duel" ? 2 : draftConfig.players}-player room`}</button><button className="generate-button rated-challenge-button" type="button" disabled={onlineBusy || onlineName.length === 0} onClick={() => void startRatedChallenge()}>{onlineBusy ? "Starting…" : `Play rated vs ${draftConfig.players - 1} server search ${draftConfig.players === 2 ? "bot" : "bots"}`}</button><p className="rated-challenge-note">Your seat rotates after every successful challenge. The replay-verified result enters Server League Elo.</p><div className="online-divider"><span>OR JOIN</span></div><div className="config-grid online-join-grid"><label className="config-field"><span>ROOM CODE</span><input type="text" spellCheck={false} value={joinCode} onChange={(event) => setJoinCode(event.target.value.trim())} /></label><label className="config-field"><span>SEAT</span><input type="number" min="2" max="8" value={joinSeat + 1} onChange={(event) => setJoinSeat(Number(event.target.value) - 1)} /></label></div><button className="generate-button" type="button" disabled={onlineBusy || onlineName.length === 0 || joinCode.length !== 32 || joinSeat < 1 || joinSeat > 7} onClick={() => void joinOnlineRoom()}>{onlineBusy ? "Claiming seat…" : `Join as seat ${joinSeat + 1}`}</button></> : <div className="online-session"><div className="online-status-row"><span className={`online-status-dot online-status-${connectionStatus}`} /><span>{connectionStatus}</span><span>seat {onlineSession.credential.seat + 1}/{onlineSession.snapshot.seats.length}</span></div><p className="online-room-code">{onlineSession.snapshot.match_id}</p><p className={`online-rating-status online-rating-${onlineSession.snapshot.rating_status.toLowerCase()}`}>ELO · {onlineSession.snapshot.rating_status}</p>{openSeatInvites.length > 0 && <div className="online-invites"><p className="eyebrow">PRIVATE SEAT LINKS · {openSeatInvites.length} OPEN</p>{openSeatInvites.map((invite) => <button className="generate-button" type="button" onClick={() => void copyInvite(invite.seat, invite.url)} key={invite.seat}>{copiedInviteSeat === invite.seat ? `Seat ${invite.seat + 1} invite copied` : `Copy invite for seat ${invite.seat + 1}`}</button>)}<p>Match starts automatically after every open seat is claimed.</p></div>}<button className="online-leave" type="button" onClick={reset}>Leave room</button></div>}<label className="config-field online-endpoint"><span>AUTHORITATIVE SERVER</span><input type="url" spellCheck={false} disabled={onlineSession !== null} value={onlineEndpoint} onChange={(event) => setOnlineEndpoint(event.target.value)} /></label></div></details>}
          <details className="panel-section panel-section-league" onToggle={(event) => { if (event.currentTarget.open && displayedLeagueStatus === "idle") void refreshLeague(); }}><summary>SERVER LEAGUE · {displayedLeague === null ? "—" : `${displayedLeague.matches.length} RATED`}</summary><div className="panel-section-body"><LeaguePanel league={displayedLeague} standings={standings} status={displayedLeagueStatus} error={displayedLeagueError} currentName={onlineSession?.credential.name ?? onlineName} onRefresh={refreshLeague} /></div></details>
          {replayMetadata === null && onlineSession === null && <details className="panel-section"><summary>GAME CONFIG</summary><div className="panel-section-body map-config"><label className="config-field"><span>RULESET</span><select value={draftConfig.profile} onChange={(event) => setDraftConfig((current) => ({ ...current, profile: event.target.value as RulesProfileName }))}>{RULES_PROFILES.map((profile) => <option value={profile.id} key={profile.id}>{profile.label}</option>)}</select></label><div className="config-grid"><label className="config-field"><span>MODE</span><select value={draftConfig.map} onChange={(event) => setDraftConfig((current) => ({ ...current, map: event.target.value as LiveConfig["map"], players: event.target.value === "duel" ? 2 : current.players }))}><option value="duel">Symmetric duel</option><option value="procedural">Procedural v1</option></select></label><label className="config-field"><span>SEED</span><input type="text" inputMode="numeric" pattern="[0-9]+" value={draftConfig.seed} onChange={(event) => setDraftConfig((current) => ({ ...current, seed: event.target.value }))} /></label><label className="config-field"><span>WIDTH</span><input type="number" min="5" max="41" value={draftConfig.width} onChange={(event) => setDraftConfig((current) => ({ ...current, width: Number(event.target.value) }))} /></label><label className="config-field"><span>HEIGHT</span><input type="number" min="2" max="31" value={draftConfig.height} onChange={(event) => setDraftConfig((current) => ({ ...current, height: Number(event.target.value) }))} /></label><label className="config-field"><span>PLAYERS</span><input type="number" min="2" max="8" disabled={draftConfig.map === "duel"} value={draftConfig.map === "duel" ? 2 : draftConfig.players} onChange={(event) => setDraftConfig((current) => ({ ...current, players: Number(event.target.value) }))} /></label><label className="config-field"><span>LAND PPM</span><input type="number" min="200000" max="1000000" step="50000" disabled={draftConfig.map === "duel"} value={draftConfig.map === "duel" ? 650000 : draftConfig.landDensity} onChange={(event) => setDraftConfig((current) => ({ ...current, landDensity: Number(event.target.value) }))} /></label></div><button className="generate-button" type="button" onClick={generate}>Generate deterministic map</button></div></details>}
          <details className="panel-section"><summary>TERRITORY</summary><div className="panel-section-body space-y-3 text-xs">{territories.map((cells, player) => <Bar label={playerLabel(player)} value={cells} width={`${territoryShares[player]}%`} player={player} key={player} />)}</div></details>
          <details className="panel-section"><summary>ENGINE · V{engineVersion ?? "…"}</summary><div className="panel-section-body"><div className="engine-note mt-0"><p className="eyebrow">RUST + WEBASSEMBLY</p><p className="mt-2 text-sm leading-6 text-[#555752]">Every displayed transition is executed by the same deterministic headless environment used for training.</p></div></div></details>
          <details className="panel-section"><summary>BETA POLICY · 336–0</summary><div className="panel-section-body model-card">
            <div className="flex items-center justify-between gap-3"><p className="eyebrow">BETA POLICY</p><span className="font-mono text-[0.65rem] text-[#62645f]">41 experts</span></div>
            <p className="mt-2 text-sm font-semibold">universal routed · 2–8 players</p>
            <p className="mt-3 font-mono text-[0.65rem] uppercase tracking-[0.12em] text-[#62645f]">core v6 · vs search-2048 · 336–0</p>
            <dl className="mt-4 space-y-2 font-mono text-xs">{MODEL_RESULTS.map(([profile, score]) => <Row label={profile} value={score} accent key={profile} />)}</dl>
            <a className="model-download" href={MODEL_URL} target="_blank" rel="noreferrer">Download verified bundle ↗</a>
            <Link className="model-download model-results-link" href="/models">Compare agents and methods →</Link>
            <p className="mt-3 text-[0.65rem] leading-5 text-[#62645f]">Fresh held-out engine-v6 evaluation: 336 paired games, every profile and both seats, no losses or action-limit adjudications. This is not an absolute human rating.</p>
          </div></details>
          <details className="panel-section"><summary>YOUR ELO · {Math.round(placement.elo)}</summary><div className="panel-section-body model-card">
            <div className="flex items-center justify-between gap-3"><p className="eyebrow">YOUR PLACEMENT</p><span className="font-mono text-[0.65rem] text-[#62645f]">LOCAL</span></div>
            <p className="placement-rating">{Math.round(placement.elo)}</p>
            <dl className="mt-4 space-y-2 font-mono text-xs"><Row label="Games" value={placement.games.toString()} /><Row label="Record" value={`${placement.wins}–${placement.draws}–${placement.losses}`} /><Row label="Opponent" value="search-2048" /></dl>
            <button className="generate-button" type="button" onClick={startPlacement}>{placementMode ? "Start next rated match" : "Start rated match"}</button>
            <p className="panel-fine-print">Fixed 11×9 Classic arena. New deterministic seed every attempt, alternating seats, provisional K=40 for ten completed games. Stored only in this browser.</p>
          </div></details>
          </div>
        </aside>

        <section className="board-panel">
          <div className="board-controls">
            <div className="economy-hud" aria-label={`Player economy: ${economy.money} money, ${economy.profit >= 0 ? "+" : ""}${economy.profit} income`}><span className="coin-mark">$</span><strong>{economy.money}</strong><b>{economy.profit >= 0 ? "+" : ""}{economy.profit}</b></div>
            <div className="turn-chip"><span className={`turn-dot territory-player-${(state?.active_player ?? 0) % PLAYER_NAMES.length}`} /><span className="turn-copy">{onlineSession?.snapshot.status === "Waiting" ? "Waiting for player" : `Turn ${state?.round ?? "…"} · ${state === null ? "Loading" : state.active_player === humanSeat && humanMode ? "your move" : `${playerLabel(state.active_player).toLowerCase()} moves`}`}</span></div>
            {onlineSession === null && humanMode && replayMetadata === null && <label className="bot-strength"><span>Opponent</span><select aria-label="Bot opponent" disabled={placementMode || botThinking} value={placementMode ? "brutal" : botOpponent} onChange={(event) => setBotOpponent(event.target.value as BotOpponentName)}><option value="neural" disabled={!supportsNeuralPolicy(activeConfig) || policyStatus === "error"}>Neural policy{policyStatus === "loading" ? " · loading" : ""}</option>{BOT_STRENGTH_NAMES.map((strength) => <option value={strength} key={strength}>{BOT_STRENGTHS[strength].label}</option>)}</select></label>}
            {!humanMode && <><button className="control control-primary" type="button" disabled={state === null || state.terminal || (botOpponent === "neural" && policyStatus !== "ready") || (replayMetadata !== null && actions === replayMetadata.frames)} onClick={() => setPlaying((current) => !current)} aria-label={playing ? "Pause" : "Play"}>{playing ? "Ⅱ" : "▶"}</button><button className="control" type="button" disabled={state === null || state.terminal || playing || (botOpponent === "neural" && policyStatus !== "ready") || (replayMetadata !== null && actions === replayMetadata.frames)} onClick={() => void step()}>Step</button></>}
            <button className="control control-icon" type="button" disabled={state === null || botThinking} onClick={reset} aria-label={onlineSession === null ? "New game" : "Leave room"}>↻</button>
            {onlineSession === null && (replayMetadata === null ? <><button className={`control ${humanMode ? "control-active" : ""}`} type="button" disabled={botThinking} onClick={toggleHumanMode}>{humanMode ? "Auto" : "Play"}</button><label className="control cursor-pointer">Replay<input className="sr-only" type="file" accept=".antiyoy,application/octet-stream" onChange={(event) => void loadReplay(event.target.files?.[0])} /></label></> : <button className="control" type="button" onClick={restoreLive}>Live</button>)}
            <Link className="control research-link" href="/models">Models</Link>
            <button aria-expanded={openPanel === "right"} aria-controls="inspector-drawer" className="control control-icon" type="button" onClick={() => setOpenPanel("right")} aria-label="Inspect selected hex">i</button>
            <button aria-expanded={openPanel === "left"} aria-controls="match-drawer" className="control control-icon panel-toggle" type="button" onClick={() => setOpenPanel("left")} aria-label="Open game menu">⋮</button>
          </div>
          <div className={`board-scroll ${humanMode && replayMetadata === null ? "board-scroll-human" : ""}`} ref={boardViewport} aria-label="Interactive hex game board">
            <div className="board-transform" style={{ transform: `translate(-50%, -50%) scale(${boardScale})` }}>
              <div ref={boardContent} className={`hex-board ${state !== null && state.width > 15 ? "hex-board-compact" : ""}`}>
                {rows.map((row, rowIndex) => <div className="hex-row" key={rowIndex}>{row.map((cell) => <Hex cell={cell} selected={cell.id === selectedId} actionable={humanCanAct && actionableTargets.has(cell.id)} onSelect={setSelectedId} key={cell.id} />)}</div>)}
              </div>
            </div>
          </div>
          {state?.terminal && <div className="result-banner">{state.winner === null ? "DRAW" : `${playerLabel(state.winner)} WINS`} · {actions} ACTIONS</div>}
          {error !== null && <div className="error-banner">ENGINE ERROR · {error}</div>}
          {humanMode && replayMetadata === null && <div className="action-dock"><div className="action-dock-heading"><div><p className="eyebrow action-dock-eyebrow">YOUR MOVE</p><p className="action-dock-title">HEX {String(selectedQ).padStart(2, "0")},{String(selectedR).padStart(2, "0")} · {selected === null ? "LOADING" : pieceLabel(selected)}</p></div><span className="action-dock-status">{onlineSession?.snapshot.status === "Waiting" ? "WAITING FOR PLAYER" : connectionStatus !== "authenticated" && onlineSession !== null ? "CONNECTING" : state?.terminal ? "GAME OVER" : botThinking ? onlineSession !== null ? "SYNCING MOVE" : placementMode || botOpponent !== "neural" ? "BOT THINKING" : "NEURAL THINKING" : onlineSession === null && !placementMode && botOpponent === "neural" && policyStatus === "loading" ? "MODEL LOADING" : humanCanAct ? `${selectedActions.length + globalActions.length} OPTIONS` : "OPPONENT TURN"}</span></div><div className="action-dock-buttons">{selectedActions.map(({ action, index }) => <button className="action-button action-button-inline" type="button" disabled={!humanCanAct} onClick={() => void playHumanAction(index)} key={index}>{actionLabel(action, state?.width ?? WIDTH)}</button>)}{globalActions.map(({ action, index }) => <button className="action-button action-button-inline action-button-global" type="button" disabled={!humanCanAct} onClick={() => void playHumanAction(index)} key={index}>{actionLabel(action, state?.width ?? WIDTH)}</button>)}{selectedActions.length === 0 && globalActions.length === 0 && <p className="action-dock-empty">{onlineSession?.snapshot.status === "Waiting" ? "Share the private invite link to unlock the match." : "Select a glowing hex to move, recruit, or build."}</p>}</div></div>}
          <div className={`timeline ${humanMode && replayMetadata === null ? "timeline-human" : ""}`}><div className="flex items-center justify-between font-mono text-[0.65rem] text-[#8d9690]"><span>ACTION {actions}{replayMetadata === null ? "" : ` / ${replayMetadata.frames}`}</span><span>{state?.terminal ? "TERMINAL" : replayMetadata === null ? "DETERMINISTIC TRACE" : "REPLAY VERIFIED"}</span></div>{replayMetadata === null ? <div className="mt-3 flex h-1.5 overflow-hidden bg-white/10">{territoryShares.map((share, player) => <div className={`territory-player-${player % PLAYER_NAMES.length}`} style={{ width: `${share}%` }} key={player} />)}</div> : <input className="replay-scrubber" type="range" min="0" max={replayMetadata.frames} value={actions} aria-label="Replay action" onChange={(event) => seekReplay(Number(event.target.value))} />}</div>
        </section>

        <aside id="inspector-drawer" aria-label="Selected hex inspector" className={`arena-sidebar arena-sidebar-right ${openPanel === "right" ? "panel-open" : ""}`}>
          <div className="panel-heading"><div><p className="eyebrow">BOARD INSPECTOR</p><p className="panel-heading-title">Hex details</p></div><button className="panel-close" type="button" aria-label="Close inspector panel" onClick={() => setOpenPanel(null)}>×</button></div>
          <div className="panel-scroll">
          <details className="panel-section panel-section-hero" open><summary>SELECTED HEX · {String(selectedQ).padStart(2, "0")},{String(selectedR).padStart(2, "0")}</summary><div className="panel-section-body"><p className="font-mono text-lg">q: {String(selectedQ).padStart(2, "0")} · r: {String(selectedR).padStart(2, "0")}</p><div className="mt-4 grid grid-cols-2 gap-px bg-white/10"><Stat label="OWNER" value={selected?.owner === null || selected === null ? "NEUTRAL" : playerLabel(selected.owner)} /><Stat label="PIECE" value={selected === null ? "…" : pieceLabel(selected)} /><Stat label="DEFENSE" value={selected?.defense.toString() ?? "…"} /><Stat label="READY" value={selected?.strength === 0 ? "—" : selected?.ready ? "YES" : "NO"} /></div></div></details>
          <details className="panel-section" open><summary>PROVINCE ECONOMY</summary><div className="panel-section-body">{province === null ? <p className="text-sm leading-6 text-[#77817b]">This hex is not part of a connected province.</p> : <dl className="space-y-3 font-mono text-xs"><Row label="Treasury" value={`$${province.money}`} /><Row label="Hex income" value={`+${province.income}`} /><Row label="Upkeep" value={`−${province.upkeep}`} /><Row label="Next turn" value={`${province.profit >= 0 ? "+" : "−"}$${Math.abs(province.profit)}`} accent /></dl>}</div></details>
          {humanMode && replayMetadata === null && <details className="panel-section"><summary>LEGAL ACTIONS · {selectedActions.length + globalActions.length}</summary><div className="panel-section-body"><p className="text-xs leading-5 text-[#77817b]">Select a destination hex, then choose an action. Other players answer automatically.</p><div className="mt-3 grid gap-2">{selectedActions.map(({ action, index }) => <button className="action-button" type="button" disabled={!humanCanAct} onClick={() => void playHumanAction(index)} key={index}>{actionLabel(action, state?.width ?? WIDTH)}</button>)}{selectedActions.length === 0 && <p className="font-mono text-[0.65rem] text-[#626b66]">No targeted action is legal on this hex.</p>}</div><div className="mt-4 grid gap-2">{globalActions.map(({ action, index }) => <button className="action-button action-button-global" type="button" disabled={!humanCanAct} onClick={() => void playHumanAction(index)} key={index}>{actionLabel(action, state?.width ?? WIDTH)}</button>)}</div></div></details>}
          <details className="panel-section"><summary>STATE CONTRACT</summary><div className="panel-section-body"><dl className="space-y-2 font-mono text-xs"><Row label="Cells" value={state?.cells.length.toString() ?? "…"} /><Row label="Provinces" value={state?.provinces.length.toString() ?? "…"} /><Row label="Relations" value={state?.relations.length.toString() ?? "…"} /><Row label="Terminal" value={state?.terminal ? "YES" : "NO"} /></dl></div></details>
          </div>
        </aside>
        <button className={`panel-backdrop ${openPanel === null ? "" : "panel-backdrop-open"}`} type="button" aria-label="Close side panel" onClick={() => setOpenPanel(null)} />
      </div>
    </main>
  );
}

function LeaguePanel({
  league,
  standings,
  status,
  error,
  currentName,
  onRefresh,
}: {
  league: LeagueSnapshot | null;
  standings: LeagueStanding[];
  status: LeagueStatus;
  error: string | null;
  currentName: string;
  onRefresh: () => Promise<void>;
}) {
  const currentStanding = standings.find((standing) => standing.name === currentName) ?? null;
  const recentMatches = league?.matches.slice(-5).reverse() ?? [];
  return <div className="league-panel">
    <div className="league-heading"><div><p className="eyebrow">REPLAY-VERIFIED ELO</p><p>{league === null ? "Authoritative server" : `${standings.length} players · ${league.matches.length} matches`}</p></div><button className="league-refresh" type="button" disabled={status === "loading"} onClick={() => void onRefresh()}>{status === "loading" ? "Loading…" : "Refresh"}</button></div>
    {status === "error" && <div className="league-message league-message-error"><strong>League unavailable</strong><span>{error}</span><small>The private server may not be reachable from this device.</small></div>}
    {(status === "idle" || status === "loading") && league === null && <div className="league-message"><strong>{status === "loading" ? "Loading verified ledger…" : "Ready to connect"}</strong><span>Ratings come only from server-verified completed rooms.</span></div>}
    {status === "ready" && league !== null && league.matches.length === 0 && <div className="league-message"><strong>No rated matches yet</strong><span>Finish an authoritative room to create the first verified result.</span></div>}
    {currentStanding !== null && <div className="league-self"><span>YOUR SERVER RANK</span><strong>#{currentStanding.rank}</strong><b>{Math.round(currentStanding.rating.elo)} Elo</b><small>{currentStanding.wins}–{currentStanding.draws}–{currentStanding.losses}</small></div>}
    {standings.length > 0 && <div className="league-block"><div className="league-block-title"><span>STANDINGS</span><span>ELO · W–D–L</span></div><ol className="league-standings">{standings.slice(0, 20).map((standing) => <li className={standing.name === currentName ? "league-standing-current" : ""} key={standing.name}><span>{standing.rank}</span><strong title={standing.name}>{standing.name}</strong><b>{Math.round(standing.rating.elo)}</b><small>{standing.wins}–{standing.draws}–{standing.losses}</small></li>)}</ol></div>}
    {recentMatches.length > 0 && <div className="league-block"><div className="league-block-title"><span>VERIFIED LEDGER</span><span>LATEST {recentMatches.length}</span></div><div className="league-ledger">{recentMatches.map((match) => <LeagueMatchRow match={match} key={match.id} />)}</div></div>}
    {league !== null && <p className="league-proof">K={league.elo.k_factor} · deterministic replay digest required · duplicate replays rejected</p>}
  </div>;
}

function LeagueMatchRow({ match }: { match: LeagueMatch }) {
  const winner = match.outcome.winner === null ? "DRAW" : `${match.agents[match.outcome.winner]} WON`;
  const digest = match.final_digest.map((byte) => byte.toString(16).padStart(2, "0")).join("");
  return <div className="league-match"><div><strong title={match.agents.join(" · ")}>{match.agents.join(" vs ")}</strong><span>{winner} · {match.outcome.actions} actions</span></div><div><b>{match.player_count}P · {match.outcome.termination === "Victory" ? "VICTORY" : "ADJUDICATED"}</b><span title={`seed ${match.seed} · digest ${digest}`}>{digest.slice(0, 10)}…</span></div></div>;
}

function Hex({ cell, selected, actionable, onSelect }: { cell: CellView; selected: boolean; actionable: boolean; onSelect: (id: number) => void }) {
  const owner = cell.owner === null ? "neutral" : `player-${cell.owner % PLAYER_NAMES.length}`;
  const pieceClass = cell.strength > 0 ? "unit" : `piece piece-${cell.object.toLowerCase()}`;
  return <button className={`hex hex-${owner} ${cell.playable ? "" : "hex-void"} ${selected ? "hex-selected" : ""} ${actionable ? "hex-actionable" : ""}`} type="button" disabled={!cell.playable} aria-label={cell.playable ? `Hex ${cell.id}, ${pieceLabel(cell)}${actionable ? ", legal target" : ""}` : `Inactive hex ${cell.id}`} onClick={() => onSelect(cell.id)}><span className={pieceClass}>{pieceGlyph(cell)}</span></button>;
}

function Metric({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return <div><p className="eyebrow">{label}</p><p className={`metric-value ${accent ? "metric-accent" : ""}`}>{value}</p></div>;
}

function Bar({ label, value, width, player }: { label: string; value: number; width: string; player: number }) {
  return <div><div className="bar-heading"><span>{label}</span><span>{value}</span></div><div className="bar-track"><div className={`bar-fill territory-player-${player % PLAYER_NAMES.length}`} style={{ width }} /></div></div>;
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div className="stat-cell"><dt className="eyebrow">{label}</dt><dd>{value}</dd></div>;
}

function Row({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return <div className="data-row"><dt>{label}</dt><dd className={accent ? "row-accent" : ""}>{value}</dd></div>;
}
