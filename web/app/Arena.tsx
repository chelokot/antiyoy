"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  WasmGame as WasmGameType,
  WasmReplay as WasmReplayType,
} from "@/lib/antiyoy-wasm/antiyoy_wasm";

type WasmModule = typeof import("@/lib/antiyoy-wasm/antiyoy_wasm");

type CellView = {
  id: number;
  playable: boolean;
  owner: number | null;
  object: string;
  strength: number;
  ready: boolean;
  province: number | null;
  defense: number;
};

type ProvinceView = {
  id: number;
  owner: number;
  money: number;
  income: number;
  upkeep: number;
  profit: number;
  capital: number;
  size: number;
};

type CoreAction =
  | "EndTurn"
  | { Move: { source: number; target: number } }
  | { Recruit: { province: number; target: number; strength: number } }
  | { Build: { target: number; structure: string } }
  | { PlantTree: { target: number } }
  | { Diplomacy: { target: number; command: string } };

type StateView = {
  width: number;
  height: number;
  round: number;
  active_player: number;
  terminal: boolean;
  winner: number | null;
  cells: CellView[];
  provinces: ProvinceView[];
  relations: Array<{
    first: number;
    second: number;
    relation: string;
    proposal: string | null;
  }>;
  legal_actions: CoreAction[];
};

type ReplayMetadata = {
  seed: string;
  frames: number;
  engine_version: number;
  format_version: number;
  rules_profile: string;
};

type LiveConfig = {
  map: "duel" | "procedural";
  width: number;
  height: number;
  players: number;
  seed: string;
  landDensity: number;
};

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
const PLACEMENT_SEED_BASE = 9_140_003n;
const PLACEMENT_SEED_STEP = 104_729n;
const SEARCH_ELO = 1_000;
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
  width: WIDTH,
  height: HEIGHT,
  players: 2,
  seed: SEED.toString(),
  landDensity: 650_000,
};
const PLAYER_NAMES = ["CYAN", "AMBER", "VIOLET", "CORAL", "LIME", "BLUE", "PINK", "SILVER"] as const;
const MODEL_URL = "https://github.com/chelokot/antiyoy/releases/tag/model-v0.3.0-beta.1";
const MODEL_RESULTS = [
  ["Classic Generic", "48–0"],
  ["Online Default", "48–0"],
  ["Online Duel", "48–0"],
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
    return bindings.WasmGame.procedural(
      config.width,
      config.height,
      config.players,
      seed,
      config.landDensity,
    );
  }
  return new bindings.WasmGame(config.width, config.height, seed);
}

function advanceBotsUntilHuman(
  instance: WasmGameType,
  initial: StateView,
  humanSeat: number,
): { state: StateView; actions: number } {
  let state = initial;
  let actions = 0;
  while (!state.terminal && state.active_player !== humanSeat && actions < 2_000) {
    state = parseState(instance.step_bot());
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
    Capital: "●",
    Farm: "⌂",
    Tower: "◆",
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

export default function Arena() {
  const wasmModule = useRef<WasmModule | null>(null);
  const game = useRef<WasmGameType | null>(null);
  const replay = useRef<WasmReplayType | null>(null);
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
  const [humanMode, setHumanMode] = useState(false);
  const [humanSeat, setHumanSeat] = useState(0);
  const [placementMode, setPlacementMode] = useState(false);
  const [placement, setPlacement] = useState<PlacementRating>(INITIAL_PLACEMENT);
  const [draftConfig, setDraftConfig] = useState<LiveConfig>(DEFAULT_CONFIG);
  const [activeConfig, setActiveConfig] = useState<LiveConfig>(DEFAULT_CONFIG);
  const [boardScale, setBoardScale] = useState(1);
  const [openPanel, setOpenPanel] = useState<"left" | "right" | null>(null);

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

  const generate = useCallback(() => {
    const bindings = wasmModule.current;
    if (bindings === null) {
      return;
    }
    let candidate: WasmGameType | null = null;
    try {
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
      setActiveConfig(draftConfig);
      setState(next);
      setSelectedId(centerPlayableCell(next));
      setActions(0);
      setPlaying(false);
      setError(null);
    } catch (reason: unknown) {
      candidate?.free();
      setPlaying(false);
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [draftConfig]);

  const startPlacement = useCallback(() => {
    const bindings = wasmModule.current;
    if (bindings === null) {
      return;
    }
    const attempt = placement.attempts;
    const seat = attempt % 2;
    const placementConfig: LiveConfig = {
      map: "duel",
      width: WIDTH,
      height: HEIGHT,
      players: 2,
      seed: (PLACEMENT_SEED_BASE + BigInt(attempt) * PLACEMENT_SEED_STEP).toString(),
      landDensity: DEFAULT_CONFIG.landDensity,
    };
    let candidate: WasmGameType | null = null;
    try {
      candidate = createGame(bindings, placementConfig);
      const advanced = advanceBotsUntilHuman(
        candidate,
        parseState(candidate.state_json()),
        seat,
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
      setActiveConfig(placementConfig);
      setState(advanced.state);
      setSelectedId(centerPlayableCell(advanced.state));
      setActions(advanced.actions);
      setHumanSeat(seat);
      setHumanMode(true);
      setPlacementMode(true);
      setPlaying(false);
      setError(null);
    } catch (reason: unknown) {
      candidate?.free();
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [placement]);

  const step = useCallback(() => {
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
  }, [actions, replayMetadata]);

  useEffect(() => {
    if (!playing) {
      return;
    }
    const interval = window.setInterval(step, 180);
    return () => window.clearInterval(interval);
  }, [playing, step]);

  const reset = useCallback(() => {
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
    setPlaying(false);
    setError(null);
  }, []);

  const playHumanAction = useCallback((actionIndex: number) => {
    const instance = game.current;
    if (instance === null || replay.current !== null) {
      return;
    }
    try {
      const afterHuman = parseState(instance.step(actionIndex));
      const response = advanceBotsUntilHuman(instance, afterHuman, humanSeat);
      setState(response.state);
      setActions((current) => current + response.actions + 1);
      setPlaying(false);
      setError(null);
    } catch (reason: unknown) {
      setPlaying(false);
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, [humanSeat]);

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
    setPlacementMode(false);
    setHumanSeat(0);
    setHumanMode((current) => !current);
    reset();
  }, [reset]);

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
  }, []);

  const restoreLive = useCallback(() => {
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
  }, []);

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

  return (
    <main className="arena-shell bg-[#080b0d] text-[#e9eee9]">
      <header className="arena-header">
        <div className="flex items-center gap-4">
          <div className="brand-mark">AY</div>
          <div><p className="eyebrow">ARENA LAB</p><h1 className="text-sm font-semibold">Deterministic policy evaluation</h1></div>
        </div>
        <div className="flex items-center gap-3 font-mono text-xs"><span className="live-pill">● RUST WASM</span><span className="hidden text-[#8d9690] sm:inline">engine v{engineVersion ?? "…"}</span></div>
      </header>

      <div className="arena-layout">
        <aside className={`arena-sidebar arena-sidebar-left ${openPanel === "left" ? "panel-open" : ""}`}>
          <button className="panel-close" type="button" aria-label="Close overview panel" onClick={() => setOpenPanel(null)}>×</button>
          <details className="panel-section panel-section-hero" open>
            <summary>MATCH</summary>
            <div className="panel-section-body"><p className="eyebrow">{replayMetadata === null ? placementMode ? "RATED PLACEMENT" : humanMode ? "HUMAN VS SEARCH" : "LIVE SELF-PLAY" : "VERIFIED REPLAY"}</p><h2 className="mt-2 text-xl font-semibold">{replayMetadata === null ? humanMode ? `you are ${playerLabel(humanSeat).toLowerCase()}` : "greedy vs turn-search" : "training trace"}</h2><p className="mt-1 text-sm text-[#8d9690]">{replayMetadata === null ? placementMode ? "local Elo vs fixed search-2048" : humanMode ? "2048-node Rust agent" : "deterministic whole-turn planning" : `${replayMetadata.frames} deterministic actions`}</p><div className="mt-6 space-y-4"><Metric label="RULESET" value={replayMetadata?.rules_profile ?? "classic_generic_2022"} /><Metric label="MAP" value={replayMetadata === null ? activeConfig.map === "procedural" ? "procedural_v1" : "symmetric_duel_v1" : "replay scenario"} /><Metric label="SEED" value={`${replayMetadata?.seed ?? activeConfig.seed} · reproducible`} /><Metric label="ROUND" value={state === null ? "loading" : `${state.round} · ${playerLabel(state.active_player)} to move`} /><Metric label="LEGAL ACTIONS" value={state?.legal_actions.length.toString() ?? "…"} accent /></div></div>
          </details>
          {replayMetadata === null && <details className="panel-section" open><summary>MAP GENERATOR</summary><div className="panel-section-body map-config"><div className="config-grid mt-0"><label className="config-field"><span>MODE</span><select value={draftConfig.map} onChange={(event) => setDraftConfig((current) => ({ ...current, map: event.target.value as LiveConfig["map"], players: event.target.value === "duel" ? 2 : current.players }))}><option value="duel">Symmetric duel</option><option value="procedural">Procedural v1</option></select></label><label className="config-field"><span>SEED</span><input type="text" inputMode="numeric" pattern="[0-9]+" value={draftConfig.seed} onChange={(event) => setDraftConfig((current) => ({ ...current, seed: event.target.value }))} /></label><label className="config-field"><span>WIDTH</span><input type="number" min="5" max="41" value={draftConfig.width} onChange={(event) => setDraftConfig((current) => ({ ...current, width: Number(event.target.value) }))} /></label><label className="config-field"><span>HEIGHT</span><input type="number" min="2" max="31" value={draftConfig.height} onChange={(event) => setDraftConfig((current) => ({ ...current, height: Number(event.target.value) }))} /></label><label className="config-field"><span>PLAYERS</span><input type="number" min="2" max="8" disabled={draftConfig.map === "duel"} value={draftConfig.map === "duel" ? 2 : draftConfig.players} onChange={(event) => setDraftConfig((current) => ({ ...current, players: Number(event.target.value) }))} /></label><label className="config-field"><span>LAND PPM</span><input type="number" min="200000" max="1000000" step="50000" disabled={draftConfig.map === "duel"} value={draftConfig.map === "duel" ? 650000 : draftConfig.landDensity} onChange={(event) => setDraftConfig((current) => ({ ...current, landDensity: Number(event.target.value) }))} /></label></div><button className="generate-button" type="button" onClick={generate}>Generate deterministic map</button></div></details>}
          <details className="panel-section" open><summary>TERRITORY</summary><div className="panel-section-body space-y-3 text-xs">{territories.map((cells, player) => <Bar label={playerLabel(player)} value={cells} width={`${territoryShares[player]}%`} player={player} key={player} />)}</div></details>
          <details className="panel-section"><summary>ENGINE</summary><div className="panel-section-body"><div className="engine-note mt-0"><p className="eyebrow text-[#d8ff3e]">SAME CORE</p><p className="mt-2 text-sm leading-6 text-[#b8c0ba]">Every displayed transition is executed by the headless Rust environment compiled to WebAssembly.</p></div></div></details>
          <details className="panel-section"><summary>BETA POLICY · 336–0</summary><div className="panel-section-body model-card">
            <div className="flex items-center justify-between gap-3"><p className="eyebrow text-[#d8ff3e]">BETA POLICY</p><span className="font-mono text-[0.65rem] text-[#8d9690]">2 experts</span></div>
            <p className="mt-2 text-sm font-semibold">routed search-dagger · 1.83M</p>
            <p className="mt-3 font-mono text-[0.65rem] uppercase tracking-[0.12em] text-[#8d9690]">vs search-2048 · 336–0</p>
            <dl className="mt-4 space-y-2 font-mono text-xs">{MODEL_RESULTS.map(([profile, score]) => <Row label={profile} value={score} accent key={profile} />)}</dl>
            <a className="model-download" href={MODEL_URL} target="_blank" rel="noreferrer">Download verified bundle ↗</a>
            <p className="mt-3 text-[0.65rem] leading-5 text-[#77817b]">Seven profiles, three held-out seed windows, both seats: every profile finished 48–0 on the fixed 11×9 arena. This is an arena-specific benchmark, not an absolute rating.</p>
          </div></details>
          <details className="panel-section"><summary>YOUR ELO · {Math.round(placement.elo)}</summary><div className="panel-section-body model-card">
            <div className="flex items-center justify-between gap-3"><p className="eyebrow text-[#d8ff3e]">YOUR PLACEMENT</p><span className="font-mono text-[0.65rem] text-[#8d9690]">LOCAL</span></div>
            <p className="mt-2 font-mono text-3xl font-semibold text-[#d8ff3e]">{Math.round(placement.elo)}</p>
            <dl className="mt-4 space-y-2 font-mono text-xs"><Row label="Games" value={placement.games.toString()} /><Row label="Record" value={`${placement.wins}–${placement.draws}–${placement.losses}`} /><Row label="Opponent" value="search-2048" /></dl>
            <button className="generate-button" type="button" onClick={startPlacement}>{placementMode ? "Start next rated match" : "Start rated match"}</button>
            <p className="mt-3 text-[0.65rem] leading-5 text-[#77817b]">Fixed 11×9 Classic arena. New deterministic seed every attempt, alternating seats, provisional K=40 for ten completed games. Stored only in this browser.</p>
          </div></details>
        </aside>

        <section className="board-panel">
          <div className="board-controls"><button className="control panel-toggle" type="button" onClick={() => setOpenPanel("left")}>Overview</button><div className="turn-chip"><span className={`turn-dot territory-player-${(state?.active_player ?? 0) % PLAYER_NAMES.length}`} /><span className="turn-copy">ROUND {state?.round ?? "…"} · {state === null ? "LOADING" : `${playerLabel(state.active_player)} TO MOVE`}</span></div><button className="control control-primary" type="button" disabled={humanMode || state === null || state.terminal || (replayMetadata !== null && actions === replayMetadata.frames)} onClick={() => setPlaying((current) => !current)}>{playing ? "Ⅱ Pause" : "▶ Play"}</button><button className="control" type="button" disabled={humanMode || state === null || state.terminal || playing || (replayMetadata !== null && actions === replayMetadata.frames)} onClick={step}>Step</button><button className="control" type="button" disabled={state === null} onClick={reset}>Reset</button>{replayMetadata === null ? <><button className={`control ${humanMode ? "control-active" : ""}`} type="button" onClick={toggleHumanMode}>{humanMode ? "Human: on" : "Human: off"}</button><label className="control cursor-pointer">Load replay<input className="sr-only" type="file" accept=".antiyoy,application/octet-stream" onChange={(event) => void loadReplay(event.target.files?.[0])} /></label></> : <button className="control" type="button" onClick={restoreLive}>Live game</button>}<button className="control panel-toggle" type="button" onClick={() => setOpenPanel("right")}>Inspect</button></div>
          <div className="board-scroll" ref={boardViewport} aria-label="Interactive hex game board">
            <div className="board-transform" style={{ transform: `translate(-50%, -50%) scale(${boardScale})` }}>
              <div ref={boardContent} className={`hex-board ${state !== null && state.width > 15 ? "hex-board-compact" : ""}`}>
                {rows.map((row, rowIndex) => <div className="hex-row" key={rowIndex}>{row.map((cell) => <Hex cell={cell} selected={cell.id === selectedId} onSelect={setSelectedId} key={cell.id} />)}</div>)}
              </div>
            </div>
          </div>
          {state?.terminal && <div className="result-banner">{state.winner === null ? "DRAW" : `${playerLabel(state.winner)} WINS`} · {actions} ACTIONS</div>}
          {error !== null && <div className="error-banner">WASM ERROR · {error}</div>}
          <div className="timeline"><div className="flex items-center justify-between font-mono text-[0.65rem] text-[#8d9690]"><span>ACTION {actions}{replayMetadata === null ? "" : ` / ${replayMetadata.frames}`}</span><span>{state?.terminal ? "TERMINAL" : replayMetadata === null ? "DETERMINISTIC TRACE" : "REPLAY VERIFIED"}</span></div>{replayMetadata === null ? <div className="mt-3 flex h-1.5 overflow-hidden bg-white/10">{territoryShares.map((share, player) => <div className={`territory-player-${player % PLAYER_NAMES.length}`} style={{ width: `${share}%` }} key={player} />)}</div> : <input className="replay-scrubber" type="range" min="0" max={replayMetadata.frames} value={actions} aria-label="Replay action" onChange={(event) => seekReplay(Number(event.target.value))} />}</div>
        </section>

        <aside className={`arena-sidebar arena-sidebar-right ${openPanel === "right" ? "panel-open" : ""}`}>
          <button className="panel-close" type="button" aria-label="Close inspector panel" onClick={() => setOpenPanel(null)}>×</button>
          <details className="panel-section panel-section-hero" open><summary>SELECTED HEX · {String(selectedQ).padStart(2, "0")},{String(selectedR).padStart(2, "0")}</summary><div className="panel-section-body"><p className="font-mono text-lg">q: {String(selectedQ).padStart(2, "0")} · r: {String(selectedR).padStart(2, "0")}</p><div className="mt-4 grid grid-cols-2 gap-px bg-white/10"><Stat label="OWNER" value={selected?.owner === null || selected === null ? "NEUTRAL" : playerLabel(selected.owner)} /><Stat label="PIECE" value={selected === null ? "…" : pieceLabel(selected)} /><Stat label="DEFENSE" value={selected?.defense.toString() ?? "…"} /><Stat label="READY" value={selected?.strength === 0 ? "—" : selected?.ready ? "YES" : "NO"} /></div></div></details>
          <details className="panel-section" open><summary>PROVINCE ECONOMY</summary><div className="panel-section-body">{province === null ? <p className="text-sm leading-6 text-[#77817b]">This hex is not part of a connected province.</p> : <dl className="space-y-3 font-mono text-xs"><Row label="Treasury" value={`$${province.money}`} /><Row label="Hex income" value={`+${province.income}`} /><Row label="Upkeep" value={`−${province.upkeep}`} /><Row label="Next turn" value={`${province.profit >= 0 ? "+" : "−"}$${Math.abs(province.profit)}`} accent /></dl>}</div></details>
          {humanMode && replayMetadata === null && <details className="panel-section" open><summary>LEGAL ACTIONS · {selectedActions.length + globalActions.length}</summary><div className="panel-section-body"><p className="text-xs leading-5 text-[#77817b]">Select a destination hex, then choose an action. Other players answer automatically.</p><div className="mt-3 grid gap-2">{selectedActions.map(({ action, index }) => <button className="action-button" type="button" disabled={state?.active_player !== humanSeat || state?.terminal} onClick={() => playHumanAction(index)} key={index}>{actionLabel(action, state?.width ?? WIDTH)}</button>)}{selectedActions.length === 0 && <p className="font-mono text-[0.65rem] text-[#626b66]">No targeted action is legal on this hex.</p>}</div><div className="mt-4 grid gap-2">{globalActions.map(({ action, index }) => <button className="action-button action-button-global" type="button" disabled={state?.active_player !== humanSeat || state?.terminal} onClick={() => playHumanAction(index)} key={index}>{actionLabel(action, state?.width ?? WIDTH)}</button>)}</div></div></details>}
          <details className="panel-section"><summary>STATE CONTRACT</summary><div className="panel-section-body"><dl className="space-y-2 font-mono text-xs"><Row label="Cells" value={state?.cells.length.toString() ?? "…"} /><Row label="Provinces" value={state?.provinces.length.toString() ?? "…"} /><Row label="Relations" value={state?.relations.length.toString() ?? "…"} /><Row label="Terminal" value={state?.terminal ? "YES" : "NO"} /></dl></div></details>
        </aside>
        <button className={`panel-backdrop ${openPanel === null ? "" : "panel-backdrop-open"}`} type="button" aria-label="Close side panel" onClick={() => setOpenPanel(null)} />
      </div>
    </main>
  );
}

function Hex({ cell, selected, onSelect }: { cell: CellView; selected: boolean; onSelect: (id: number) => void }) {
  const owner = cell.owner === null ? "neutral" : `player-${cell.owner % PLAYER_NAMES.length}`;
  return <button className={`hex hex-${owner} ${cell.playable ? "" : "hex-void"} ${selected ? "hex-selected" : ""}`} type="button" disabled={!cell.playable} aria-label={cell.playable ? `Hex ${cell.id}, ${pieceLabel(cell)}` : `Inactive hex ${cell.id}`} onClick={() => onSelect(cell.id)}><span className={cell.strength > 0 ? "unit" : "piece"}>{pieceGlyph(cell)}</span></button>;
}

function Metric({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return <div><p className="eyebrow">{label}</p><p className={`mt-1 break-words font-mono text-sm ${accent ? "text-[#d8ff3e]" : ""}`}>{value}</p></div>;
}

function Bar({ label, value, width, player }: { label: string; value: number; width: string; player: number }) {
  return <div><div className="mb-1 flex justify-between"><span>{label}</span><span className="font-mono">{value}</span></div><div className="h-1 bg-white/10"><div className={`h-full territory-player-${player % PLAYER_NAMES.length}`} style={{ width }} /></div></div>;
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div className="bg-[#0d1215] p-3"><dt className="eyebrow">{label}</dt><dd className="mt-1 break-words font-mono text-xs">{value}</dd></div>;
}

function Row({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return <div className="flex justify-between gap-3"><dt className="text-[#8d9690]">{label}</dt><dd className={accent ? "text-[#d8ff3e]" : ""}>{value}</dd></div>;
}
