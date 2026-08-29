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

const WIDTH = 11;
const HEIGHT = 9;
const SEED = 47n;
const DEFAULT_CONFIG: LiveConfig = {
  map: "duel",
  width: WIDTH,
  height: HEIGHT,
  players: 2,
  seed: SEED.toString(),
  landDensity: 650_000,
};
const PLAYER_NAMES = ["CYAN", "AMBER", "VIOLET", "CORAL", "LIME", "BLUE", "PINK", "SILVER"] as const;
const MODEL_URL = "https://github.com/chelokot/antiyoy/releases/tag/model-v0.1.0-alpha.1";
const MODEL_RESULTS = [
  ["Classic Generic", "+229"],
  ["Online Default", "±0"],
  ["Online Duel", "+83"],
  ["Experimental v2", "−206"],
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
  const [state, setState] = useState<StateView | null>(null);
  const [selectedId, setSelectedId] = useState(Math.floor((WIDTH * HEIGHT) / 2));
  const [playing, setPlaying] = useState(false);
  const [actions, setActions] = useState(0);
  const [engineVersion, setEngineVersion] = useState<number | null>(null);
  const [replayMetadata, setReplayMetadata] = useState<ReplayMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [humanMode, setHumanMode] = useState(false);
  const [draftConfig, setDraftConfig] = useState<LiveConfig>(DEFAULT_CONFIG);
  const [activeConfig, setActiveConfig] = useState<LiveConfig>(DEFAULT_CONFIG);

  useEffect(() => {
    let disposed = false;
    void import("@/lib/antiyoy-wasm/antiyoy_wasm").then(async (module) => {
      await module.default();
      if (disposed) {
        return;
      }
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
      let next = parseState(instance.step(actionIndex));
      let advanced = 1;
      while (!next.terminal && next.active_player !== 0 && advanced < 2_000) {
        next = parseState(instance.step_bot());
        advanced += 1;
      }
      if (!next.terminal && next.active_player !== 0) {
        throw new Error("Bot response exceeded 2000 actions");
      }
      setState(next);
      setActions((current) => current + advanced);
      setPlaying(false);
      setError(null);
    } catch (reason: unknown) {
      setPlaying(false);
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }, []);

  const toggleHumanMode = useCallback(() => {
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
    <main className="min-h-screen bg-[#080b0d] text-[#e9eee9]">
      <header className="arena-header">
        <div className="flex items-center gap-4">
          <div className="brand-mark">AY</div>
          <div><p className="eyebrow">ARENA LAB</p><h1 className="text-sm font-semibold">Deterministic policy evaluation</h1></div>
        </div>
        <div className="flex items-center gap-3 font-mono text-xs"><span className="live-pill">● RUST WASM</span><span className="hidden text-[#8d9690] sm:inline">engine v{engineVersion ?? "…"}</span></div>
      </header>

      <div className="arena-layout">
        <aside className="arena-sidebar arena-sidebar-left">
          <p className="eyebrow">{replayMetadata === null ? humanMode ? "HUMAN VS BOTS" : "LIVE SELF-PLAY" : "VERIFIED REPLAY"}</p><h2 className="mt-2 text-xl font-semibold">{replayMetadata === null ? humanMode ? "you are cyan" : "greedy-baseline" : "training trace"}</h2><p className="mt-1 text-sm text-[#8d9690]">{replayMetadata === null ? humanMode ? "legal actions from Rust" : "versus seeded-random" : `${replayMetadata.frames} deterministic actions`}</p>
          <div className="mt-8 space-y-5"><Metric label="RULESET" value={replayMetadata?.rules_profile ?? "classic_generic_2022"} /><Metric label="MAP" value={replayMetadata === null ? activeConfig.map === "procedural" ? "procedural_v1" : "symmetric_duel_v1" : "replay scenario"} /><Metric label="SEED" value={`${replayMetadata?.seed ?? activeConfig.seed} · reproducible`} /><Metric label="ROUND" value={state === null ? "loading" : `${state.round} · ${playerLabel(state.active_player)} to move`} /><Metric label="LEGAL ACTIONS" value={state?.legal_actions.length.toString() ?? "…"} accent /></div>
          {replayMetadata === null && <div className="map-config"><p className="eyebrow text-[#d8ff3e]">MAP GENERATOR</p><div className="config-grid"><label className="config-field"><span>MODE</span><select value={draftConfig.map} onChange={(event) => setDraftConfig((current) => ({ ...current, map: event.target.value as LiveConfig["map"], players: event.target.value === "duel" ? 2 : current.players }))}><option value="duel">Symmetric duel</option><option value="procedural">Procedural v1</option></select></label><label className="config-field"><span>SEED</span><input type="text" inputMode="numeric" pattern="[0-9]+" value={draftConfig.seed} onChange={(event) => setDraftConfig((current) => ({ ...current, seed: event.target.value }))} /></label><label className="config-field"><span>WIDTH</span><input type="number" min="5" max="41" value={draftConfig.width} onChange={(event) => setDraftConfig((current) => ({ ...current, width: Number(event.target.value) }))} /></label><label className="config-field"><span>HEIGHT</span><input type="number" min="2" max="31" value={draftConfig.height} onChange={(event) => setDraftConfig((current) => ({ ...current, height: Number(event.target.value) }))} /></label><label className="config-field"><span>PLAYERS</span><input type="number" min="2" max="8" disabled={draftConfig.map === "duel"} value={draftConfig.map === "duel" ? 2 : draftConfig.players} onChange={(event) => setDraftConfig((current) => ({ ...current, players: Number(event.target.value) }))} /></label><label className="config-field"><span>LAND PPM</span><input type="number" min="200000" max="1000000" step="50000" disabled={draftConfig.map === "duel"} value={draftConfig.landDensity} onChange={(event) => setDraftConfig((current) => ({ ...current, landDensity: Number(event.target.value) }))} /></label></div><button className="generate-button" type="button" onClick={generate}>Generate deterministic map</button></div>}
          <div className="mt-8 border-t border-white/10 pt-5"><p className="eyebrow">TERRITORY</p><div className="mt-4 space-y-3 text-xs">{territories.map((cells, player) => <Bar label={playerLabel(player)} value={cells} width={`${territoryShares[player]}%`} player={player} key={player} />)}</div></div>
          <div className="engine-note"><p className="eyebrow text-[#d8ff3e]">SAME CORE</p><p className="mt-2 text-sm leading-6 text-[#b8c0ba]">Every displayed transition is executed by the headless Rust environment compiled to WebAssembly.</p></div>
          <div className="model-card">
            <div className="flex items-center justify-between gap-3"><p className="eyebrow text-[#d8ff3e]">ALPHA POLICY</p><span className="font-mono text-[0.65rem] text-[#8d9690]">128 games</span></div>
            <p className="mt-2 text-sm font-semibold">distilled-ppo · 0.91M</p>
            <dl className="mt-4 space-y-2 font-mono text-xs">{MODEL_RESULTS.map(([profile, elo]) => <Row label={profile} value={`${elo} Elo`} accent={elo.startsWith("+")} key={profile} />)}</dl>
            <a className="model-download" href={MODEL_URL} target="_blank" rel="noreferrer">Download verified checkpoint ↗</a>
            <p className="mt-3 text-[0.65rem] leading-5 text-[#77817b]">Relative to the native greedy baseline. Experimental, not an absolute rating.</p>
          </div>
        </aside>

        <section className="board-panel">
          <div className="board-controls"><button className="control control-primary" type="button" disabled={humanMode || state === null || state.terminal || (replayMetadata !== null && actions === replayMetadata.frames)} onClick={() => setPlaying((current) => !current)}>{playing ? "Ⅱ Pause" : "▶ Play"}</button><button className="control" type="button" disabled={humanMode || state === null || state.terminal || playing || (replayMetadata !== null && actions === replayMetadata.frames)} onClick={step}>Step</button><button className="control" type="button" disabled={state === null} onClick={reset}>Reset</button>{replayMetadata === null ? <><button className={`control ${humanMode ? "control-active" : ""}`} type="button" onClick={toggleHumanMode}>{humanMode ? "Human: on" : "Human: off"}</button><label className="control cursor-pointer">Load replay<input className="sr-only" type="file" accept=".antiyoy,application/octet-stream" onChange={(event) => void loadReplay(event.target.files?.[0])} /></label></> : <button className="control" type="button" onClick={restoreLive}>Live game</button>}</div>
          <div className="board-scroll" aria-label="Interactive hex game board">
            <div className={`hex-board ${state !== null && state.width > 15 ? "hex-board-compact" : ""}`}>
              {rows.map((row, rowIndex) => <div className="hex-row" key={rowIndex}>{row.map((cell) => <Hex cell={cell} selected={cell.id === selectedId} onSelect={setSelectedId} key={cell.id} />)}</div>)}
            </div>
          </div>
          {state?.terminal && <div className="result-banner">{state.winner === null ? "DRAW" : `${playerLabel(state.winner)} WINS`} · {actions} ACTIONS</div>}
          {error !== null && <div className="error-banner">WASM ERROR · {error}</div>}
          <div className="timeline"><div className="flex items-center justify-between font-mono text-[0.65rem] text-[#8d9690]"><span>ACTION {actions}{replayMetadata === null ? "" : ` / ${replayMetadata.frames}`}</span><span>{state?.terminal ? "TERMINAL" : replayMetadata === null ? "DETERMINISTIC TRACE" : "REPLAY VERIFIED"}</span></div>{replayMetadata === null ? <div className="mt-3 flex h-1.5 overflow-hidden bg-white/10">{territoryShares.map((share, player) => <div className={`territory-player-${player % PLAYER_NAMES.length}`} style={{ width: `${share}%` }} key={player} />)}</div> : <input className="replay-scrubber" type="range" min="0" max={replayMetadata.frames} value={actions} aria-label="Replay action" onChange={(event) => seekReplay(Number(event.target.value))} />}</div>
        </section>

        <aside className="arena-sidebar arena-sidebar-right">
          <p className="eyebrow">SELECTED HEX</p><p className="mt-2 font-mono text-lg">q: {String(selectedQ).padStart(2, "0")} · r: {String(selectedR).padStart(2, "0")}</p>
          <div className="mt-6 grid grid-cols-2 gap-px bg-white/10"><Stat label="OWNER" value={selected?.owner === null || selected === null ? "NEUTRAL" : playerLabel(selected.owner)} /><Stat label="PIECE" value={selected === null ? "…" : pieceLabel(selected)} /><Stat label="DEFENSE" value={selected?.defense.toString() ?? "…"} /><Stat label="READY" value={selected?.strength === 0 ? "—" : selected?.ready ? "YES" : "NO"} /></div>
          <div className="mt-8"><p className="eyebrow">PROVINCE ECONOMY</p>{province === null ? <p className="mt-4 text-sm leading-6 text-[#77817b]">This hex is not part of a connected province.</p> : <dl className="mt-4 space-y-3 font-mono text-xs"><Row label="Treasury" value={`$${province.money}`} /><Row label="Hex income" value={`+${province.income}`} /><Row label="Upkeep" value={`−${province.upkeep}`} /><Row label="Next turn" value={`${province.profit >= 0 ? "+" : "−"}$${Math.abs(province.profit)}`} accent /></dl>}</div>
          {humanMode && replayMetadata === null && <div className="human-actions"><p className="eyebrow text-[#d8ff3e]">LEGAL ACTIONS HERE</p><p className="mt-2 text-xs leading-5 text-[#77817b]">Select a destination hex, then choose an action. Other players answer automatically.</p><div className="mt-3 grid gap-2">{selectedActions.map(({ action, index }) => <button className="action-button" type="button" disabled={state?.active_player !== 0 || state?.terminal} onClick={() => playHumanAction(index)} key={index}>{actionLabel(action, state?.width ?? WIDTH)}</button>)}{selectedActions.length === 0 && <p className="font-mono text-[0.65rem] text-[#626b66]">No targeted action is legal on this hex.</p>}</div><div className="mt-4 grid gap-2">{globalActions.map(({ action, index }) => <button className="action-button action-button-global" type="button" disabled={state?.active_player !== 0 || state?.terminal} onClick={() => playHumanAction(index)} key={index}>{actionLabel(action, state?.width ?? WIDTH)}</button>)}</div></div>}
          <div className="mt-8 border border-white/10 p-4"><p className="eyebrow">STATE CONTRACT</p><dl className="mt-3 space-y-2 font-mono text-xs"><Row label="Cells" value={state?.cells.length.toString() ?? "…"} /><Row label="Provinces" value={state?.provinces.length.toString() ?? "…"} /><Row label="Relations" value={state?.relations.length.toString() ?? "…"} /><Row label="Terminal" value={state?.terminal ? "YES" : "NO"} /></dl></div>
        </aside>
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
