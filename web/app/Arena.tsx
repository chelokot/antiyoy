"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { WasmGame as WasmGameType } from "@/lib/antiyoy-wasm/antiyoy_wasm";

type CellView = {
  id: number;
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

type StateView = {
  width: number;
  height: number;
  round: number;
  active_player: number;
  terminal: boolean;
  winner: number | null;
  cells: CellView[];
  provinces: ProvinceView[];
  legal_actions: number;
};

const WIDTH = 11;
const HEIGHT = 9;
const SEED = 47n;
const PLAYER_NAMES = ["CYAN", "AMBER"] as const;

function parseState(serialized: string): StateView {
  return JSON.parse(serialized) as StateView;
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

export default function Arena() {
  const game = useRef<WasmGameType | null>(null);
  const [state, setState] = useState<StateView | null>(null);
  const [selectedId, setSelectedId] = useState(Math.floor((WIDTH * HEIGHT) / 2));
  const [playing, setPlaying] = useState(false);
  const [actions, setActions] = useState(0);
  const [engineVersion, setEngineVersion] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    void import("@/lib/antiyoy-wasm/antiyoy_wasm").then(async (module) => {
      await module.default();
      if (disposed) {
        return;
      }
      const instance = new module.WasmGame(WIDTH, HEIGHT, SEED);
      game.current = instance;
      setEngineVersion(module.engine_version());
      setState(parseState(instance.state_json()));
    }).catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : String(reason));
    });
    return () => {
      disposed = true;
      game.current?.free();
      game.current = null;
    };
  }, []);

  const step = useCallback(() => {
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
  }, []);

  useEffect(() => {
    if (!playing) {
      return;
    }
    const interval = window.setInterval(step, 180);
    return () => window.clearInterval(interval);
  }, [playing, step]);

  const reset = useCallback(() => {
    if (game.current === null) {
      return;
    }
    setState(parseState(game.current.reset()));
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
  const selectedQ = selectedId % WIDTH;
  const selectedR = Math.floor(selectedId / WIDTH);
  const cyanCells = state?.cells.filter((cell) => cell.owner === 0).length ?? 0;
  const amberCells = state?.cells.filter((cell) => cell.owner === 1).length ?? 0;
  const controlledCells = cyanCells + amberCells;
  const cyanShare = controlledCells === 0 ? 50 : (cyanCells / controlledCells) * 100;

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
          <p className="eyebrow">LIVE SELF-PLAY</p><h2 className="mt-2 text-xl font-semibold">greedy-baseline</h2><p className="mt-1 text-sm text-[#8d9690]">versus seeded-random</p>
          <div className="mt-8 space-y-5"><Metric label="RULESET" value="classic_generic_2022" /><Metric label="SEED" value="47 · reproducible" /><Metric label="ROUND" value={state === null ? "loading" : `${state.round} · ${PLAYER_NAMES[state.active_player]} to move`} /><Metric label="LEGAL ACTIONS" value={state?.legal_actions.toString() ?? "…"} accent /></div>
          <div className="mt-8 border-t border-white/10 pt-5"><p className="eyebrow">TERRITORY</p><div className="mt-4 space-y-3 text-xs"><Bar label="Cyan" value={cyanCells} width={`${cyanShare}%`} kind="cyan" /><Bar label="Amber" value={amberCells} width={`${100 - cyanShare}%`} kind="amber" /></div></div>
          <div className="engine-note"><p className="eyebrow text-[#d8ff3e]">SAME CORE</p><p className="mt-2 text-sm leading-6 text-[#b8c0ba]">Every displayed transition is executed by the headless Rust environment compiled to WebAssembly.</p></div>
        </aside>

        <section className="board-panel">
          <div className="board-controls"><button className="control control-primary" type="button" disabled={state === null || state.terminal} onClick={() => setPlaying((current) => !current)}>{playing ? "Ⅱ Pause" : "▶ Play"}</button><button className="control" type="button" disabled={state === null || state.terminal || playing} onClick={step}>Step</button><button className="control" type="button" disabled={state === null} onClick={reset}>Reset</button></div>
          <div className="board-scroll" aria-label="Interactive hex game board">
            <div className="hex-board">
              {rows.map((row, rowIndex) => <div className="hex-row" key={rowIndex}>{row.map((cell) => <Hex cell={cell} selected={cell.id === selectedId} onSelect={setSelectedId} key={cell.id} />)}</div>)}
            </div>
          </div>
          {state?.terminal && <div className="result-banner">{state.winner === null ? "DRAW" : `${PLAYER_NAMES[state.winner]} WINS`} · {actions} ACTIONS</div>}
          {error !== null && <div className="error-banner">WASM ERROR · {error}</div>}
          <div className="timeline"><div className="flex items-center justify-between font-mono text-[0.65rem] text-[#8d9690]"><span>ACTION {actions}</span><span>{state?.terminal ? "TERMINAL" : "DETERMINISTIC TRACE"}</span></div><div className="mt-3 h-1.5 overflow-hidden bg-white/10"><div className="territory-progress" style={{ width: `${cyanShare}%` }} /></div></div>
        </section>

        <aside className="arena-sidebar arena-sidebar-right">
          <p className="eyebrow">SELECTED HEX</p><p className="mt-2 font-mono text-lg">q: {String(selectedQ).padStart(2, "0")} · r: {String(selectedR).padStart(2, "0")}</p>
          <div className="mt-6 grid grid-cols-2 gap-px bg-white/10"><Stat label="OWNER" value={selected?.owner === null || selected === null ? "NEUTRAL" : PLAYER_NAMES[selected.owner]} /><Stat label="PIECE" value={selected === null ? "…" : pieceLabel(selected)} /><Stat label="DEFENSE" value={selected?.defense.toString() ?? "…"} /><Stat label="READY" value={selected?.strength === 0 ? "—" : selected?.ready ? "YES" : "NO"} /></div>
          <div className="mt-8"><p className="eyebrow">PROVINCE ECONOMY</p>{province === null ? <p className="mt-4 text-sm leading-6 text-[#77817b]">This hex is not part of a connected province.</p> : <dl className="mt-4 space-y-3 font-mono text-xs"><Row label="Treasury" value={`$${province.money}`} /><Row label="Hex income" value={`+${province.income}`} /><Row label="Upkeep" value={`−${province.upkeep}`} /><Row label="Next turn" value={`${province.profit >= 0 ? "+" : "−"}$${Math.abs(province.profit)}`} accent /></dl>}</div>
          <div className="mt-8 border border-white/10 p-4"><p className="eyebrow">STATE CONTRACT</p><dl className="mt-3 space-y-2 font-mono text-xs"><Row label="Cells" value={state?.cells.length.toString() ?? "…"} /><Row label="Provinces" value={state?.provinces.length.toString() ?? "…"} /><Row label="Terminal" value={state?.terminal ? "YES" : "NO"} /></dl></div>
        </aside>
      </div>
    </main>
  );
}

function Hex({ cell, selected, onSelect }: { cell: CellView; selected: boolean; onSelect: (id: number) => void }) {
  const owner = cell.owner === 0 ? "cyan" : cell.owner === 1 ? "amber" : "neutral";
  return <button className={`hex hex-${owner} ${selected ? "hex-selected" : ""}`} type="button" aria-label={`Hex ${cell.id}, ${pieceLabel(cell)}`} onClick={() => onSelect(cell.id)}><span className={cell.strength > 0 ? "unit" : "piece"}>{pieceGlyph(cell)}</span></button>;
}

function Metric({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return <div><p className="eyebrow">{label}</p><p className={`mt-1 break-words font-mono text-sm ${accent ? "text-[#d8ff3e]" : ""}`}>{value}</p></div>;
}

function Bar({ label, value, width, kind }: { label: string; value: number; width: string; kind: "cyan" | "amber" }) {
  return <div><div className="mb-1 flex justify-between"><span>{label}</span><span className="font-mono">{value}</span></div><div className="h-1 bg-white/10"><div className={`h-full bar-${kind}`} style={{ width }} /></div></div>;
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div className="bg-[#0d1215] p-3"><dt className="eyebrow">{label}</dt><dd className="mt-1 break-words font-mono text-xs">{value}</dd></div>;
}

function Row({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return <div className="flex justify-between gap-3"><dt className="text-[#8d9690]">{label}</dt><dd className={accent ? "text-[#d8ff3e]" : ""}>{value}</dd></div>;
}
