"use client";

import { type KeyboardEvent, useRef, useState } from "react";

type Outcome = "B" | "0" | "1" | "N";

const outcomeLabels: Record<Outcome, string> = {
  B: "Search won from both seats",
  "0": "Search won only from the first seat",
  "1": "Search won only from the second seat",
  N: "Direct model won from both seats",
};

export default function PairedMapExplorer({ firstSeed, outcomes }: { firstSeed: number; outcomes: string }) {
  const cells = Array.from(outcomes) as Outcome[];
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const [selected, setSelected] = useState(0);
  const counts = cells.reduce<Record<Outcome, number>>((total, outcome) => {
    total[outcome] += 1;
    return total;
  }, { B: 0, "0": 0, "1": 0, N: 0 });

  function moveSelection(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const step = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -16, ArrowDown: 16 }[event.key];
    if (step === undefined) return;
    event.preventDefault();
    const next = Math.max(0, Math.min(cells.length - 1, index + step));
    setSelected(next);
    buttons.current[next]?.focus();
  }

  return (
    <div className="paired-map-explorer">
      <div className="paired-map-heading"><h3>Every paired map</h3><p>Each square is one procedural map, played twice with seats swapped. Seed order is not a timeline.</p></div>
      <div className="paired-map-grid" aria-label={`Paired outcomes for ${cells.length} independent maps`}>
        {cells.map((outcome, index) => (
          <button
            key={firstSeed + index}
            ref={(button) => { buttons.current[index] = button; }}
            type="button"
            className="paired-map-cell"
            data-outcome={outcome}
            aria-label={`Map seed ${firstSeed + index}: ${outcomeLabels[outcome]}`}
            aria-pressed={index === selected}
            tabIndex={index === selected ? 0 : -1}
            onClick={() => setSelected(index)}
            onKeyDown={(event) => moveSelection(event, index)}
          />
        ))}
      </div>
      <div className="paired-map-details">
        <label htmlFor="paired-map-range">Map seed {firstSeed + selected} <strong>{outcomeLabels[cells[selected]]}</strong></label>
        <input id="paired-map-range" type="range" min={0} max={cells.length - 1} value={selected} onChange={(event) => setSelected(Number(event.target.value))} />
      </div>
      <div className="paired-map-legend" aria-label="Map outcome totals">
        {(["B", "0", "1", "N"] as const).map((outcome) => <span key={outcome}><i data-outcome={outcome} aria-hidden="true" />{counts[outcome]} · {outcomeLabels[outcome]}</span>)}
      </div>
    </div>
  );
}
