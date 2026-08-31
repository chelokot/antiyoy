import type { Metadata } from "next";
import Link from "next/link";
import benchmarkData from "./benchmark-data.json";

export const metadata: Metadata = {
  title: "Model Arena · Antiyoy",
  description: "Reproducible Antiyoy agent ratings, search ladders, and amplification reports.",
};

const engineSixRatings = [
  {
    rank: 1,
    agent: "Routed v6 candidate",
    method: "41 experts · search-distilled policy",
    rating: 2131,
    delta: "+1131",
    record: "336–0–0",
    games: 336,
    evidence: "2026-08-31-engine-v6-amplify-distill-rocm.json",
  },
  {
    rank: 2,
    agent: "Routed v5 bundle on core v6",
    method: "38 experts · policy intuition",
    rating: 1446,
    delta: "+446",
    record: "312–0–24",
    games: 336,
    evidence: "2026-08-31-engine-v6-amplify-distill-rocm.json",
  },
  {
    rank: 3,
    agent: "Turn search 2048",
    method: "deterministic whole-turn beam search",
    rating: 1000,
    delta: "anchor",
    record: "—",
    games: 336,
    evidence: "2026-08-31-engine-v6-amplify-distill-rocm.json",
  },
] as const;

const ratingPools = [
  {
    name: "Exact-seat PUCT distillation",
    arena: "core v6 · classic Generic 11×9 · candidate fixed as seat 2",
    evidence: "2026-08-31-puct-seat1-soft-distillation-rocm.json",
    rows: [
      ["Soft-PUCT seat 2", "seat-routed · zero search", "1420", "+420 vs opponent", "470–0–42 / 512"],
      ["Source seat 2", "same frozen source policy", "—", "9.4% score", "48–0–464 / 512"],
      ["Frozen source seat 1", "common opponent · positional anchor", "1000", "anchor", "512 maps"],
    ],
  },
  {
    name: "PUCT amplification loop",
    arena: "core v6 · classic Generic 11×9 · frozen source policy",
    evidence: "2026-08-31-puct-soft-distillation-rocm.json",
    rows: [
      ["Soft-PUCT distilled", "seat-routed · zero search", "1034", "+34 vs source", "281–0–231 / 512"],
      ["Value-calibrated PUCT-8", "policy + search", "1033", "+33 vs direct", "280–0–232 / 512"],
      ["Direct routed v6", "policy intuition", "1000", "anchor", "frozen source"],
    ],
  },
  {
    name: "Search ladder",
    arena: "core v5 · fixed 11×9 · seven profiles",
    evidence: "2026-08-30-turn-search-i9-11900k.json",
    rows: [
      ["Turn search 4096", "search", "1434", "+434 vs greedy", "207–0–17 / 224"],
      ["Greedy", "heuristic", "1000", "anchor", "224 mirrored"],
    ],
  },
  {
    name: "Distillation history",
    arena: "core v5 · fixed 11×9 · versus search 2048",
    evidence: "2026-08-31-engine-v6-amplify-distill-rocm.json",
    rows: [
      ["Routed v0.3", "2 experts", "2131", "+1131", "336–0–0"],
      ["Search DAgger v0.2", "single policy", "1085", "+85", "138–2–84"],
      ["Turn search 2048", "teacher", "1000", "anchor", "224–336 games"],
    ],
  },
] as const;

const experiments = [
  {
    method: "Policy intuition",
    status: "measured",
    description: "One ONNX/PyTorch forward pass selects directly from the authoritative legal-action mask.",
    result: "336–0 against search 2048 for the accepted v6 route bundle.",
  },
  {
    method: "Whole-turn search",
    status: "measured",
    description: "Deterministic beam search plans an entire economic turn rather than one atomic move.",
    result: "+434 relative Elo over greedy at 4096 nodes in its fixed-duel pool.",
  },
  {
    method: "Policy-guided PUCT / MCTS",
    status: "measured",
    description: "Batched native PUCT uses policy priors, a frozen-policy calibrated value head, and a controlled policy/value root blend.",
    result: "PUCT-8: 280–232 over direct policy on 512 fresh paired games, +32.67 relative Elo; 95% score CI 50.36–58.95%.",
  },
  {
    method: "Amplify → distill",
    status: "measured",
    description: "Export the full PUCT root distribution, train a cheap policy, then route only a seat that passes its paired gate.",
    result: "Seat 1 student: 281–231 in a paired gate. Seat 2 student: 470–42 in its fixed-seat pool versus 48–464 for the source, +0.824 score and +419.54 relative Elo.",
  },
  {
    method: "Procedural multiplayer PUCT",
    status: "not promoted",
    description: "Calibrate one five-player seat, verify online PUCT twice, then distill only that route with strong KL retention.",
    result: "Binary-value PUCT repeated +2 wins but its student tied 19–19. Zero-sum lost 16–18. Ranking value leads 45–39 over three windows, but the 320-map paired sign test is p=0.377; none is promoted.",
  },
  {
    method: "Vector MaxN distillation",
    status: "runtime accepted",
    description: "Predict every player utility from one shared encoder pass while preserving the active player's exact scalar value.",
    result: "4.47× faster than exact MaxN and identical on 63/64 held-out map scores. The agent still regressed against direct policy, so only the runtime path was accepted.",
  },
  {
    method: "Outcome-grounded values",
    status: "not promoted",
    description: "Train relative utilities from complete self-play winners with game-disjoint validation instead of copying scalar predictions.",
    result: "Holdout sign accuracy rose from 51.7% to 85.5%, but outcome and scalar heads both finished 42–40 over direct policy and tied 5–5 head-to-head.",
  },
] as const;

function EvidenceLink({ file }: { file: string }) {
  return <a href={`https://github.com/chelokot/antiyoy/blob/main/benchmarks/${file}`} target="_blank" rel="noreferrer">raw ↗</a>;
}

function SnapshotEvidenceLink({ evidence }: { evidence: keyof typeof benchmarkData.evidence }) {
  const source = benchmarkData.evidence[evidence];
  return <a href={`https://github.com/chelokot/antiyoy/blob/main/benchmarks/${source.file}`} title={`SHA-256 ${source.sha256}`} target="_blank" rel="noreferrer">{source.sha256.slice(0, 7)} ↗</a>;
}

export default function ModelsPage() {
  return (
    <main className="models-page">
      <nav className="models-nav"><Link href="/">← Play</Link><strong>Antiyoy Model Arena</strong><a href="https://github.com/chelokot/antiyoy" target="_blank" rel="noreferrer">Repository ↗</a></nav>
      <header className="models-hero">
        <p>Reproducible agent ratings · engine v6</p>
        <h1>Who actually wins?</h1>
        <div className="models-lead"><p>Every number below names its opponent, engine, map family, seeds, seats, and sample count. Ratings from different pools are deliberately not merged.</p><dl><div><dt>Champion</dt><dd>Routed v6</dd></div><div><dt>Held-out</dt><dd>336–0</dd></div><div><dt>Relative Elo</dt><dd>+1131</dd></div></dl></div>
      </header>

      <section className="models-section">
        <div className="section-heading"><div><p>Pool 01</p><h2>Engine-v6 fixed duel</h2></div><p>Search-2048 is fixed at 1000. Same 11×9 map generator, seven rules profiles, paired seeds and opposite seats.</p></div>
        <div className="ranking-table-wrap"><table className="ranking-table"><thead><tr><th>#</th><th>Agent</th><th>Method</th><th>Rating</th><th>Δ</th><th>W–D–L</th><th>Games</th><th>Proof</th></tr></thead><tbody>{engineSixRatings.map((row) => <tr key={row.agent}><td>{row.rank}</td><th scope="row">{row.agent}</th><td>{row.method}</td><td className="rating-number">{row.rating}</td><td>{row.delta}</td><td>{row.record}</td><td>{row.games}</td><td><EvidenceLink file={row.evidence} /></td></tr>)}</tbody></table></div>
        <p className="method-note">A perfect finite sample uses the evaluator&apos;s edge correction. +1131 is an arena-relative estimate, not a claim about human Elo or unseen multiplayer maps.</p>
      </section>

      <section className="models-section pools-section">
        <div className="section-heading"><div><p>Pools 02–05</p><h2>Other controlled ladders</h2></div><p>Useful comparisons stay inside their own protocol. This avoids laundering incompatible measurements into one impressive-looking number.</p></div>
        <div className="pool-grid">{ratingPools.map((pool) => <article className="rating-pool" key={pool.name}><header><h3>{pool.name}</h3><p>{pool.arena}</p><EvidenceLink file={pool.evidence} /></header><table><tbody>{pool.rows.map(([agent, method, rating, delta, record]) => <tr key={agent}><th scope="row"><span>{agent}</span><small>{method}</small></th><td><strong>{rating}</strong><small>{delta}</small></td><td>{record}</td></tr>)}</tbody></table></article>)}</div>
        <p className="method-note"><Link href="/">Play either promoted Soft-PUCT seat now. Choose Cyan or Amber; the opponent runs locally through ONNX.</Link></p>
      </section>

      <section className="models-section method-matrix-section">
        <div className="section-heading"><div><p>Compute ledger</p><h2>What does search actually buy?</h2></div><p>Each row stays inside its named pool. Paired flips count maps changed by the method; p-values prevent a visually large Elo estimate from hiding a tiny discordant sample. Proof hashes are verified against the repository benchmark at build time.</p></div>
        <div className="method-matrix-wrap"><table className="method-matrix"><thead><tr><th>Pool</th><th>Agent / method</th><th>Compute</th><th>Opponent</th><th>W–D–L</th><th>Games</th><th>Relative Elo</th><th>Paired flips</th><th>Gate</th><th>Proof SHA</th></tr></thead><tbody>{benchmarkData.comparisons.map((row) => <tr key={`${row.pool}:${row.method}`}><td>{row.pool}</td><th scope="row">{row.method}</th><td>{row.compute}</td><td>{row.opponent}</td><td>{row.record}</td><td>{row.games}</td><td className="method-delta">{row.relativeElo}</td><td>{row.pairedFlips}<small>{row.significance}</small></td><td><span className={`verdict verdict-${row.verdict.replaceAll(" ", "-")}`}>{row.verdict}</span></td><td><SnapshotEvidenceLink evidence={row.evidence as keyof typeof benchmarkData.evidence} /></td></tr>)}</tbody></table></div>
        <p className="method-note">The table is a comparison index, not one global ladder. “Relative Elo” is local to the opponent and protocol named on that row.</p>
      </section>

      <section className="models-section">
        <div className="section-heading"><div><p>Research loop</p><h2>Intuition, amplification, distillation</h2></div><p>The loop is viable, but each arrow needs a paired outcome test. Imitation accuracy alone has already produced rejected regressions in this project.</p></div>
        <ol className="experiment-ledger">{experiments.map((experiment, index) => <li key={experiment.method}><span>{String(index + 1).padStart(2, "0")}</span><div><div className="experiment-title"><h3>{experiment.method}</h3><b>{experiment.status}</b></div><p>{experiment.description}</p><strong>{experiment.result}</strong></div></li>)}</ol>
      </section>

      <section className="models-section report-section">
        <div className="section-heading"><div><p>Promotion gate</p><h2>How a model earns the top row</h2></div></div>
        <div className="promotion-flow"><div><b>AMPLIFY</b><span>Search or PUCT labels policy-visited states.</span></div><i>→</i><div><b>DISTILL</b><span>A compact policy learns priors and values.</span></div><i>→</i><div><b>ATTACK</b><span>Fresh seeds, both seats, every rules profile.</span></div><i>→</i><div><b>PROMOTE</b><span>Only if the weakest slice does not regress.</span></div></div>
        <footer><span>Next controlled experiment</span><strong>Learn action-conditioned regret or advantage targets: better terminal-value prediction alone did not improve procedural MaxN decisions.</strong></footer>
      </section>
    </main>
  );
}
