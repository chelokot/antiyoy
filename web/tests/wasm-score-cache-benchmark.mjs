import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import init, { WasmGame } from "../lib/antiyoy-wasm/antiyoy_wasm.js";

const protocol = "benchmarks/protocols/2026-09-25-duel-wasm-exact-score-cache-runtime-v1.json";
const firstSeed = 6583000;
const maps = 8;
const actionLimit = 400;
const warmupActions = 4;

function median(values) {
  const ordered = [...values].sort((left, right) => left - right);
  const center = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 0
    ? (ordered[center - 1] + ordered[center]) / 2
    : ordered[center];
}

function percentile95(values) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(0.95 * ordered.length) - 1];
}

function replay(seed) {
  const plain = WasmGame.procedural_v2_with_profile(11, 9, 2, BigInt(seed), 650_000, "classic_generic_2022");
  const cached = WasmGame.procedural_v2_with_profile(11, 9, 2, BigInt(seed), 650_000, "classic_generic_2022");
  const decisions = [];
  try {
    for (let index = 0; index < actionLimit; index += 1) {
      const before = plain.state_json();
      assert.equal(cached.state_json(), before, `pre-action state ${seed}/${index}`);
      const activeSeat = JSON.parse(before).active_player;
      const orderedModes = (seed + index) % 2 === 0 ? ["plain", "cached"] : ["cached", "plain"];
      const result = {};
      const elapsed = {};
      for (const mode of orderedModes) {
        const start = performance.now();
        result[mode] = mode === "plain"
          ? plain.step_three_turn_search_replanned()
          : cached.step_three_turn_search_replanned_cached();
        elapsed[mode] = performance.now() - start;
      }
      assert.equal(result.cached, result.plain, `post-action state ${seed}/${index}`);
      decisions.push({ seat: activeSeat, plain_ms: elapsed.plain, cached_ms: elapsed.cached });
      if (JSON.parse(result.plain).terminal) {
        break;
      }
    }
    return {
      seed,
      actions: decisions.length,
      terminal: JSON.parse(plain.state_json()).terminal,
      cached_score_reuses: Number(cached.three_turn_search_cache_hits()),
      decisions,
    };
  } finally {
    plain.free();
    cached.free();
  }
}

function summarize(records) {
  const warm = records.map((record) => record.decisions.slice(warmupActions));
  const mapImprovement = warm.map((decisions) => 1 - decisions.reduce((total, decision) => total + decision.cached_ms, 0)
    / decisions.reduce((total, decision) => total + decision.plain_ms, 0));
  const seatImprovement = [0, 1].map((seat) => median(warm.map((decisions) => {
    const sameSeat = decisions.filter((decision) => decision.seat === seat);
    return 1 - sameSeat.reduce((total, decision) => total + decision.cached_ms, 0)
      / sameSeat.reduce((total, decision) => total + decision.plain_ms, 0);
  })));
  const plainWall = warm.flat().map((decision) => decision.plain_ms);
  const cachedWall = warm.flat().map((decision) => decision.cached_ms);
  const gate = {
    exact_bounded_traces: records.length === maps,
    score_cache_reused: records.reduce((total, record) => total + record.cached_score_reuses, 0) > 0,
    median_map_improvement_at_least_five_percent: median(mapImprovement) >= 0.05,
    both_seats_median_nonnegative: seatImprovement.every((value) => value >= 0),
    pooled_warm_p95_within_ten_percent: percentile95(cachedWall) <= 1.1 * percentile95(plainWall),
  };
  return {
    maps: records.length,
    actions_compared: records.reduce((total, record) => total + record.actions, 0),
    terminal_prefixes: records.filter((record) => record.terminal).length,
    nonterminal_400_action_prefixes: records.filter((record) => !record.terminal).length,
    cached_score_reuses: records.reduce((total, record) => total + record.cached_score_reuses, 0),
    median_paired_map_warm_wall_improvement: median(mapImprovement),
    median_warm_wall_improvement_by_seat: seatImprovement,
    pooled_warm_action_wall_ms: {
      plain_median: median(plainWall),
      cached_median: median(cachedWall),
      plain_p95: percentile95(plainWall),
      cached_p95: percentile95(cachedWall),
    },
    gate,
    feasibility_gate_passed: Object.values(gate).every(Boolean),
  };
}

await init({ module_or_path: await readFile(new URL("../lib/antiyoy-wasm/antiyoy_wasm_bg.wasm", import.meta.url)) });
const records = Array.from({ length: maps }, (_, index) => replay(firstSeed + index));
const report = {
  kind: "node_wasm_exact_score_cache_prefix_runtime",
  protocol,
  seed_first: firstSeed,
  independent_maps: maps,
  action_limit: actionLimit,
  warmup_actions_per_map: warmupActions,
  records,
  summary: summarize(records),
  qualification: "Node-hosted WASM bounded-prefix wall timing, not full-game strength or interactive browser latency",
};
if (process.argv[2]) {
  await writeFile(process.argv[2], `${JSON.stringify(report)}\n`);
  process.stdout.write(`${JSON.stringify(report.summary)}\n`);
} else {
  process.stdout.write(`${JSON.stringify(report)}\n`);
}
