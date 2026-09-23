import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

const snapshotUrl = new URL("../app/models/benchmark-data.json", import.meta.url);
const benchmarkRoot = new URL("../../benchmarks/", import.meta.url);

async function readJson(url) {
  return JSON.parse(await readFile(url, "utf8"));
}

test("model arena snapshot is bound to immutable benchmark contents", async () => {
  const snapshot = await readJson(snapshotUrl);
  const evidenceNames = new Set(Object.keys(snapshot.evidence));

  assert.equal(snapshot.schemaVersion, 1);
  assert.equal(snapshot.comparisons.length, 14);
  for (const [name, evidence] of Object.entries(snapshot.evidence)) {
    const contents = await readFile(new URL(evidence.file, benchmarkRoot));
    assert.equal(
      createHash("sha256").update(contents).digest("hex"),
      evidence.sha256,
      name,
    );
  }
  for (const comparison of snapshot.comparisons) {
    assert.ok(evidenceNames.has(comparison.evidence), comparison.method);
  }
});

test("model arena snapshot preserves the measured search and value gates", async () => {
  const [snapshot, procedural, vector, outcomes, regret, actionQ, actionSlate, cpuScout, counterfactual] = await Promise.all([
    readJson(snapshotUrl),
    readJson(new URL("2026-08-31-procedural-5p-puct-loop-rocm.json", benchmarkRoot)),
    readJson(new URL("2026-08-31-one-pass-maxn-vector-distillation-rocm.json", benchmarkRoot)),
    readJson(new URL("2026-08-31-outcome-vector-value-rocm.json", benchmarkRoot)),
    readJson(new URL("2026-08-31-positive-regret-distillation-rocm.json", benchmarkRoot)),
    readJson(new URL("2026-08-31-replayable-action-q-distillation-rocm.json", benchmarkRoot)),
    readJson(new URL("2026-08-31-conservative-action-slate-distillation-rocm.json", benchmarkRoot)),
    readJson(new URL("2026-09-23-seat4-ranking-value-cpu-scout.json", benchmarkRoot)),
    readJson(new URL("2026-09-23-counterfactual-fork-cpu-scout.json", benchmarkRoot)),
  ]);
  const rows = new Map(snapshot.comparisons.map((row) => [row.method, row]));

  assert.equal(
    rows.get("Ranking-value PUCT-8").relativeElo,
    `+${procedural.ranking_value_ablation.combined.baseline_adjusted_elo_delta.toFixed(2)}`,
  );
  assert.equal(rows.get("Ranking-value PUCT-8 · CPU scout").games, cpuScout.fresh_outcome_scout.games);
  assert.equal(rows.get("Ranking-value PUCT-8 · CPU scout").relativeElo, `+${cpuScout.fresh_outcome_scout.baseline_adjusted_elo_delta.toFixed(2)} (unstable)`);
  assert.equal(rows.get("Ranking-value PUCT-8 · CPU scout").pairedFlips, `${cpuScout.fresh_outcome_scout.candidate_better}–${cpuScout.fresh_outcome_scout.baseline_better}`);
  assert.equal(rows.get("Ranking-value PUCT-8 · CPU scout").significance, `p=${cpuScout.fresh_outcome_scout.exact_two_sided_sign_test_p.toFixed(3)} · 4 flips`);
  assert.equal(counterfactual.state_scout.frozen_policy_to_terminal.root_seat_wins_across_all_57_branches, 0);
  assert.equal(counterfactual.paired_single_intervention.fresh_gate.games, 64);
  assert.equal(counterfactual.paired_single_intervention.fresh_gate.direct_policy_seat_4_wins, 7);
  assert.equal(counterfactual.paired_single_intervention.fresh_gate.intervention_seat_4_wins, 8);
  assert.equal(counterfactual.paired_single_intervention.fresh_gate.exact_two_sided_sign_test_p, 1);
  assert.equal(
    rows.get("Exact MaxN PUCT-8").record,
    `${procedural.ranking_maxn_puct_ablation.combined.maxn_wins}–0–${128 - procedural.ranking_maxn_puct_ablation.combined.maxn_wins} vs ${procedural.ranking_maxn_puct_ablation.combined.source_policy_wins}–0–${128 - procedural.ranking_maxn_puct_ablation.combined.source_policy_wins}`,
  );
  assert.equal(
    rows.get("One-pass MaxN").relativeElo,
    vector.heldout_strength_gate.baseline_adjusted_elo_delta.toFixed(2).replace("-", "−"),
  );
  assert.equal(rows.get("Outcome-value MaxN").games, outcomes.combined.games);
  assert.equal(rows.get("Outcome-value MaxN").relativeElo, "+10.09");
  assert.equal(
    rows.get("Positive-regret distill").relativeElo,
    `+${regret.retention_8_candidate.combined.baseline_adjusted_elo_delta.toFixed(2)}`,
  );
  assert.equal(
    rows.get("Positive-regret distill").pairedFlips,
    `${regret.retention_8_candidate.combined.candidate_better}–${regret.retention_8_candidate.combined.baseline_better}`,
  );
  assert.equal(
    rows.get("Replayable action-Q shared").relativeElo,
    actionQ.selected_shared_head.combined_development_and_fresh.baseline_adjusted_elo_delta.toFixed(2).replace("-", "−"),
  );
  assert.equal(
    rows.get("Replayable action-Q exact-seat").pairedFlips,
    `${actionQ.seat_specific_heads.combined_development_and_fresh.candidate_better}–${actionQ.seat_specific_heads.combined_development_and_fresh.baseline_better}`,
  );
  for (const [index, scale] of [1, 2].entries()) {
    const candidate = actionSlate.candidates[index];
    const row = rows.get(`Full-slate conservative · scale ${scale}`);
    assert.equal(row.games, candidate.development.games);
    assert.equal(
      row.pairedFlips,
      `${candidate.development.candidate_better}–${candidate.development.baseline_better}`,
    );
    assert.equal(
      row.relativeElo,
      candidate.development.baseline_adjusted_elo_delta.toFixed(2).replace("-", "−"),
    );
  }
  assert.deepEqual(outcomes.combined.outcome_against_scalar, {
    outcome_better: 5,
    scalar_better: 5,
    same: 246,
    discordant: 10,
    net_improvements: 0,
    exact_two_sided_sign_test_p: 1,
  });
});

test("model arena seat audit stays separate from Elo and matches both reports", async () => {
  const [snapshot, bias, rotation, policyScout] = await Promise.all([
    readJson(snapshotUrl),
    readJson(new URL("2026-09-23-procedural-seat-bias-cpu.json", benchmarkRoot)),
    readJson(new URL("2026-09-23-rotated-seat-generator-cpu.json", benchmarkRoot)),
    readJson(new URL("2026-09-23-routed-v6-rotated-map-cpu-scout.json", benchmarkRoot)),
  ]);
  assert.equal(snapshot.seatAudit.mapsPerSchema, rotation.combined.maps_per_schema);
  assert.deepEqual(bias.controlled_start_rotation.combined_wins_by_original_start, [187, 172, 146, 87, 48]);
  for (const row of snapshot.seatAudit.rows) {
    const seat = row.seat;
    assert.equal(row.legacyWins, rotation.combined.schema_1.wins_by_seat[seat]);
    assert.equal(row.rotatedWins, rotation.combined.schema_2.wins_by_seat[seat]);
    assert.equal(row.legacyRegion, Number(rotation.combined.schema_1.initial_voronoi_region_mean_by_seat[seat].toFixed(2)));
    assert.equal(row.rotatedRegion, Number(rotation.combined.schema_2.initial_voronoi_region_mean_by_seat[seat].toFixed(2)));
  }
  assert.ok(snapshot.comparisons.every((row) => row.evidence !== "rotated-seat-generator"));
  assert.equal(policyScout.all_seats.games, 80);
  assert.equal(policyScout.all_seats.policy_wins, 17);
  assert.equal(policyScout.all_seats.greedy_self_play_reference_wins, 16);
  assert.equal(policyScout.all_seats.paired_better, 8);
  assert.equal(policyScout.all_seats.paired_worse, 7);
  assert.equal(policyScout.all_seats.exact_two_sided_sign_test_p, 1);
  assert.ok(snapshot.comparisons.every((row) => row.evidence !== "routed-rotated-scout"));
});
