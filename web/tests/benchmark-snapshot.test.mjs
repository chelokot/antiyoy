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
  assert.equal(snapshot.comparisons.length, 11);
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
  const [snapshot, procedural, vector, outcomes, regret, actionQ] = await Promise.all([
    readJson(snapshotUrl),
    readJson(new URL("2026-08-31-procedural-5p-puct-loop-rocm.json", benchmarkRoot)),
    readJson(new URL("2026-08-31-one-pass-maxn-vector-distillation-rocm.json", benchmarkRoot)),
    readJson(new URL("2026-08-31-outcome-vector-value-rocm.json", benchmarkRoot)),
    readJson(new URL("2026-08-31-positive-regret-distillation-rocm.json", benchmarkRoot)),
    readJson(new URL("2026-08-31-replayable-action-q-distillation-rocm.json", benchmarkRoot)),
  ]);
  const rows = new Map(snapshot.comparisons.map((row) => [row.method, row]));

  assert.equal(
    rows.get("Ranking-value PUCT-8").relativeElo,
    `+${procedural.ranking_value_ablation.combined.baseline_adjusted_elo_delta.toFixed(2)}`,
  );
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
  assert.deepEqual(outcomes.combined.outcome_against_scalar, {
    outcome_better: 5,
    scalar_better: 5,
    same: 246,
    discordant: 10,
    net_improvements: 0,
    exact_two_sided_sign_test_p: 1,
  });
});
