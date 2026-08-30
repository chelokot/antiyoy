import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

import initEngine, { WasmGame } from "../lib/antiyoy-wasm/antiyoy_wasm.js";
import { RoutedBrowserPolicy } from "../app/browser-policy";


test("browser policy selects a legal action from the live Rust observation", async (context) => {
  const origin = process.env.NEURAL_POLICY_ORIGIN;
  assert.ok(origin, "NEURAL_POLICY_ORIGIN must point to a running production build");
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { location: { origin } },
  });
  const engineBytes = await readFile("lib/antiyoy-wasm/antiyoy_wasm_bg.wasm");
  await initEngine({ module_or_path: engineBytes });
  const game = WasmGame.with_profile(11, 9, 47n, "classic_generic_2022");
  const runtimeBaseUrl = pathToFileURL(
    `${resolve("node_modules/onnxruntime-web/dist")}/`,
  ).href;
  const policy = await RoutedBrowserPolicy.load(
    `${origin}/classic-generic-browser.onnx`,
    runtimeBaseUrl,
  );
  try {
    const decision = await policy.decide(game.policy_observation_json());
    const legalActions = JSON.parse(game.legal_actions_json()) as unknown[];
    assert.equal(decision.legalActions, legalActions.length);
    assert.ok(decision.actionIndex >= 0);
    assert.ok(decision.actionIndex < legalActions.length);
    assert.equal(decision.actionIndex, 6);
    assert.ok(decision.milliseconds > 0);
    assert.ok(decision.milliseconds < 2_000);
    context.diagnostic(
      `selected ${decision.actionIndex} from ${decision.legalActions} actions in ${decision.milliseconds.toFixed(1)} ms`,
    );
    assert.doesNotThrow(() => game.step(decision.actionIndex));
  } finally {
    policy.release();
    game.free();
  }
});
