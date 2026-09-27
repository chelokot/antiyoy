import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { advanceBotReply, advanceBotReplyCooperatively } from "../app/bot-reply";
import type { StateView } from "../app/game-types";
import initEngine, { WasmGame } from "../lib/antiyoy-wasm/antiyoy_wasm.js";

type State = { terminal: boolean; active_player: number; marker: number };

function replySteps() {
  const states: State[] = [
    { terminal: false, active_player: 1, marker: 1 },
    { terminal: false, active_player: 1, marker: 2 },
    { terminal: false, active_player: 0, marker: 3 },
  ];
  let index = 0;
  return () => states[index++];
}

test("cooperative bot reply preserves the complete atomic action sequence", async () => {
  const initial: State = { terminal: false, active_player: 1, marker: 0 };
  const synchronousTrace: number[] = [];
  const cooperativeTrace: number[] = [];
  const synchronousStep = replySteps();
  const cooperativeStep = replySteps();
  const synchronous = advanceBotReply(initial, 0, () => {
    const state = synchronousStep();
    synchronousTrace.push(state.marker);
    return state;
  });
  let yields = 0;
  const cooperative = await advanceBotReplyCooperatively(initial, 0, () => {
    const state = cooperativeStep();
    cooperativeTrace.push(state.marker);
    return state;
  }, async () => { yields += 1; }, () => true);

  assert.deepEqual(cooperative, synchronous);
  assert.deepEqual(cooperativeTrace, synchronousTrace);
  assert.equal(yields, 3);
});

test("cancelled cooperative reply does not take another action", async () => {
  const initial: State = { terminal: false, active_player: 1, marker: 0 };
  let current = true;
  let actions = 0;
  const response = await advanceBotReplyCooperatively(initial, 0, () => {
    actions += 1;
    return { terminal: false, active_player: 1, marker: actions };
  }, async () => { current = false; }, () => current);

  assert.equal(response, null);
  assert.equal(actions, 0);
});

test("cancellation between atomic actions preserves the completed prefix", async () => {
  const initial: State = { terminal: false, active_player: 1, marker: 0 };
  let current = true;
  let actions = 0;
  const response = await advanceBotReplyCooperatively(initial, 0, () => {
    actions += 1;
    return { terminal: false, active_player: 1, marker: actions };
  }, async () => {
    if (actions === 1) {
      current = false;
    }
  }, () => current);

  assert.equal(response, null);
  assert.equal(actions, 1);
});

test("bot action ceiling remains unchanged", () => {
  const initial: State = { terminal: false, active_player: 1, marker: 0 };
  assert.throws(() => advanceBotReply(initial, 0, () => initial), /exceeded 2000 actions/);
});

test("cached replanned WebAssembly actions remain exact across a complete reply", async () => {
  const bytes = await readFile("lib/antiyoy-wasm/antiyoy_wasm_bg.wasm");
  await initEngine({ module_or_path: bytes });
  const synchronousGame = WasmGame.with_profile(11, 9, 47n, "classic_generic_2022");
  const cooperativeGame = WasmGame.with_profile(11, 9, 47n, "classic_generic_2022");
  try {
    const initial = JSON.parse(synchronousGame.state_json()) as StateView;
    assert.equal(initial.active_player, 0);
    const synchronousTrace: string[] = [];
    const cooperativeTrace: string[] = [];
    const synchronous = advanceBotReply(initial, 1, () => {
      const serialized = synchronousGame.step_three_turn_search_replanned_cached();
      synchronousTrace.push(serialized);
      return JSON.parse(serialized) as StateView;
    });
    const cooperative = await advanceBotReplyCooperatively(initial, 1, () => {
      const serialized = cooperativeGame.step_three_turn_search_replanned_cached();
      cooperativeTrace.push(serialized);
      return JSON.parse(serialized) as StateView;
    }, async () => {}, () => true);

    assert.ok(synchronous.actions > 0);
    assert.deepEqual(cooperativeTrace, synchronousTrace);
    assert.deepEqual(cooperative, synchronous);
    assert.equal(cooperativeGame.state_json(), synchronousGame.state_json());
  } finally {
    synchronousGame.free();
    cooperativeGame.free();
  }
});
