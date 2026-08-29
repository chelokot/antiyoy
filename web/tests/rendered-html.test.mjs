import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the arena shell and metadata", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Antiyoy Arena Lab<\/title>/i);
  assert.match(html, /Deterministic policy evaluation/);
  assert.match(html, /RUST WASM/);
  assert.match(html, /classic_generic_2022/);
  assert.match(html, /ALPHA POLICY/);
  assert.match(html, /vs search/);
  assert.match(html, /Paired relative Elo/);
  assert.match(html, /Download verified checkpoint/);
  assert.match(html, /Load replay/);
  assert.match(html, /MAP GENERATOR/);
  assert.match(html, /Generate deterministic map/);
  assert.match(html, /Human: off/);
  assert.match(html, /\/og\.png/);
  assert.doesNotMatch(html, /Your site is taking shape|react-loading-skeleton/);
});

test("ships the generated engine and social card", async () => {
  const [bindings, packageJson] = await Promise.all([
    readFile(new URL("../lib/antiyoy-wasm/antiyoy_wasm.js", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    access(new URL("../lib/antiyoy-wasm/antiyoy_wasm_bg.wasm", import.meta.url)),
    access(new URL("../public/og.png", import.meta.url)),
  ]);
  assert.match(bindings, /class WasmGame/);
  assert.match(bindings, /class WasmReplay/);
  assert.match(packageJson, /"name": "antiyoy-arena-lab"/);
  assert.match(packageJson, /"build:wasm"/);
});

test("executes greedy and whole-turn search in the compiled WebAssembly engine", async () => {
  const moduleUrl = new URL("../lib/antiyoy-wasm/antiyoy_wasm.js", import.meta.url);
  moduleUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const bindings = await import(moduleUrl.href);
  const bytes = await readFile(new URL("../lib/antiyoy-wasm/antiyoy_wasm_bg.wasm", import.meta.url));
  await bindings.default({ module_or_path: bytes });
  const game = new bindings.WasmGame(11, 9, 47n);
  const initial = JSON.parse(game.state_json());
  let next = initial;
  for (let action = 0; next.active_player === 0 && action < 100; action += 1) {
    next = JSON.parse(game.step_bot());
  }
  assert.equal(initial.cells.length, 99);
  assert.equal(initial.round, 1);
  assert.ok(initial.legal_actions.length > 0);
  assert.equal(next.active_player, 1);
  assert.notDeepEqual(next, initial);
  const searchReply = JSON.parse(game.step_bot());
  assert.notDeepEqual(searchReply, next);
  game.free();
});

test("generates reproducible masked multiplayer maps in WebAssembly", async () => {
  const moduleUrl = new URL("../lib/antiyoy-wasm/antiyoy_wasm.js", import.meta.url);
  moduleUrl.searchParams.set("procedural-test", `${process.pid}-${Date.now()}`);
  const bindings = await import(moduleUrl.href);
  const bytes = await readFile(new URL("../lib/antiyoy-wasm/antiyoy_wasm_bg.wasm", import.meta.url));
  await bindings.default({ module_or_path: bytes });
  const first = bindings.WasmGame.procedural(17, 13, 4, 800n, 600_000);
  const second = bindings.WasmGame.procedural(17, 13, 4, 800n, 600_000);
  const other = bindings.WasmGame.procedural(17, 13, 4, 801n, 600_000);
  const firstState = JSON.parse(first.state_json());
  const secondState = JSON.parse(second.state_json());
  const otherState = JSON.parse(other.state_json());
  assert.deepEqual(firstState, secondState);
  assert.equal(firstState.cells.filter((cell) => cell.playable).length, 133);
  assert.deepEqual(
    [...new Set(firstState.cells.flatMap((cell) => cell.owner === null ? [] : [cell.owner]))]
      .sort((firstOwner, secondOwner) => firstOwner - secondOwner),
    [0, 1, 2, 3],
  );
  assert.notDeepEqual(
    firstState.cells.map((cell) => cell.playable),
    otherState.cells.map((cell) => cell.playable),
  );
  assert.deepEqual(JSON.parse(first.reset()), firstState);
  first.free();
  second.free();
  other.free();
});

test("verifies and seeks through a binary replay in WebAssembly", async () => {
  const moduleUrl = new URL("../lib/antiyoy-wasm/antiyoy_wasm.js", import.meta.url);
  moduleUrl.searchParams.set("replay-test", `${process.pid}-${Date.now()}`);
  const bindings = await import(moduleUrl.href);
  const [wasm, replayBytes] = await Promise.all([
    readFile(new URL("../lib/antiyoy-wasm/antiyoy_wasm_bg.wasm", import.meta.url)),
    readFile(new URL("fixtures/duel-24.antiyoy", import.meta.url)),
  ]);
  await bindings.default({ module_or_path: wasm });
  const replay = new bindings.WasmReplay(replayBytes);
  const metadata = JSON.parse(replay.metadata_json());
  const initial = JSON.parse(replay.seek(0));
  const final = JSON.parse(replay.seek(24));
  const rewound = JSON.parse(replay.seek(3));
  assert.equal(metadata.seed, "47");
  assert.equal(metadata.rules_profile, "classic_generic_2022");
  assert.equal(metadata.engine_version, 5);
  assert.equal(metadata.format_version, 5);
  assert.equal(metadata.frames, 24);
  assert.equal(replay.frame_count(), 24);
  assert.equal(initial.cells.length, 35);
  assert.equal(initial.relations.length, 4);
  assert.notDeepEqual(final, initial);
  assert.notDeepEqual(rewound, final);
  replay.free();
});
