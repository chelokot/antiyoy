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
  assert.match(html, /Download verified checkpoint/);
  assert.match(html, /Load replay/);
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

test("executes a deterministic transition in the compiled WebAssembly engine", async () => {
  const moduleUrl = new URL("../lib/antiyoy-wasm/antiyoy_wasm.js", import.meta.url);
  moduleUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const bindings = await import(moduleUrl.href);
  const bytes = await readFile(new URL("../lib/antiyoy-wasm/antiyoy_wasm_bg.wasm", import.meta.url));
  await bindings.default({ module_or_path: bytes });
  const game = new bindings.WasmGame(11, 9, 47n);
  const initial = JSON.parse(game.state_json());
  const next = JSON.parse(game.step_bot());
  assert.equal(initial.cells.length, 99);
  assert.equal(initial.round, 1);
  assert.ok(initial.legal_actions > 0);
  assert.notDeepEqual(next, initial);
  game.free();
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
  assert.equal(metadata.frames, 24);
  assert.equal(replay.frame_count(), 24);
  assert.equal(initial.cells.length, 35);
  assert.notDeepEqual(final, initial);
  assert.notDeepEqual(rewound, final);
  replay.free();
});
