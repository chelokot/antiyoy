import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request(new URL(pathname, "http://localhost/"), { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the arena shell and metadata", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  assert.equal(response.headers.get("cache-control"), "no-cache, must-revalidate");
  const html = await response.text();
  assert.match(html, /<title>Antiyoy Arena Lab<\/title>/i);
  assert.match(html, /classic_generic_2022/);
  assert.match(html, /BETA POLICY/);
  assert.match(html, /vs search-2048/);
  assert.match(html, /core v6 · vs search-2048 · 336–0/);
  assert.match(html, /41 experts/);
  assert.match(html, /universal routed · 2–8 players/);
  assert.match(html, /Fresh held-out engine-v6 evaluation: 336 paired games/);
  assert.match(html, /no losses or action-limit adjudications/);
  assert.match(html, /Download verified bundle/);
  assert.match(html, /Compare agents and methods/);
  assert.match(html, /Replay/);
  assert.match(html, /GAME CONFIG/);
  assert.match(html, /Generate deterministic map/);
  assert.match(html, />Auto</);
  assert.match(html, /YOUR MOVE/);
  assert.match(html, /choose a unit or shop item/);
  assert.match(html, /YOUR PLACEMENT/);
  assert.match(html, /Start rated match/);
  assert.match(html, /Fixed 11×9 Classic arena/);
  assert.match(html, /Stored only in this browser/);
  assert.match(html, /ONLINE MULTIPLAYER/);
  assert.match(html, /Create 2-player room/);
  assert.match(html, /Play rated vs 1 server search bot/);
  assert.match(html, /seat rotates after every successful challenge/i);
  assert.match(html, /Join as seat 2/);
  assert.match(html, /ROOM SETTINGS/);
  assert.match(html, /AUTHORITATIVE SERVER/);
  assert.match(html, /SERVER LEAGUE/);
  assert.match(html, /REPLAY-VERIFIED ELO/);
  assert.match(html, /\/og\.png/);
  assert.doesNotMatch(html, /Your site is taking shape|react-loading-skeleton/);
});

test("ships the generated engine and social card", async () => {
  const [bindings, packageJson] = await Promise.all([
    readFile(new URL("../lib/antiyoy-wasm/antiyoy_wasm.js", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    access(new URL("../lib/antiyoy-wasm/antiyoy_wasm_bg.wasm", import.meta.url)),
    access(new URL("../public/og.png", import.meta.url)),
    access(new URL("../public/browser-primary.onnx", import.meta.url)),
    access(new URL("../public/browser-experimental-v2.onnx", import.meta.url)),
    access(new URL("../public/browser-online-default-seat0-v6.onnx", import.meta.url)),
    access(new URL("../public/browser-online-duel-seat0-v6.onnx", import.meta.url)),
    access(new URL("../public/browser-online-experimental-v1-seat0-v6.onnx", import.meta.url)),
    access(new URL("../public/ort-wasm-simd-threaded.wasm", import.meta.url)),
    access(new URL("../public/ort-wasm-simd-threaded.mjs", import.meta.url)),
    access(new URL("../public/game-pieces/unit-1.png", import.meta.url)),
    access(new URL("../public/game-pieces/capital.png", import.meta.url)),
    access(new URL("../public/game-pieces/selection.png", import.meta.url)),
  ]);
  assert.match(bindings, /class WasmGame/);
  assert.match(bindings, /class WasmReplay/);
  assert.match(bindings, /step_search\(\)/);
  assert.match(bindings, /step_search_with_budget\(node_budget\)/);
  assert.match(bindings, /search_node_budget\(\)/);
  assert.match(bindings, /search_nodes\(\)/);
  assert.match(bindings, /with_profile\(/);
  assert.match(bindings, /procedural_with_profile\(/);
  assert.match(bindings, /policy_observation_json\(\)/);
  assert.match(packageJson, /"name": "antiyoy-arena-lab"/);
  assert.match(packageJson, /"build:wasm"/);
  assert.match(packageJson, /"stage:wasm"/);
  await access(new URL("../public/antiyoy_wasm_bg.wasm", import.meta.url));
  await access(new URL("../dist/client/browser-primary.onnx", import.meta.url));
  await access(new URL("../dist/client/browser-experimental-v2.onnx", import.meta.url));
  await access(new URL("../dist/client/browser-online-default-seat0-v6.onnx", import.meta.url));
  await access(new URL("../dist/client/browser-online-duel-seat0-v6.onnx", import.meta.url));
  await access(new URL("../dist/client/browser-online-experimental-v1-seat0-v6.onnx", import.meta.url));
  await access(new URL("../dist/client/ort-wasm-simd-threaded.wasm", import.meta.url));
  await access(new URL("../dist/client/ort-wasm-simd-threaded.mjs", import.meta.url));
});

test("renders the benchmark-backed model arena", async () => {
  const response = await render("/models");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /Who actually wins\?/);
  assert.match(html, /Engine-v6 fixed duel/);
  assert.match(html, /Routed v6 candidate/);
  assert.match(html, /336–0–0/);
  assert.match(html, /Policy-guided PUCT \/ MCTS/);
  assert.match(html, /Value-calibrated PUCT-8/);
  assert.match(html, /Soft-PUCT distilled/);
  assert.match(html, /281–231 over its frozen source/);
  assert.match(html, /280–0–232/);
  assert.match(html, /\+32\.67 relative Elo/);
  assert.match(html, /Ratings from different pools are deliberately not merged/);
});

test("loads WebAssembly from a deployable same-origin URL", async () => {
  const arena = await readFile(new URL("../app/Arena.tsx", import.meta.url), "utf8");
  assert.match(arena, /new URL\("\/antiyoy_wasm_bg\.wasm", window\.location\.origin\)/);
  assert.doesNotMatch(arena, /await module\.default\(\);/);
  await access(new URL("../dist/client/antiyoy_wasm_bg.wasm", import.meta.url));
  const chunkDirectory = new URL("../dist/client/_next/static/chunks/", import.meta.url);
  const arenaChunk = (await readdir(chunkDirectory)).find((name) => name.startsWith("Arena-"));
  assert.ok(arenaChunk);
  const bundledArena = await readFile(new URL(arenaChunk, chunkDirectory), "utf8");
  assert.match(bundledArena, /antiyoy_wasm_bg\.wasm/);
  assert.match(bundledArena, /location\.origin/);
});

test("keeps the arena inside the viewport with independently scrolling panels", async () => {
  const [arena, styles] = await Promise.all([
    readFile(new URL("../app/Arena.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  assert.match(styles, /html, body \{[^}]*height: 100%;[^}]*overflow: hidden;[^}]*overscroll-behavior: none;/);
  assert.match(styles, /body > \* \{[^}]*height: 100%;[^}]*overflow: hidden;/);
  assert.match(styles, /\.arena-shell \{[^}]*position: fixed;[^}]*height: 100dvh;[^}]*overflow: hidden;/);
  assert.match(styles, /\.arena-layout \{[^}]*min-width: 0;[^}]*min-height: 0;[^}]*overflow: hidden;[^}]*grid-template-columns: minmax\(0, 1fr\);/);
  assert.match(styles, /\.arena-sidebar \{[^}]*position: absolute;[^}]*display: flex;[^}]*overflow: hidden;[^}]*overscroll-behavior: contain;/);
  assert.match(styles, /\.panel-scroll \{[^}]*min-height: 0;[^}]*overflow-x: hidden;[^}]*overflow-y: auto;/);
  assert.match(styles, /\.panel-section-body \{[^}]*max-height: min\(24rem, calc\(100dvh - 11rem\)\);[^}]*overflow-y: auto;[^}]*overscroll-behavior: contain;/);
  assert.match(styles, /\.arena-sidebar\.panel-open \{[^}]*transform: translateX\(0\);[^}]*visibility: visible;/);
  assert.doesNotMatch(styles, /@media \(min-width: 90rem\)/);
  assert.match(styles, /\.board-scroll \{[^}]*overflow: hidden;/);
  assert.match(styles, /\.board-scroll-human \{[^}]*bottom: 7\.25rem;/);
  assert.match(styles, /\.action-dock \{[^}]*position: absolute;/);
  assert.match(styles, /\.action-dock-buttons \{[^}]*overflow-x: auto;[^}]*overscroll-behavior-inline: contain;/);
  assert.match(styles, /clip-path: polygon\(25% 0,75% 0,100% 50%,75% 100%,25% 100%,0 50%\)/);
  assert.match(styles, /\.hex \{[^}]*width: 4\.625rem;[^}]*height: 4rem;[^}]*flex: 0 0 4\.625rem;[^}]*margin: 0 0 0 -1\.15625rem;[^}]*appearance: none;/);
  assert.match(styles, /\.hex:nth-child\(even\) \{ --hex-shift: 2rem; \}/);
  assert.match(styles, /\.panel-section > summary/);
  assert.match(arena, /className="panel-heading"/);
  assert.match(arena, /className="panel-scroll"/);
  assert.match(arena, /new ResizeObserver\(fit\)/);
  assert.match(arena, /event\.key === "Escape"/);
  assert.match(arena, /translate\(-50%, -50%\) scale\(\$\{boardScale\}\)/);
  assert.match(arena, /aria-controls="match-drawer"/);
  assert.match(arena, /aria-controls="inspector-drawer"/);
  assert.match(arena, /aria-label="Bot opponent"/);
  assert.match(arena, /resolveHexClick\(intentActions, movementSources, cellId\)/);
  assert.match(arena, /setActionIntent\(resolution\.intent\)/);
  assert.match(arena, /const shopProvince = province\?\.owner === economyPlayer \? province : null;/);
  assert.doesNotMatch(arena, /find\(\(candidate\) => candidate\.owner === economyPlayer\)/);
  assert.match(arena, /LEGAL TARGETS/);
  assert.doesNotMatch(arena, /pieceGlyph/);
  assert.doesNotMatch(arena, /Select a destination hex, then choose an action/);
  assert.match(arena, /Neural policy/);
  assert.match(arena, /Quick · 64/);
  assert.match(arena, /Strong · 256/);
  assert.match(arena, /Brutal · full turn/);
  assert.match(arena, /useState<BotOpponentName>\("neural"\)/);
  assert.match(arena, /placementMode\s+\? RATED_SEARCH_NODES/);
  assert.match(arena, /step_search_with_budget\(searchNodes\)/);
  assert.match(arena, /aria-label="Open game menu"/);
  assert.match(arena, /aria-label="Inspect selected hex"/);
  assert.match(arena, /<GamePiece cell=\{cell\} \/>/);
  assert.match(arena, /className="selection-ring"/);
  assert.match(arena, /className="move-target"/);
  assert.match(arena, /actionable=\{humanCanAct && actionableTargets\.has\(cell\.id\)\}/);
  assert.match(arena, /<summary>GAME CONFIG<\/summary>/);
  assert.match(arena, /RULES_PROFILES\.map/);
  assert.match(arena, /<summary>PROVINCE ECONOMY<\/summary>/);
  assert.match(arena, /createJoinableMatch\(onlineEndpoint, onlineName, draftConfig\)/);
  assert.match(arena, /createRatedBotChallenge\(/);
  assert.match(arena, /SERVER_CHALLENGE_ATTEMPT_STORAGE_KEY/);
  assert.match(arena, /createOpenSeatInvites\(inviteBaseUrl, onlineSession\.snapshot\)/);
  assert.match(arena, /Copy invite for seat/);
  assert.match(arena, /roomConfigFromSnapshot\(session\.snapshot\)/);
  assert.match(arena, /claimOpenSeat\(onlineEndpoint, joinCode, joinSeat, onlineName\)/);
  assert.match(arena, /multiplayerConnection\.current\?\.disconnect\(\)/);
  assert.match(arena, /connection\.submit\(action, onlineSession\.snapshot\.revision\)/);
  assert.match(arena, /fetchLeague\(onlineEndpoint\)/);
  assert.match(arena, /leagueStandings\(displayedLeague\)/);
  assert.match(arena, /snapshot\.rating_status === "Recorded"/);
  assert.match(arena, /className="league-standings"/);
  assert.match(arena, /VERIFIED LEDGER/);
  assert.match(arena, /duplicate replays rejected/);
  assert.match(styles, /\.league-standings \{[^}]*max-height: 13rem;[^}]*overflow-y: auto;/);
  assert.match(styles, /\.league-ledger \{[^}]*max-height: 13rem;[^}]*overflow-y: auto;/);
  assert.match(styles, /\.rated-challenge-button \{[^}]*background: var\(--acid\);/);
});

test("executes greedy and whole-turn search in the compiled WebAssembly engine", async () => {
  const moduleUrl = new URL("../lib/antiyoy-wasm/antiyoy_wasm.js", import.meta.url);
  moduleUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const bindings = await import(moduleUrl.href);
  const bytes = await readFile(new URL("../lib/antiyoy-wasm/antiyoy_wasm_bg.wasm", import.meta.url));
  await bindings.default({ module_or_path: bytes });
  const game = new bindings.WasmGame(11, 9, 47n);
  const classicRules = JSON.parse(bindings.rules_json_for_profile("classic_generic_2022"));
  const experimentalRules = JSON.parse(bindings.rules_json_for_profile("online_experimental_v2_260801"));
  assert.equal(classicRules.economy.unit_price_per_level, 10);
  assert.equal(classicRules.economy.farm_base_price, 12);
  assert.equal(experimentalRules.economy.farm_base_price, 8);
  const initial = JSON.parse(game.state_json());
  let next = initial;
  for (let action = 0; next.active_player === 0 && action < 100; action += 1) {
    next = JSON.parse(game.step_bot());
  }
  assert.equal(initial.cells.length, 99);
  assert.equal(initial.round, 1);
  assert.ok(initial.legal_actions.length > 0);
  const policyObservation = JSON.parse(game.policy_observation_json());
  assert.equal(policyObservation.widths[0], 11);
  assert.equal(policyObservation.heights[0], 9);
  assert.equal(policyObservation.rule_features.length, 45);
  assert.equal(policyObservation.actions.length, initial.legal_actions.length);
  assert.equal(next.active_player, 1);
  assert.notDeepEqual(next, initial);
  const searchReply = JSON.parse(game.step_bot());
  assert.notDeepEqual(searchReply, next);
  const quickSearch = new bindings.WasmGame(11, 9, 49n);
  const strongSearch = new bindings.WasmGame(11, 9, 49n);
  const brutalSearch = new bindings.WasmGame(11, 9, 49n);
  const beforeSearch = JSON.parse(brutalSearch.state_json());
  quickSearch.step_search_with_budget(64);
  strongSearch.step_search_with_budget(256);
  const afterSearch = JSON.parse(brutalSearch.step_search_with_budget(2048));
  assert.equal(beforeSearch.active_player, 0);
  assert.notDeepEqual(afterSearch, beforeSearch);
  assert.equal(quickSearch.search_nodes(), 64);
  assert.equal(strongSearch.search_nodes(), 256);
  assert.ok(brutalSearch.search_nodes() > strongSearch.search_nodes());
  assert.ok(brutalSearch.search_nodes() <= 2048);
  assert.equal(brutalSearch.search_count(), 1n);
  assert.throws(() => brutalSearch.step_search_with_budget(1), /at least two/);
  quickSearch.free();
  strongSearch.free();
  brutalSearch.free();
  const slay = bindings.WasmGame.with_profile(11, 9, 51n, "classic_slay_2022");
  assert.equal(slay.rules_profile(), "classic_slay_2022");
  slay.free();
  assert.throws(
    () => bindings.WasmGame.with_profile(11, 9, 51n, "unknown"),
    /unknown rules profile/,
  );
  const experimental = bindings.WasmGame.procedural_with_profile(
    17,
    13,
    4,
    53n,
    650_000,
    "online_experimental_v2_260801",
  );
  assert.equal(experimental.rules_profile(), "online_experimental_v2_260801");
  experimental.free();
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
  assert.equal(metadata.engine_version, 6);
  assert.equal(metadata.format_version, 5);
  assert.equal(metadata.frames, 24);
  assert.equal(replay.frame_count(), 24);
  assert.equal(initial.cells.length, 35);
  assert.equal(initial.relations.length, 4);
  assert.notDeepEqual(final, initial);
  assert.notDeepEqual(rewound, final);
  replay.free();
});
