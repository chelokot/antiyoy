# Antiyoy Arena Lab

Interactive inspection surface for the deterministic Antiyoy Rust engine. The
browser runs the same state transition code as headless evaluation through a
generated WebAssembly package. Its rated placement mode alternates seats on a
fixed arena and persists a device-local Elo against the 2048-node Rust search
agent. The default fixed Classic duel also runs the published neural expert
locally through ONNX Runtime Web. Two deduplicated neural exports cover all
seven fixed 11×9 duel profiles, while three explicit search budgets remain
available for every map. Live games expose all seven versioned rules profiles and
deterministic duel or two-to-eight-player procedural maps.
The Online Multiplayer panel creates a server-owned waiting room, copies a
token-free private invitation, atomically claims the guest seat, and streams
revisioned snapshots over WSS. The board renders the server's legal-action list
instead of reimplementing move validation in TypeScript.

[Open the hosted arena](https://antiyoy-arena-lab.chelokot.chatgpt.site)

## Run locally

From the repository root:

```bash
./scripts/build-wasm.sh
cd web
npm install
npm run dev
```

The arena opens at `http://localhost:3000`.

Run the authoritative server in another terminal:

```bash
cargo run --release -p antiyoy-server -- \
  --host 127.0.0.1 --port 8080 \
  --allowed-origin http://localhost:3000 \
  --data-directory server-data
```

Set `AUTHORITATIVE SERVER` to `http://127.0.0.1:8080`. The hosted private arena
defaults to the resource-limited TLS service at `https://antiyoy.test`.

## Verify

```bash
npm run lint
npm test
```

The test suite builds the production worker, verifies server-rendered metadata,
and executes a real transition in the compiled WebAssembly engine.

To smoke-test the real neural runtime against a running production build:

```bash
npm run start
NEURAL_POLICY_ORIGIN=http://127.0.0.1:3000 npm run test:neural:live
```

That test loads the shipped ONNX model and WebAssembly runtime, consumes a live
Rust observation, checks the exact first policy decision, and executes it
through the legal-action index.

To verify two genuinely separate clients against a running server:

```bash
ANTIYOY_MULTIPLAYER_ENDPOINT=http://127.0.0.1:8080 \
  npm run test:multiplayer:live
```

The test creates and claims a waiting room, authenticates both WebSockets,
plays one complete round, compares both broadcasts with the authoritative HTTP
snapshot, downloads the replay, checks the CORS preflight, and deletes the room.
