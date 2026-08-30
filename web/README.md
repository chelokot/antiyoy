# Antiyoy Arena Lab

Interactive inspection surface for the deterministic Antiyoy Rust engine. The
browser runs the same state transition code as headless evaluation through a
generated WebAssembly package. Its rated placement mode alternates seats on a
fixed arena and persists a device-local Elo against the 2048-node Rust search
agent. Live games expose all seven versioned rules profiles and deterministic
duel or two-to-eight-player procedural maps.

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

## Verify

```bash
npm run lint
npm test
```

The test suite builds the production worker, verifies server-rendered metadata,
and executes a real transition in the compiled WebAssembly engine.
