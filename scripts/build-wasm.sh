#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
wasm-pack build "$project_root/crates/antiyoy-wasm" \
  --target web \
  --out-dir "$project_root/web/lib/antiyoy-wasm" \
  --release
rm -f "$project_root/web/lib/antiyoy-wasm/.gitignore"
