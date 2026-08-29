#![forbid(unsafe_code)]

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn engine_version() -> u16 {
    antiyoy_core::ENGINE_VERSION
}
