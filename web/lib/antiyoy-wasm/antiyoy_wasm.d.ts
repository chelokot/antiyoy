/* tslint:disable */
/* eslint-disable */

export class WasmGame {
    free(): void;
    [Symbol.dispose](): void;
    legal_actions_json(): string;
    constructor(width: number, height: number, seed: bigint);
    policy_observation_json(): string;
    static procedural(width: number, height: number, players: number, seed: bigint, land_density_per_million: number): WasmGame;
    static procedural_with_profile(width: number, height: number, players: number, seed: bigint, land_density_per_million: number, profile: string): WasmGame;
    reset(): string;
    rules_profile(): string;
    search_count(): bigint;
    search_node_budget(): number;
    search_nodes(): number;
    state_json(): string;
    step(action_index: number): string;
    step_bot(): string;
    step_search(): string;
    step_search_with_budget(node_budget: number): string;
    static with_profile(width: number, height: number, seed: bigint, profile: string): WasmGame;
}

export class WasmReplay {
    free(): void;
    [Symbol.dispose](): void;
    frame_count(): number;
    metadata_json(): string;
    constructor(bytes: Uint8Array);
    seek(frame: number): string;
}

export function engine_version(): number;

export function rules_json_for_profile(profile: string): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmgame_free: (a: number, b: number) => void;
    readonly rules_json_for_profile: (a: number, b: number, c: number) => void;
    readonly wasmgame_new: (a: number, b: number, c: number, d: bigint) => void;
    readonly wasmgame_with_profile: (a: number, b: number, c: number, d: bigint, e: number, f: number) => void;
    readonly wasmgame_procedural: (a: number, b: number, c: number, d: number, e: bigint, f: number) => void;
    readonly wasmgame_procedural_with_profile: (a: number, b: number, c: number, d: number, e: bigint, f: number, g: number, h: number) => void;
    readonly wasmgame_rules_profile: (a: number, b: number) => void;
    readonly wasmgame_reset: (a: number, b: number) => void;
    readonly wasmgame_state_json: (a: number, b: number) => void;
    readonly wasmgame_legal_actions_json: (a: number, b: number) => void;
    readonly wasmgame_policy_observation_json: (a: number, b: number) => void;
    readonly wasmgame_step: (a: number, b: number, c: number) => void;
    readonly wasmgame_step_bot: (a: number, b: number) => void;
    readonly wasmgame_step_search: (a: number, b: number) => void;
    readonly wasmgame_step_search_with_budget: (a: number, b: number, c: number) => void;
    readonly wasmgame_search_node_budget: (a: number) => number;
    readonly wasmgame_search_nodes: (a: number) => number;
    readonly wasmgame_search_count: (a: number) => bigint;
    readonly __wbg_wasmreplay_free: (a: number, b: number) => void;
    readonly wasmreplay_new: (a: number, b: number, c: number) => void;
    readonly wasmreplay_frame_count: (a: number) => number;
    readonly wasmreplay_metadata_json: (a: number, b: number) => void;
    readonly wasmreplay_seek: (a: number, b: number, c: number) => void;
    readonly engine_version: () => number;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
