/* tslint:disable */
/* eslint-disable */

export class WasmGame {
    free(): void;
    [Symbol.dispose](): void;
    legal_actions_json(): string;
    constructor(width: number, height: number, seed: bigint);
    reset(): string;
    state_json(): string;
    step(action_index: number): string;
    step_bot(): string;
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

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmgame_free: (a: number, b: number) => void;
    readonly wasmgame_new: (a: number, b: number, c: number, d: bigint) => void;
    readonly wasmgame_reset: (a: number, b: number) => void;
    readonly wasmgame_state_json: (a: number, b: number) => void;
    readonly wasmgame_legal_actions_json: (a: number, b: number) => void;
    readonly wasmgame_step: (a: number, b: number, c: number) => void;
    readonly wasmgame_step_bot: (a: number, b: number) => void;
    readonly __wbg_wasmreplay_free: (a: number, b: number) => void;
    readonly wasmreplay_new: (a: number, b: number, c: number) => void;
    readonly wasmreplay_frame_count: (a: number) => number;
    readonly wasmreplay_metadata_json: (a: number, b: number) => void;
    readonly wasmreplay_seek: (a: number, b: number, c: number) => void;
    readonly engine_version: () => number;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export2: (a: number, b: number) => number;
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
