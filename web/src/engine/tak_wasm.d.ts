/* tslint:disable */
/* eslint-disable */

export class NeuralPolicy {
    free(): void;
    [Symbol.dispose](): void;
    constructor(buffer: Uint8Array);
    /**
     * Evaluates the policy MLP for the given move descriptors using the spatial and global pools.
     * `spatial_pool`: [64, 8, 8] (or similar) from the ONNX trunk.
     * `global_pool`: [64] from the ONNX trunk.
     */
    scoreMoves(game: TakGame, spatial_pool: Float32Array, global_pool: Float32Array): Float32Array;
}

export class TakGame {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Apply a move by its index in the legal moves list.
     */
    applyMoveIndex(index: number): void;
    /**
     * Apply a move by PTN string.
     */
    applyMovePtn(ptn_str: string): void;
    /**
     * Run search and apply the best move. Returns search info.
     */
    botMove(max_depth: number, time_ms: number): any;
    /**
     * Encode the current position as the NN board tensor.
     */
    encodeBoard(): Float32Array;
    /**
     * Create a game from a TPS string.
     */
    static fromTps(tps_str: string): TakGame;
    /**
     * Get the board state as a flat array for rendering.
     * Returns a JSON array of 64 square objects (8x8 grid, row-major).
     * Each square: { pieces: [{color, type}], active: bool }
     */
    getBoard(): any;
    /**
     * Get game info as a JSON object.
     */
    getInfo(): any;
    /**
     * Get the move history as PTN.
     */
    getMoveHistory(): string;
    /**
     * Get TPS string for the current position.
     */
    getTps(): string;
    /**
     * Check if the game is over.
     */
    isGameOver(): boolean;
    /**
     * Get list of legal moves as JSON array of {index, ptn}.
     */
    legalMoves(): any;
    /**
     * Create a new game with the given board size (3-8).
     */
    constructor(size: number);
    /**
     * Get current ply.
     */
    ply(): number;
    /**
     * Run heuristic search and return the best move + info.
     * `max_depth` and `time_ms` control search limits.
     */
    searchMove(max_depth: number, time_ms: number): any;
    /**
     * Get the board size.
     */
    size(): number;
    /**
     * Get the size-id expected by the NN trunk input (3x3 -> 0, ..., 8x8 -> 5).
     */
    sizeId(): number;
    /**
     * Undo the last move. Returns false if no moves to undo.
     */
    undo(): boolean;
}

/**
 * Install panic hook so WASM panics produce readable console errors.
 */
export function init(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_neuralpolicy_free: (a: number, b: number) => void;
    readonly __wbg_takgame_free: (a: number, b: number) => void;
    readonly neuralpolicy_new: (a: number, b: number) => [number, number, number];
    readonly neuralpolicy_scoreMoves: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly takgame_applyMoveIndex: (a: number, b: number) => [number, number];
    readonly takgame_applyMovePtn: (a: number, b: number, c: number) => [number, number];
    readonly takgame_botMove: (a: number, b: number, c: number) => [number, number, number];
    readonly takgame_encodeBoard: (a: number) => any;
    readonly takgame_fromTps: (a: number, b: number) => [number, number, number];
    readonly takgame_getBoard: (a: number) => [number, number, number];
    readonly takgame_getInfo: (a: number) => [number, number, number];
    readonly takgame_getMoveHistory: (a: number) => [number, number];
    readonly takgame_getTps: (a: number) => [number, number];
    readonly takgame_isGameOver: (a: number) => number;
    readonly takgame_legalMoves: (a: number) => [number, number, number];
    readonly takgame_new: (a: number) => [number, number, number];
    readonly takgame_ply: (a: number) => number;
    readonly takgame_searchMove: (a: number, b: number, c: number) => [number, number, number];
    readonly takgame_size: (a: number) => number;
    readonly takgame_sizeId: (a: number) => number;
    readonly takgame_undo: (a: number) => number;
    readonly init: () => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
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
