// WASM module loader for the Tak engine.

export interface PieceInfo {
  color: "white" | "black";
  pieceType: "flat" | "wall" | "cap";
}

export interface SquareInfo {
  pieces: PieceInfo[];
  active: boolean;
}

export interface MoveInfo {
  index: number;
  ptn: string;
}

export interface GameInfo {
  size: number;
  ply: number;
  sideToMove: string;
  result: string;
  tps: string;
  komi: number;
  halfKomi: boolean;
  whiteStones: number;
  whiteCaps: number;
  blackStones: number;
  blackCaps: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let wasmModule: any = null;

export async function initWasm() {
  if (wasmModule) return wasmModule;
  // Cache-bust during development to avoid stale WASM.
  const v = typeof window !== "undefined" ? `?v=${Date.now()}` : "";
  const wasmUrl = `/wasm/tak_wasm_bg.wasm${v}`;
  const jsUrl = `/wasm/tak_wasm.js${v}`;
  // Use dynamic import with a variable to bypass TypeScript module resolution.
  const mod = await (Function('url', 'return import(url)')(jsUrl));
  await mod.default(wasmUrl);
  wasmModule = mod;
  return mod;
}
