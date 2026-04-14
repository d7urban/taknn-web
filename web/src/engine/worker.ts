/// Web Worker for running Tak search off the main thread.
///
/// Protocol:
///   Main → Worker: { type: "search", tps: string, komi: number, halfKomi: boolean, maxDepth: number, timeMs: number }
///   Worker → Main: { type: "result", ...SearchResultInfo } | { type: "error", message: string }

import { lookupOpeningBookMove } from "./book";
import { selectMoveWithNeural } from "./inference";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let wasmModule: any = null;

async function initWasm() {
  if (wasmModule) return wasmModule;
  // In a worker, self.location gives us the worker script URL.
  // The WASM files are in /wasm/ relative to the origin.
  const origin = self.location.origin;
  const wasmUrl = `${origin}/wasm/tak_wasm_bg.wasm`;
  const jsUrl = `${origin}/wasm/tak_wasm.js`;
  const mod = await (Function("url", "return import(url)")(jsUrl));
  await mod.default(wasmUrl);
  wasmModule = mod;
  return mod;
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;

  if (msg.type === "search") {
    try {
      const mod = await initWasm();
      const game = mod.TakGame.fromTps(msg.tps);
      game.setKomi(msg.komi ?? 0, msg.halfKomi ?? false);

      const bookResult = await lookupOpeningBookMove(game);
      if (bookResult) {
        self.postMessage({
          type: "result",
          searchResult: bookResult,
        });
        game.free();
        return;
      }

      const neuralResult = await selectMoveWithNeural(
        mod,
        game,
        msg.maxDepth || 20,
        msg.timeMs || 3000,
      );
      const result = neuralResult ?? {
        ...game.botMove(msg.maxDepth || 20, msg.timeMs || 3000),
        engineMode: "heuristic",
      };

      self.postMessage({
        type: "result",
        searchResult: result,
      });

      game.free();
    } catch (err: unknown) {
      // WASM panics log to console.error (via panic hook) then throw RuntimeError("unreachable").
      // Include any available details in the error message.
      const msg = err instanceof Error
        ? `${err.message}${err.stack ? '\n' + err.stack : ''}`
        : String(err);
      self.postMessage({
        type: "error",
        message: msg,
      });
    }
  }
};
