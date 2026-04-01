"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Board from "@/components/Board";
import Controls from "@/components/Controls";
import {
  initWasm,
  SquareInfo,
  MoveInfo,
  GameInfo,
  PieceInfo,
} from "@/engine/wasm";

// TakGame class from WASM.
type TakGame = {
  getBoard(): SquareInfo[];
  legalMoves(): MoveInfo[];
  getInfo(): GameInfo;
  getTps(): string;
  applyMoveIndex(index: number): void;
  applyMovePtn(ptn: string): void;
  undo(): boolean;
  isGameOver(): boolean;
  free(): void;
};

type WasmModule = {
  TakGame: {
    new(size: number): TakGame;
  };
};

export type SearchInfo = {
  bestMove: string;
  score: number;
  depth: number;
  nodes: number;
  pv: string[];
  ttHits: number;
};

export type SpreadState = {
  sourceIdx: number;
  sourceLabel: string;
  pickupCount: number;
  direction: string | null;
  drops: number[];
  piecesRemaining: number;
  carriedPieces: PieceInfo[];
};

function idxToLabel(idx: number): string {
  return String.fromCharCode(97 + (idx % 8)) + (Math.floor(idx / 8) + 1);
}

function directionFromIdxs(srcIdx: number, dstIdx: number): string | null {
  const srcCol = srcIdx % 8;
  const srcRow = Math.floor(srcIdx / 8);
  const dstCol = dstIdx % 8;
  const dstRow = Math.floor(dstIdx / 8);
  if (dstCol === srcCol && dstRow > srcRow) return "+";
  if (dstCol === srcCol && dstRow < srcRow) return "-";
  if (dstRow === srcRow && dstCol > srcCol) return ">";
  if (dstRow === srcRow && dstCol < srcCol) return "<";
  return null;
}

function adjacentIdx(idx: number, dir: string, boardSize: number): number | null {
  const col = idx % 8;
  const row = Math.floor(idx / 8);
  let nr = row, nc = col;
  if (dir === "+") nr++;
  else if (dir === "-") nr--;
  else if (dir === ">") nc++;
  else if (dir === "<") nc--;
  if (nr < 0 || nr >= boardSize || nc < 0 || nc >= boardSize) return null;
  return nr * 8 + nc;
}

function buildSpreadPtn(spread: SpreadState): string {
  let ptn = "";
  if (spread.pickupCount > 1) ptn += spread.pickupCount.toString();
  ptn += spread.sourceLabel;
  ptn += spread.direction;
  if (spread.drops.length > 1 || (spread.drops.length === 1 && spread.drops[0] !== spread.pickupCount)) {
    ptn += spread.drops.map((d) => d.toString()).join("");
  }
  return ptn;
}

export default function Home() {
  const [wasm, setWasm] = useState<WasmModule | null>(null);
  const gameRef = useRef<TakGame | null>(null);
  const [squares, setSquares] = useState<SquareInfo[]>([]);
  const [legalMoves, setLegalMoves] = useState<MoveInfo[]>([]);
  const [gameInfo, setGameInfo] = useState<GameInfo | null>(null);
  const [placementType, setPlacementType] = useState<"flat" | "wall" | "cap">("flat");
  const [loading, setLoading] = useState(true);

  // Interaction state: idle, stackSelected (clicked a stack), or spreading (building a spread move).
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [spreadState, setSpreadState] = useState<SpreadState | null>(null);

  // Bot state.
  const [botThinking, setBotThinking] = useState(false);
  const [lastSearchInfo, setLastSearchInfo] = useState<SearchInfo | null>(null);
  const workerRef = useRef<Worker | null>(null);

  const resetInteraction = useCallback(() => {
    setSelectedSquare(null);
    setSpreadState(null);
  }, []);

  // Initialize WASM.
  useEffect(() => {
    initWasm().then((mod) => {
      setWasm(mod as unknown as WasmModule);
      setLoading(false);
    });
  }, []);

  // Initialize Web Worker.
  useEffect(() => {
    const worker = new Worker(
      new URL("../engine/worker.ts", import.meta.url)
    );
    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      setBotThinking(false);
      if (msg.type === "result") {
        setLastSearchInfo(msg.searchResult);
        // The worker applied the move on its copy. We need to apply it on ours.
        // Use the PTN from the search result to apply on our game instance.
        const game = gameRef.current;
        if (game && msg.searchResult?.bestMove) {
          try {
            game.applyMovePtn(msg.searchResult.bestMove);
            // Refresh from our authoritative game state.
            setSquares(game.getBoard());
            setGameInfo(game.getInfo());
            if (!game.isGameOver()) {
              setLegalMoves(game.legalMoves());
            } else {
              setLegalMoves([]);
            }
          } catch (err) {
            console.error("Failed to apply bot move:", err);
          }
        }
      } else if (msg.type === "error") {
        console.error("Worker error:", msg.message);
      }
    };
    workerRef.current = worker;
    return () => worker.terminate();
  }, []);

  const refreshState = useCallback(() => {
    const game = gameRef.current;
    if (!game) return;
    setSquares(game.getBoard());
    setGameInfo(game.getInfo());
    if (!game.isGameOver()) {
      setLegalMoves(game.legalMoves());
    } else {
      setLegalMoves([]);
    }
    resetInteraction();
  }, [resetInteraction]);

  // Start a new game.
  const startGame = useCallback(
    (size: number) => {
      if (!wasm) return;
      gameRef.current?.free();
      gameRef.current = new wasm.TakGame(size);
      setPlacementType("flat");
      refreshState();
    },
    [wasm, refreshState]
  );

  // Auto-start a 6x6 game.
  useEffect(() => {
    if (wasm && !gameRef.current) {
      startGame(6);
    }
  }, [wasm, startGame]);

  const handleMoveSelect = useCallback(
    (index: number) => {
      const game = gameRef.current;
      if (!game) return;
      try {
        game.applyMoveIndex(index);
        refreshState();
      } catch (e) {
        console.error("Move failed:", e);
      }
    },
    [refreshState]
  );

  const handleUndo = useCallback(() => {
    const game = gameRef.current;
    if (!game) return;
    game.undo();
    refreshState();
  }, [refreshState]);

  const handleBotMove = useCallback(() => {
    const game = gameRef.current;
    const worker = workerRef.current;
    if (!game || !worker || game.isGameOver() || botThinking) return;
    const tps = game.getTps();
    setBotThinking(true);
    resetInteraction();
    worker.postMessage({ type: "search", tps, maxDepth: 20, timeMs: 3000 });
  }, [botThinking, resetInteraction]);

  // Escape key cancels interaction.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") resetInteraction();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [resetInteraction]);

  // Called from Controls when user clicks a piece in the stack to set pickup count.
  const handlePieceClick = useCallback(
    (pickupCount: number) => {
      if (selectedSquare === null) return;
      const sq = squares[selectedSquare];
      if (!sq || sq.pieces.length === 0) return;
      const carried = sq.pieces.slice(sq.pieces.length - pickupCount);
      setSpreadState({
        sourceIdx: selectedSquare,
        sourceLabel: idxToLabel(selectedSquare),
        pickupCount,
        direction: null,
        drops: [],
        piecesRemaining: pickupCount,
        carriedPieces: carried,
      });
    },
    [selectedSquare, squares]
  );

  // Called from Controls to confirm a completed spread.
  const handleConfirmSpread = useCallback(() => {
    if (!spreadState || spreadState.piecesRemaining > 0 || !spreadState.direction) return;
    const ptn = buildSpreadPtn(spreadState);
    const move = legalMoves.find((m) => m.ptn === ptn);
    if (move) {
      handleMoveSelect(move.index);
    } else {
      console.error("Spread move not legal:", ptn);
      resetInteraction();
    }
  }, [spreadState, legalMoves, handleMoveSelect, resetInteraction]);

  const handleSquareClick = useCallback(
    (idx: number) => {
      const game = gameRef.current;
      if (!game || game.isGameOver()) return;
      const boardSize = gameInfo?.size ?? 6;
      const moves = legalMoves;

      // --- Spreading mode: user is building a multi-step spread ---
      if (spreadState) {
        // Direction not yet chosen: must click an adjacent square.
        if (!spreadState.direction) {
          const dir = directionFromIdxs(spreadState.sourceIdx, idx);
          if (!dir) return; // ignore non-adjacent clicks
          // Must be exactly 1 square away.
          const expected = adjacentIdx(spreadState.sourceIdx, dir, boardSize);
          if (expected !== idx) return;
          setSpreadState({
            ...spreadState,
            direction: dir,
            drops: [1],
            piecesRemaining: spreadState.pickupCount - 1,
          });
          return;
        }

        // Direction is set. User can:
        // 1. Click the "next" square in line to drop 1 there.
        // 2. Click the current last-drop square to add 1 more to it.

        if (spreadState.piecesRemaining <= 0) return; // already done, wait for confirm

        // Compute the current "head" position (the last square we dropped on).
        let headIdx = spreadState.sourceIdx;
        for (let i = 0; i < spreadState.drops.length; i++) {
          const next = adjacentIdx(headIdx, spreadState.direction, boardSize);
          if (next === null) break;
          headIdx = next;
        }

        // Compute the next square beyond the head.
        const nextIdx = adjacentIdx(headIdx, spreadState.direction, boardSize);

        if (idx === headIdx) {
          // Add 1 more piece to the current square.
          const newDrops = [...spreadState.drops];
          newDrops[newDrops.length - 1]++;
          setSpreadState({
            ...spreadState,
            drops: newDrops,
            piecesRemaining: spreadState.piecesRemaining - 1,
          });
        } else if (nextIdx !== null && idx === nextIdx) {
          // Drop 1 on the next square in line.
          setSpreadState({
            ...spreadState,
            drops: [...spreadState.drops, 1],
            piecesRemaining: spreadState.piecesRemaining - 1,
          });
        }
        // Ignore any other clicks.
        return;
      }

      // --- Stack selected mode: user clicked a stack, hasn't picked up yet ---
      if (selectedSquare !== null) {
        // Click same square to deselect.
        if (idx === selectedSquare) {
          resetInteraction();
          return;
        }
        // Click a different square: deselect and fall through to idle logic.
        resetInteraction();
      }

      // --- Idle mode ---
      // Try placement first.
      const ptnPrefix = placementType === "flat" ? "" : placementType === "wall" ? "S" : "C";
      const sqStr = idxToLabel(idx);
      const placePtn = ptnPrefix + sqStr;
      const placeMove = moves.find((m) => m.ptn === placePtn);
      if (placeMove) {
        handleMoveSelect(placeMove.index);
        return;
      }

      // Check if this square has spread moves available.
      const hasSpread = moves.some(
        (m) => m.ptn.includes(sqStr) && /[+\-><]/.test(m.ptn)
      );
      if (hasSpread) {
        setSelectedSquare(idx);
      }
    },
    [selectedSquare, spreadState, legalMoves, placementType, gameInfo, handleMoveSelect, resetInteraction]
  );

  // Compute highlight set for the board.
  const highlights = new Map<number, "source" | "drop" | "valid">();
  if (spreadState) {
    const boardSize = gameInfo?.size ?? 6;
    highlights.set(spreadState.sourceIdx, "source");

    if (spreadState.direction) {
      // Mark squares that already received drops.
      let cur = spreadState.sourceIdx;
      for (let i = 0; i < spreadState.drops.length; i++) {
        const next = adjacentIdx(cur, spreadState.direction, boardSize);
        if (next === null) break;
        highlights.set(next, "drop");
        cur = next;
      }
      // Mark valid next targets.
      if (spreadState.piecesRemaining > 0) {
        highlights.set(cur, "valid"); // current head (click to add more)
        const nextInLine = adjacentIdx(cur, spreadState.direction, boardSize);
        if (nextInLine !== null) highlights.set(nextInLine, "valid");
      }
    } else {
      // No direction yet: highlight all valid adjacent squares.
      for (const dir of ["+", "-", ">", "<"]) {
        const adj = adjacentIdx(spreadState.sourceIdx, dir, boardSize);
        if (adj !== null) highlights.set(adj, "valid");
      }
    }
  }

  if (loading) {
    return <div style={{ padding: 40, fontSize: 18 }}>Loading Tak engine...</div>;
  }

  const selectedStack =
    selectedSquare !== null && squares[selectedSquare]
      ? squares[selectedSquare].pieces
      : null;

  return (
    <main style={{ padding: 24 }}>
      <h1 style={{ margin: "0 0 16px" }}>TakNN</h1>
      <div style={{ display: "flex", alignItems: "flex-start" }}>
        <Board
          squares={squares}
          size={gameInfo?.size ?? 6}
          selectedSquare={spreadState ? spreadState.sourceIdx : selectedSquare}
          highlights={highlights}
          onSquareClick={handleSquareClick}
        />
        <Controls
          gameInfo={gameInfo}
          legalMoves={legalMoves}
          onNewGame={startGame}
          onMoveSelect={handleMoveSelect}
          onUndo={handleUndo}
          placementType={placementType}
          onPlacementTypeChange={setPlacementType}
          selectedStack={selectedStack}
          selectedSquareLabel={
            selectedSquare !== null ? idxToLabel(selectedSquare) : null
          }
          spreadState={spreadState}
          onPieceClick={handlePieceClick}
          onConfirmSpread={handleConfirmSpread}
          onCancelSpread={resetInteraction}
          onBotMove={handleBotMove}
          botThinking={botThinking}
          lastSearchInfo={lastSearchInfo}
        />
      </div>
    </main>
  );
}
