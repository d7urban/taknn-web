"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Board from "@/components/Board";
import Controls from "@/components/Controls";
import {
  initWasm,
  SquareInfo,
  MoveInfo,
  GameInfo,
} from "@/engine/wasm";

// TakGame class from WASM.
type TakGame = {
  getBoard(): SquareInfo[];
  legalMoves(): MoveInfo[];
  getInfo(): GameInfo;
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

export default function Home() {
  const [wasm, setWasm] = useState<WasmModule | null>(null);
  const gameRef = useRef<TakGame | null>(null);
  const [squares, setSquares] = useState<SquareInfo[]>([]);
  const [legalMoves, setLegalMoves] = useState<MoveInfo[]>([]);
  const [gameInfo, setGameInfo] = useState<GameInfo | null>(null);
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [placementType, setPlacementType] = useState<"flat" | "wall" | "cap">("flat");
  const [loading, setLoading] = useState(true);

  // Initialize WASM.
  useEffect(() => {
    initWasm().then((mod) => {
      setWasm(mod as unknown as WasmModule);
      setLoading(false);
    });
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
    setSelectedSquare(null);
  }, []);

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

  const handleSquareClick = useCallback(
    (idx: number) => {
      const game = gameRef.current;
      if (!game || game.isGameOver()) return;

      // Try to find a matching move.
      const moves = legalMoves;

      // If clicking the same square, deselect.
      if (selectedSquare === idx) {
        setSelectedSquare(null);
        return;
      }

      // If no square selected yet, check if this is a placement target or a spread source.
      if (selectedSquare === null) {
        // Check for placement at this square.
        const ptnPrefix = placementType === "flat" ? "" : placementType === "wall" ? "S" : "C";
        const col = idx % 8;
        const row = Math.floor(idx / 8);
        const sqStr = String.fromCharCode(97 + col) + (row + 1);
        const placePtn = ptnPrefix + sqStr;

        const placeMove = moves.find((m) => m.ptn === placePtn);
        if (placeMove) {
          handleMoveSelect(placeMove.index);
          return;
        }

        // Check if there's a spread from this square.
        const hasSpread = moves.some(
          (m) => m.ptn.includes(sqStr) && (m.ptn.includes("+") || m.ptn.includes("-") || m.ptn.includes(">") || m.ptn.includes("<"))
        );
        if (hasSpread) {
          setSelectedSquare(idx);
        }
      } else {
        // Second click: find a spread from selectedSquare to this square.
        const srcCol = selectedSquare % 8;
        const srcRow = Math.floor(selectedSquare / 8);
        const srcStr = String.fromCharCode(97 + srcCol) + (srcRow + 1);
        const dstCol = idx % 8;
        const dstRow = Math.floor(idx / 8);

        let dir = "";
        if (dstRow > srcRow && dstCol === srcCol) dir = "+";
        else if (dstRow < srcRow && dstCol === srcCol) dir = "-";
        else if (dstCol > srcCol && dstRow === srcRow) dir = ">";
        else if (dstCol < srcCol && dstRow === srcRow) dir = "<";

        if (dir) {
          // Find the simplest spread (pickup 1, drop on first square).
          const spreadPtn = srcStr + dir;
          const spreadMove = moves.find((m) => m.ptn === spreadPtn);
          if (spreadMove) {
            handleMoveSelect(spreadMove.index);
            return;
          }
        }
        setSelectedSquare(null);
      }
    },
    [selectedSquare, legalMoves, placementType, handleMoveSelect]
  );

  if (loading) {
    return <div style={{ padding: 40, fontSize: 18 }}>Loading Tak engine...</div>;
  }

  return (
    <main style={{ padding: 24 }}>
      <h1 style={{ margin: "0 0 16px" }}>TakNN</h1>
      <div style={{ display: "flex", alignItems: "flex-start" }}>
        <Board
          squares={squares}
          size={gameInfo?.size ?? 6}
          selectedSquare={selectedSquare}
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
        />
      </div>
    </main>
  );
}
