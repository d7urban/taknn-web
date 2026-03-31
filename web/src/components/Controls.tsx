"use client";

import { MoveInfo, GameInfo } from "@/engine/wasm";

interface ControlsProps {
  gameInfo: GameInfo | null;
  legalMoves: MoveInfo[];
  onNewGame: (size: number) => void;
  onMoveSelect: (index: number) => void;
  onUndo: () => void;
  placementType: "flat" | "wall" | "cap";
  onPlacementTypeChange: (t: "flat" | "wall" | "cap") => void;
}

export default function Controls({
  gameInfo,
  legalMoves,
  onNewGame,
  onMoveSelect,
  onUndo,
  placementType,
  onPlacementTypeChange,
}: ControlsProps) {
  return (
    <div style={{ marginLeft: 24, maxWidth: 300 }}>
      <h3 style={{ margin: "0 0 8px" }}>New Game</h3>
      <div style={{ display: "flex", gap: 4, marginBottom: 16 }}>
        {[3, 4, 5, 6, 7, 8].map((s) => (
          <button key={s} onClick={() => onNewGame(s)} style={{ padding: "4px 12px" }}>
            {s}x{s}
          </button>
        ))}
      </div>

      {gameInfo && (
        <div style={{ marginBottom: 16 }}>
          <p>
            <b>Turn:</b> {gameInfo.sideToMove} (ply {gameInfo.ply})
          </p>
          <p>
            <b>Result:</b> {gameInfo.result}
          </p>
          <p>
            <b>Reserves:</b> W: {gameInfo.whiteStones}s/{gameInfo.whiteCaps}c | B:{" "}
            {gameInfo.blackStones}s/{gameInfo.blackCaps}c
          </p>
        </div>
      )}

      <div style={{ marginBottom: 12 }}>
        <b>Place:</b>{" "}
        {(["flat", "wall", "cap"] as const)
          .filter((t) => {
            if (t !== "cap") return true;
            if (!gameInfo) return false;
            return gameInfo.whiteCaps > 0 || gameInfo.blackCaps > 0;
          })
          .map((t) => (
          <button
            key={t}
            onClick={() => onPlacementTypeChange(t)}
            style={{
              padding: "2px 8px",
              marginRight: 4,
              fontWeight: placementType === t ? "bold" : "normal",
              backgroundColor: placementType === t ? "#ddd" : "transparent",
            }}
          >
            {t}
          </button>
        ))}
      </div>

      <button onClick={onUndo} style={{ padding: "4px 16px", marginBottom: 16 }}>
        Undo
      </button>

      <h3 style={{ margin: "0 0 8px" }}>Legal Moves ({legalMoves.length})</h3>
      <div
        style={{
          maxHeight: 300,
          overflow: "auto",
          border: "1px solid #ccc",
          padding: 4,
          fontSize: 13,
        }}
      >
        {legalMoves.map((m) => (
          <button
            key={m.index}
            onClick={() => onMoveSelect(m.index)}
            style={{
              display: "inline-block",
              margin: 2,
              padding: "2px 6px",
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            {m.ptn}
          </button>
        ))}
      </div>
    </div>
  );
}
