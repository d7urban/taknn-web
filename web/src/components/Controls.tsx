"use client";

import { MoveInfo, GameInfo, PieceInfo } from "@/engine/wasm";
import { SpreadState, SearchInfo } from "@/app/page";

function PieceIcon({ piece, size = 18 }: { piece: PieceInfo; size?: number }) {
  const bg = piece.color === "white" ? "#f5f5f5" : "#333";
  const fg = piece.color === "white" ? "#333" : "#f5f5f5";
  const borderColor = piece.color === "white" ? "#999" : "#111";
  const label = piece.pieceType === "cap" ? "C" : piece.pieceType === "wall" ? "W" : "F";
  return (
    <span
      style={{
        display: "inline-block",
        width: size,
        height: size,
        backgroundColor: bg,
        border: `2px solid ${borderColor}`,
        borderRadius: piece.pieceType === "cap" ? "50%" : 2,
        textAlign: "center",
        lineHeight: `${size - 2}px`,
        fontSize: size * 0.55,
        fontWeight: "bold",
        color: fg,
      }}
    >
      {label}
    </span>
  );
}

interface ControlsProps {
  gameInfo: GameInfo | null;
  legalMoves: MoveInfo[];
  onNewGame: (size: number) => void;
  onMoveSelect: (index: number) => void;
  onUndo: () => void;
  placementType: "flat" | "wall" | "cap";
  onPlacementTypeChange: (t: "flat" | "wall" | "cap") => void;
  selectedStack: PieceInfo[] | null;
  selectedSquareLabel: string | null;
  spreadState: SpreadState | null;
  onPieceClick: (pickupCount: number) => void;
  onConfirmSpread: () => void;
  onCancelSpread: () => void;
  onBotMove: () => void;
  botThinking: boolean;
  lastSearchInfo: SearchInfo | null;
}

export default function Controls({
  gameInfo,
  legalMoves,
  onNewGame,
  onMoveSelect,
  onUndo,
  placementType,
  onPlacementTypeChange,
  selectedStack,
  selectedSquareLabel,
  spreadState,
  onPieceClick,
  onConfirmSpread,
  onCancelSpread,
  onBotMove,
  botThinking,
  lastSearchInfo,
}: ControlsProps) {
  const carryLimit = gameInfo?.size ?? 8;

  return (
    <div style={{ marginLeft: 24, maxWidth: 340 }}>
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
        {(["flat", "wall", "cap"] as const).map((t) => (
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

      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <button onClick={onUndo} style={{ padding: "4px 16px" }}>
          Undo
        </button>
        <button
          onClick={onBotMove}
          disabled={botThinking}
          style={{
            padding: "4px 16px",
            backgroundColor: botThinking ? "#ccc" : "#1976d2",
            color: "white",
            border: "none",
            borderRadius: 3,
            cursor: botThinking ? "wait" : "pointer",
          }}
        >
          {botThinking ? "Thinking..." : "Bot Move"}
        </button>
      </div>

      {lastSearchInfo && (
        <div
          style={{
            marginBottom: 12,
            padding: 6,
            fontSize: 11,
            backgroundColor: "#f0f0f0",
            borderRadius: 3,
            color: "#555",
          }}
        >
          <div style={{ marginBottom: 4 }}>
            <b>Engine:</b>{" "}
            <span
              style={{
                display: "inline-block",
                padding: "1px 6px",
                borderRadius: 999,
                backgroundColor: lastSearchInfo.engineMode === "neural" ? "#d9f2e3" : "#ececec",
                color: lastSearchInfo.engineMode === "neural" ? "#1f6b3a" : "#666",
                fontWeight: 600,
              }}
            >
              {lastSearchInfo.engineMode}
              {lastSearchInfo.modelName ? ` (${lastSearchInfo.modelName})` : ""}
            </span>
          </div>
          depth {lastSearchInfo.depth} | score {lastSearchInfo.score} | {lastSearchInfo.nodes.toLocaleString()} nodes
          {lastSearchInfo.pv.length > 0 && (
            <span> | PV: {lastSearchInfo.pv.slice(0, 5).join(" ")}</span>
          )}
        </div>
      )}

      {/* Spread in progress */}
      {spreadState && (
        <SpreadPanel
          spread={spreadState}
          carryLimit={carryLimit}
          onConfirm={onConfirmSpread}
          onCancel={onCancelSpread}
        />
      )}

      {/* Stack display (only when selected but not yet spreading) */}
      {!spreadState && selectedStack && selectedStack.length > 0 && (
        <StackSelector
          pieces={selectedStack}
          squareLabel={selectedSquareLabel ?? ""}
          carryLimit={carryLimit}
          onPieceClick={onPieceClick}
        />
      )}

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

function StackSelector({
  pieces,
  squareLabel,
  carryLimit,
  onPieceClick,
}: {
  pieces: PieceInfo[];
  squareLabel: string;
  carryLimit: number;
  onPieceClick: (pickupCount: number) => void;
}) {
  // Display top to bottom. pieces[0] is bottom, pieces[len-1] is top.
  const maxPickup = Math.min(pieces.length, carryLimit);

  return (
    <div style={{ marginBottom: 16 }}>
      <b>Stack at {squareLabel}</b>{" "}
      <span style={{ fontSize: 11, color: "#666" }}>
        — click a piece to pick up it and everything above
      </span>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 1,
          marginTop: 4,
          padding: 4,
          border: "1px solid #ccc",
          backgroundColor: "#fafafa",
        }}
      >
        {[...pieces].reverse().map((p, visualIdx) => {
          const pickupCount = visualIdx + 1;
          const canPickup = pickupCount <= maxPickup;
          return (
            <div
              key={visualIdx}
              onClick={canPickup ? () => onPieceClick(pickupCount) : undefined}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "3px 6px",
                fontSize: 12,
                fontWeight: visualIdx === 0 ? "bold" : "normal",
                cursor: canPickup ? "pointer" : "default",
                opacity: canPickup ? 1 : 0.4,
                borderRadius: 3,
                backgroundColor: "transparent",
                transition: "background-color 0.1s",
              }}
              onMouseEnter={(e) => {
                if (canPickup) e.currentTarget.style.backgroundColor = "#e3f2fd";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "transparent";
              }}
            >
              <PieceIcon piece={p} />
              <span>
                {p.color} {p.pieceType}
                {visualIdx === 0 ? " (top)" : ""}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SpreadPanel({
  spread,
  carryLimit,
  onConfirm,
  onCancel,
}: {
  spread: SpreadState;
  carryLimit: number;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const dirLabel: Record<string, string> = { "+": "north", "-": "south", ">": "east", "<": "west" };
  const done = spread.piecesRemaining === 0;

  // Split carried pieces into already-dropped and still-held.
  const droppedCount = spread.pickupCount - spread.piecesRemaining;
  // carriedPieces is bottom-to-top of what was picked up.
  // The first pieces dropped come from the top of the carried stack.
  const heldPieces = spread.carriedPieces.slice(0, spread.piecesRemaining);
  const droppedPieces = spread.carriedPieces.slice(spread.piecesRemaining);

  return (
    <div
      style={{
        marginBottom: 16,
        padding: 8,
        border: "2px solid #2196f3",
        borderRadius: 4,
        backgroundColor: "#f8f9ff",
      }}
    >
      <div style={{ marginBottom: 6, fontWeight: "bold", fontSize: 13 }}>
        Spreading from {spread.sourceLabel}
        {spread.direction ? ` ${dirLabel[spread.direction]}` : ""}
        {" "}(pickup {spread.pickupCount})
      </div>

      {/* Held pieces */}
      {heldPieces.length > 0 && (
        <div style={{ marginBottom: 6 }}>
          <span style={{ fontSize: 11, color: "#666" }}>Held:</span>
          <div style={{ display: "flex", gap: 3, marginTop: 2 }}>
            {[...heldPieces].reverse().map((p, i) => (
              <PieceIcon key={`held-${i}`} piece={p} />
            ))}
          </div>
        </div>
      )}

      {/* Drop sequence */}
      {spread.drops.length > 0 && (
        <div style={{ marginBottom: 6 }}>
          <span style={{ fontSize: 11, color: "#666" }}>Drops:</span>
          <div style={{ display: "flex", gap: 4, marginTop: 2, flexWrap: "wrap" }}>
            {spread.drops.map((count, i) => (
              <span
                key={i}
                style={{
                  fontSize: 12,
                  padding: "1px 6px",
                  backgroundColor: "#e8e8e8",
                  borderRadius: 3,
                }}
              >
                sq {i + 1}: {count}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Instructions */}
      <div style={{ fontSize: 11, color: "#666", marginBottom: 8 }}>
        {!spread.direction
          ? "Click an adjacent square to set direction"
          : done
          ? "All pieces placed — confirm or cancel"
          : `${spread.piecesRemaining} left — click next square or same square to drop more`}
      </div>

      {/* Buttons */}
      <div style={{ display: "flex", gap: 8 }}>
        {done && (
          <button
            onClick={onConfirm}
            style={{
              padding: "4px 16px",
              fontWeight: "bold",
              backgroundColor: "#4caf50",
              color: "white",
              border: "none",
              borderRadius: 3,
              cursor: "pointer",
            }}
          >
            Confirm
          </button>
        )}
        <button
          onClick={onCancel}
          style={{
            padding: "4px 16px",
            border: "1px solid #999",
            borderRadius: 3,
            cursor: "pointer",
            backgroundColor: "transparent",
          }}
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
