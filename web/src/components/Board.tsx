"use client";

import { SquareInfo, PieceInfo } from "@/engine/wasm";

interface BoardProps {
  squares: SquareInfo[];
  size: number;
  selectedSquare: number | null;
  highlights: Map<number, "source" | "drop" | "valid">;
  lastMoveSquares: Set<number>;
  onSquareClick: (index: number) => void;
}

function highlightBorder(
  isSelected: boolean,
  highlight: "source" | "drop" | "valid" | undefined
): string {
  if (isSelected) return "3px solid #2196f3";
  if (highlight === "source") return "3px solid #2196f3";
  if (highlight === "drop") return "3px solid #ff9800";
  if (highlight === "valid") return "3px dashed #4caf50";
  return "1px solid #8b7355";
}

export default function Board({ squares, size, selectedSquare, highlights, lastMoveSquares, onSquareClick }: BoardProps) {
  // Render rows from top (row size-1) to bottom (row 0) visually.
  const rows = [];
  for (let r = size - 1; r >= 0; r--) {
    const cells = [];
    for (let c = 0; c < size; c++) {
      const idx = r * 8 + c;
      const sq = squares[idx];
      if (!sq) continue;
      const isSelected = selectedSquare === idx;
      const highlight = highlights.get(idx);
      const isLastMove = lastMoveSquares.has(idx);
      const topPiece = sq.pieces.length > 0 ? sq.pieces[sq.pieces.length - 1] : null;
      const stackHeight = sq.pieces.length;

      const baseBg = (r + c) % 2 === 0 ? "#f5e6c8" : "#e8d5a8";
      const bg = isLastMove ? ((r + c) % 2 === 0 ? "#e8e0a0" : "#ddd590") : baseBg;

      cells.push(
        <div
          key={`${r}-${c}`}
          onClick={() => onSquareClick(idx)}
          style={{
            width: 64,
            height: 64,
            border: highlightBorder(isSelected, highlight),
            backgroundColor: bg,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            position: "relative",
            boxSizing: "border-box",
          }}
        >
          {topPiece && <PieceView piece={topPiece} stackHeight={stackHeight} />}
          {highlight === "drop" && (
            <span
              style={{
                position: "absolute",
                top: 1,
                right: 2,
                fontSize: 9,
                color: "#ff9800",
                fontWeight: "bold",
              }}
            >
            </span>
          )}
          <span
            style={{
              position: "absolute",
              bottom: 1,
              left: 2,
              fontSize: 9,
              color: "#999",
            }}
          >
            {String.fromCharCode(97 + c)}{r + 1}
          </span>
        </div>
      );
    }
    rows.push(
      <div key={r} style={{ display: "flex" }}>
        {cells}
      </div>
    );
  }

  return (
    <div style={{ border: "2px solid #5c4033", display: "inline-block" }}>
      {rows}
    </div>
  );
}

function PieceView({ piece, stackHeight }: { piece: PieceInfo; stackHeight: number }) {
  const color = piece.color === "white" ? "#f5f5f5" : "#333";
  const border = piece.color === "white" ? "2px solid #999" : "2px solid #111";
  const textColor = piece.color === "white" ? "#333" : "#f5f5f5";

  if (piece.pieceType === "wall") {
    return (
      <div
        style={{
          width: 10,
          height: 40,
          backgroundColor: color,
          border,
          borderRadius: 2,
        }}
      >
        {stackHeight > 1 && (
          <span style={{ fontSize: 8, color: textColor, position: "absolute", top: 0, right: 2 }}>
            {stackHeight}
          </span>
        )}
      </div>
    );
  }

  if (piece.pieceType === "cap") {
    return (
      <div style={{ position: "relative" }}>
        <div
          style={{
            width: 36,
            height: 36,
            backgroundColor: color,
            border,
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <span style={{ fontWeight: "bold", fontSize: 14, color: textColor }}>C</span>
        </div>
        {stackHeight > 1 && (
          <span style={{ fontSize: 8, color: "#666", position: "absolute", top: -4, right: -4 }}>
            {stackHeight}
          </span>
        )}
      </div>
    );
  }

  // Flat
  return (
    <div style={{ position: "relative" }}>
      <div
        style={{
          width: 42,
          height: 42,
          backgroundColor: color,
          border,
          borderRadius: 4,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {stackHeight > 1 && (
          <span style={{ fontSize: 12, fontWeight: "bold", color: textColor }}>
            {stackHeight}
          </span>
        )}
      </div>
    </div>
  );
}
