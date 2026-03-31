# Tak Rules

Tak is a two-player abstract strategy board game designed by James Ernest, based on the game described in Patrick Rothfuss's *The Wise Man's Fear*. The goal is to build a **road** — a connected path of your pieces spanning opposite edges of the board.

## Setup

| Board Size | Stones per Player | Capstones per Player |
|:----------:|:-----------------:|:--------------------:|
| 3×3        | 10                | 0                    |
| 4×4        | 15                | 0                    |
| 5×5        | 21                | 1                    |
| 6×6        | 30                | 1                    |
| 7×7        | 40                | 2                    |
| 8×8        | 50                | 2                    |

Players take turns. White moves first. All pieces start off the board in each player's **reserve**.

## Piece Types

- **Flat stone** — The basic piece. Counts for roads and flat wins. Other pieces can stack on top of it.
- **Standing stone (wall)** — A flat stone placed on its side. Blocks movement — nothing can move onto a wall. Does **not** count for roads or flat wins.
- **Capstone** — A special piece (limited supply). Counts for roads. Blocks movement like a wall, with one exception: a capstone can **flatten** an opponent's wall under specific conditions (see Spreading).

## Opening Rule

On the **first two turns** (ply 0 and ply 1), each player places one of their **opponent's** flat stones on any empty square. No walls, capstones, or spreads are allowed during the opening.

After the opening, normal play begins.

## Placement

On your turn you may place one piece from your reserve onto any empty square:

- A **flat stone** (placed flat).
- A **standing stone / wall** (uses a stone from your reserve).
- A **capstone** (if you have any remaining).

You cannot place on an occupied square.

## Spreading (Stack Movement)

Instead of placing, you may pick up a stack you control (your piece is on top) and spread it along a straight line — north, south, east, or west.

1. **Pick up** 1 to N pieces from the top of the stack, where N is the board size (the **carry limit**).
2. **Move** in a single direction, dropping **at least one** piece on each square you pass through.
3. You must drop pieces on **every** square along the path — you cannot skip squares.
4. You cannot move onto or through a **wall** or **capstone** (they block movement).

**Capstone flatten exception:** If the capstone is the **only** piece being dropped on the **final** square of a spread, and that square contains an opponent's wall, the wall is flattened into a flat stone and the capstone is placed on top. This is the only way to remove a wall.

### Drop Sequences

The pieces you pick up must be distributed across the squares you travel through. Each square must receive at least 1 piece. For example, picking up 3 pieces and moving 2 squares could drop as (2, 1) or (1, 2).

## Winning

### Road Win (immediate)

A **road** is an orthogonally connected group of your flat stones and/or capstones that spans from one edge of the board to the opposite edge — either north-to-south or west-to-east. Walls do **not** count for roads.

A road win is checked after every move. If the **moving player** completes a road, they win. If both players simultaneously complete a road (possible via a spread), the **moving player** wins.

### Flat Win (end-of-game tiebreaker)

The game ends in a flat win if:

- The board is completely full, **or**
- Either player has exhausted **all** of their reserves (stones and capstones).

When this happens, the player with **more flat stones visible on top of stacks** wins. Walls and capstones on top do not count. Standing stones buried inside stacks are irrelevant — only the top piece of each stack matters.

If flat counts are tied, the game is a **draw**.

**Komi** (optional): An integer bonus added to White's flat count to compensate for first-move advantage. **Half-komi** adds an additional 0.5 to break ties in White's favor.

### Draw by Repetition

If the same position (board + side to move + reserves) occurs for the **third** time, the game is a draw.

## Notation

### PTN (Portable Tak Notation)

Squares are named with a column letter and row number: `a1` is the bottom-left corner, `h8` is the top-right on an 8×8 board.

**Placements:**
- `a1` — place a flat stone on a1
- `Sa1` — place a standing stone (wall) on a1
- `Ca1` — place a capstone on a1

**Spreads:**
- `a1>` — pick up 1 from a1, move east, drop 1
- `3a1>12` — pick up 3 from a1, move east, drop 1 then 2

Direction symbols: `+` up, `-` down, `>` right, `<` left.

### TPS (Tak Positional System)

A FEN-like string describing a board position:

```
x5/x5/x,1,2,x2/x5/1,2,x3 1 3
```

- Rows are separated by `/`, listed top-to-bottom.
- `x` = one empty square, `xN` = N consecutive empty squares.
- `1` = white flat, `2` = black flat, `1S` = white wall, `2C` = black capstone.
- Stacks are listed bottom-to-top: `12` = white flat on bottom, black flat on top.
- After the board: player to move (`1` = white, `2` = black) and the full-move number.
