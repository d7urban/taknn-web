#!/usr/bin/env python3
"""Elo evaluation: Neural teacher vs Anchor C (heuristic PVS).

Neural player: 1-ply search using batched model inference on all children.
Anchor C: Rust PVS with heuristic eval at the same time budget.
"""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import tak_python
from train_teacher import ValueModel


class AnchorC:
    """Heuristic PVS player via Rust engine."""

    def __init__(self, max_depth=20, max_time_ms=1000, tt_size_mb=16):
        self.max_depth = max_depth
        self.max_time_ms = max_time_ms
        self.tt_size_mb = tt_size_mb

    def pick_move(self, game):
        idx, score, depth, nodes = game.search_move(
            max_depth=self.max_depth,
            max_time_ms=self.max_time_ms,
            tt_size_mb=self.tt_size_mb,
        )
        return idx


class NeuralPlayer:
    """1-ply neural evaluation: evaluate all children, pick best."""

    def __init__(self, model, device, board_size):
        self.model = model
        self.device = device
        self.board_size = board_size
        self.size_id = board_size - 3

    @torch.no_grad()
    def pick_move(self, game):
        tensors = game.encode_children_tensors()
        if not tensors:
            return 0

        # Stack into batch [N, 31, 8, 8]
        batch = torch.stack([torch.from_numpy(np.array(t)) for t in tensors]).to(self.device)
        size_ids = torch.full((len(tensors),), self.size_id, dtype=torch.long, device=self.device)

        wdl, margin = self.model(batch, size_ids)

        # Children are from opponent's perspective, so we want the child
        # where the opponent has lowest win prob = our highest win prob.
        # wdl columns: [win, draw, loss] from the child's STM perspective.
        # Child's STM is our opponent, so:
        #   opponent_win = wdl[:, 0], draw = wdl[:, 1], opponent_loss = wdl[:, 2]
        # We want to maximize opponent_loss (= our win) - opponent_win (= our loss)
        value = wdl[:, 2] - wdl[:, 0]  # higher = better for us

        return value.argmax().item()


def play_match(player_w, player_b, board_size, max_ply=300):
    """Play a single game. Returns result from white's perspective: 1.0, 0.5, or 0.0."""
    game = tak_python.PyGameState(board_size)

    for ply in range(max_ply):
        if game.is_terminal():
            break

        if game.side_to_move() == 0:  # White
            idx = player_w.pick_move(game)
        else:  # Black
            idx = player_b.pick_move(game)

        game.apply_move(idx)

    result = game.result()
    if result in (0, 2):  # White road/flat win
        return 1.0
    elif result in (1, 3):  # Black road/flat win
        return 0.0
    else:  # Draw or ongoing (count as draw)
        return 0.5


def elo_diff(score, n):
    """Estimate Elo difference from match score."""
    if score <= 0:
        return -999
    if score >= n:
        return 999
    p = score / n
    return -400 * math.log10(1 / p - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/teacher_4x4.pt")
    parser.add_argument("--board-size", type=int, default=4)
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--time-ms", type=int, default=1000, help="Time per move in ms")
    parser.add_argument("--anchor-depth", type=int, default=20)
    parser.add_argument("--anchor-tt-mb", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = ValueModel(
        channels=ckpt.get("channels", 64),
        num_blocks=ckpt.get("blocks", 6),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model: {ckpt.get('channels', 64)}ch x {ckpt.get('blocks', 6)} blocks, "
          f"val_loss={ckpt.get('val_loss', '?'):.4f}, val_acc={ckpt.get('val_acc', '?'):.1%}")

    neural = NeuralPlayer(model, device, args.board_size)
    anchor = AnchorC(
        max_depth=args.anchor_depth,
        max_time_ms=args.time_ms,
        tt_size_mb=args.anchor_tt_mb,
    )

    neural_score = 0.0
    neural_wins = 0
    neural_losses = 0
    draws = 0

    print(f"\nNeural vs Anchor C: {args.num_games} games on {args.board_size}x{args.board_size}, "
          f"{args.time_ms}ms/move")
    print("-" * 70)

    t0 = time.time()
    for game_idx in range(args.num_games):
        # Alternate colors
        if game_idx % 2 == 0:
            # Neural plays White
            result = play_match(neural, anchor, args.board_size)
            neural_pts = result
        else:
            # Neural plays Black
            result = play_match(anchor, neural, args.board_size)
            neural_pts = 1.0 - result

        neural_score += neural_pts
        if neural_pts == 1.0:
            neural_wins += 1
        elif neural_pts == 0.0:
            neural_losses += 1
        else:
            draws += 1

        n = game_idx + 1
        color = "W" if game_idx % 2 == 0 else "B"
        if n % 10 == 0 or n == args.num_games or n <= 5:
            elo = elo_diff(neural_score, n)
            elapsed = time.time() - t0
            print(
                f"Game {n:3d}/{args.num_games} | "
                f"Neural: +{neural_wins} ={draws} -{neural_losses} | "
                f"Score: {neural_score:.1f}/{n} ({neural_score/n:.1%}) | "
                f"Elo: {elo:+.0f} | "
                f"{elapsed:.0f}s",
                flush=True,
            )
        elif n <= 10:
            print(f"  g{n} neural={color} pts={neural_pts}", flush=True)

    total_elapsed = time.time() - t0
    final_elo = elo_diff(neural_score, args.num_games)
    print("-" * 70)
    print(f"Final: Neural {neural_score:.1f}/{args.num_games} "
          f"({neural_score/args.num_games:.1%}), Elo diff: {final_elo:+.0f}")
    print(f"Time: {total_elapsed:.0f}s ({total_elapsed/args.num_games:.1f}s/game)")


if __name__ == "__main__":
    main()
