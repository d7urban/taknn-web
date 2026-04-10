#!/usr/bin/env python3
"""Elo evaluation: Neural teacher vs Anchor C (heuristic PVS).

Supports two neural player modes:
  - value: 1-ply search, evaluate all children by WDL (works with ValueModel or TeacherModel)
  - policy: use policy head to rank moves directly (TeacherModel/StudentModel only)

Usage:
    python eval_elo.py --checkpoint checkpoints/teacher_best.pt --board-size 5
    python eval_elo.py --checkpoint checkpoints/teacher_best.pt --mode policy --board-size 5
"""

import argparse
import math
import time

import numpy as np
import torch

import tak_python
from taknn.data.labels import descriptors_to_tensors


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


class NeuralValuePlayer:
    """1-ply neural evaluation: evaluate all children, pick best WDL."""

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

        batch = torch.stack([torch.from_numpy(np.array(t)) for t in tensors]).to(self.device)
        size_ids = torch.full((len(tensors),), self.size_id, dtype=torch.long, device=self.device)

        outputs = self.model(batch, size_ids)
        wdl = outputs["wdl"] if isinstance(outputs, dict) else outputs[0]

        # Children are from opponent's perspective
        value = wdl[:, 2] - wdl[:, 0]
        return value.argmax().item()


class NeuralPolicyPlayer:
    """Policy-guided move selection using the policy head MLP."""

    def __init__(self, model, device, board_size):
        self.model = model
        self.device = device
        self.board_size = board_size
        self.size_id = board_size - 3

    @torch.no_grad()
    def pick_move(self, game):
        if game.legal_move_count() == 0:
            return 0

        tensor = torch.from_numpy(game.encode_tensor().copy())
        tensor = tensor.unsqueeze(0).float().to(self.device)
        size_id = torch.tensor([self.size_id], dtype=torch.long, device=self.device)

        outputs = self.model(tensor, size_id)

        descs = game.get_move_descriptors()
        desc_tensors = descriptors_to_tensors(descs)
        batched_descs = {k: v.unsqueeze(0).to(self.device) for k, v in desc_tensors.items()}
        num_moves = torch.tensor([len(descs)], dtype=torch.long, device=self.device)

        logits = self.model.score_moves(outputs, batched_descs, num_moves)
        return logits.squeeze(0).argmax().item()


class NeuralCombinedPlayer:
    """Combined policy + value: use policy top-k, rerank by 1-ply value."""

    def __init__(self, model, device, board_size, top_k=5):
        self.model = model
        self.device = device
        self.board_size = board_size
        self.size_id = board_size - 3
        self.top_k = top_k

    @torch.no_grad()
    def pick_move(self, game):
        n_legal = game.legal_move_count()
        if n_legal == 0:
            return 0

        tensor = torch.from_numpy(game.encode_tensor().copy())
        tensor = tensor.unsqueeze(0).float().to(self.device)
        size_id = torch.tensor([self.size_id], dtype=torch.long, device=self.device)

        outputs = self.model(tensor, size_id)

        # Get policy scores
        descs = game.get_move_descriptors()
        desc_tensors = descriptors_to_tensors(descs)
        batched_descs = {k: v.unsqueeze(0).to(self.device) for k, v in desc_tensors.items()}
        num_moves = torch.tensor([len(descs)], dtype=torch.long, device=self.device)

        logits = self.model.score_moves(outputs, batched_descs, num_moves).squeeze(0)

        # Get top-k candidates
        k = min(self.top_k, n_legal)
        top_indices = logits.topk(k).indices.tolist()

        # Evaluate each candidate by 1-ply value
        child_tensors = []
        for idx in top_indices:
            probe = tak_python.PyGameState(self.board_size, game.to_tps())
            probe.apply_move(idx)
            child_tensors.append(torch.from_numpy(probe.encode_tensor().copy()))

        batch = torch.stack(child_tensors).to(self.device)
        size_ids = torch.full((len(child_tensors),), self.size_id, dtype=torch.long, device=self.device)
        child_outputs = self.model(batch, size_ids)
        wdl = child_outputs["wdl"]

        # Pick child where opponent is worst off
        value = wdl[:, 2] - wdl[:, 0]
        best = value.argmax().item()
        return top_indices[best]


def play_match(player_w, player_b, board_size, max_ply=300):
    """Play a single game. Returns result from white's perspective: 1.0, 0.5, or 0.0."""
    game = tak_python.PyGameState(board_size)

    for ply in range(max_ply):
        if game.is_terminal():
            break

        if game.side_to_move() == 0:
            idx = player_w.pick_move(game)
        else:
            idx = player_b.pick_move(game)

        game.apply_move(idx)

    result = game.result()
    if result in (0, 2):
        return 1.0
    elif result in (1, 3):
        return 0.0
    else:
        return 0.5


def elo_diff(score, n):
    """Estimate Elo difference from match score."""
    if score <= 0:
        return -999
    if score >= n:
        return 999
    p = score / n
    return -400 * math.log10(1 / p - 1)


def load_model(checkpoint_path, device):
    """Load a model from checkpoint, auto-detecting type."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get("config", {})
    model_type = config.get("model_type", "teacher")
    channels = config.get("channels", ckpt.get("channels", 128))
    num_blocks = config.get("num_blocks", ckpt.get("blocks", 10))
    film_embed_dim = config.get("film_embed_dim", 32)

    if model_type == "student":
        from taknn.models.student import StudentModel
        model = StudentModel(channels, num_blocks, film_embed_dim)
    else:
        from taknn.models.teacher import TeacherModel
        model = TeacherModel(channels, num_blocks, film_embed_dim)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_type}: {channels}ch x {num_blocks} blocks, {param_count:,} params")
    if "val_loss" in ckpt:
        print(f"  val_loss={ckpt['val_loss']:.4f}, epoch={ckpt.get('epoch', '?')}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/teacher_best.pt")
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--mode", default="value", choices=["value", "policy", "combined"],
                        help="Neural player mode: value (1-ply WDL), policy (policy head), combined (policy top-k + value rerank)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for combined mode")
    parser.add_argument("--time-ms", type=int, default=1000, help="Anchor time per move in ms")
    parser.add_argument("--anchor-depth", type=int, default=20)
    parser.add_argument("--anchor-tt-mb", type=int, default=16)
    parser.add_argument("--tactical-suite", default=None, help="Path to tactical suite dir for additional eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    if args.mode == "policy":
        neural = NeuralPolicyPlayer(model, device, args.board_size)
    elif args.mode == "combined":
        neural = NeuralCombinedPlayer(model, device, args.board_size, top_k=args.top_k)
    else:
        neural = NeuralValuePlayer(model, device, args.board_size)

    anchor = AnchorC(
        max_depth=args.anchor_depth,
        max_time_ms=args.time_ms,
        tt_size_mb=args.anchor_tt_mb,
    )

    # Run tactical suite if provided
    if args.tactical_suite:
        from taknn.eval.tactical import TacticalSuiteRunner
        runner = TacticalSuiteRunner(model, args.tactical_suite, device=device)
        results = runner.run()
        print(f"\nTactical Suite: {results['correct']}/{results['total']} = {results['accuracy']:.1%}")
        for cat, stats in results.get("per_category", {}).items():
            print(f"  {cat}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1%}")
        print()

    # Elo match
    neural_score = 0.0
    neural_wins = 0
    neural_losses = 0
    draws = 0

    print(f"Neural ({args.mode}) vs Anchor C: {args.num_games} games on "
          f"{args.board_size}x{args.board_size}, {args.time_ms}ms/move")
    print("-" * 70)

    t0 = time.time()
    for game_idx in range(args.num_games):
        if game_idx % 2 == 0:
            result = play_match(neural, anchor, args.board_size)
            neural_pts = result
        else:
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

    total_elapsed = time.time() - t0
    final_elo = elo_diff(neural_score, args.num_games)
    print("-" * 70)
    print(f"Final: Neural {neural_score:.1f}/{args.num_games} "
          f"({neural_score/args.num_games:.1%}), Elo diff: {final_elo:+.0f}")
    print(f"Time: {total_elapsed:.0f}s ({total_elapsed/args.num_games:.1f}s/game)")


if __name__ == "__main__":
    main()
