#!/usr/bin/env python3
"""Quick evaluation: tactical suite + optional Elo match.

Usage:
    python eval_full.py --checkpoint checkpoints/teacher_best.pt
    python eval_full.py --checkpoint checkpoints/teacher_best.pt --elo --board-size 5
"""

import argparse
import torch
from taknn.eval.tactical import TacticalSuiteRunner


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on tactical suite and Elo")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--suite-dir", default="taknn/eval/suites", help="Tactical suite directory")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k for tactical accuracy")
    parser.add_argument("--elo", action="store_true", help="Also run Elo match (slow)")
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--num-games", type=int, default=50)
    parser.add_argument("--time-ms", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import here to avoid circular imports at module level
    from eval_elo import load_model
    model = load_model(args.checkpoint, device)

    # Tactical suite
    print("\n=== Tactical Suite ===")
    runner = TacticalSuiteRunner(model, args.suite_dir, device=device)
    for k in [1, 3]:
        results = runner.run(top_k=k)
        print(f"\nTop-{k}: {results['correct']}/{results['total']} = {results['accuracy']:.1%}")
        for cat, stats in sorted(results.get("per_category", {}).items()):
            print(f"  {cat}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1%}")

    # Elo match
    if args.elo:
        from eval_elo import NeuralPolicyPlayer, NeuralValuePlayer, AnchorC, play_match, elo_diff
        import time

        print(f"\n=== Elo Match ({args.board_size}x{args.board_size}) ===")

        policy_player = NeuralPolicyPlayer(model, device, args.board_size)
        value_player = NeuralValuePlayer(model, device, args.board_size)
        anchor = AnchorC(max_time_ms=args.time_ms)

        for name, player in [("policy", policy_player), ("value", value_player)]:
            score = 0.0
            t0 = time.time()
            for i in range(args.num_games):
                if i % 2 == 0:
                    r = play_match(player, anchor, args.board_size)
                    score += r
                else:
                    r = play_match(anchor, player, args.board_size)
                    score += 1.0 - r

            dt = time.time() - t0
            elo = elo_diff(score, args.num_games)
            print(f"  {name}: {score:.1f}/{args.num_games} ({score/args.num_games:.1%}), "
                  f"Elo: {elo:+.0f} ({dt:.0f}s)")


if __name__ == "__main__":
    main()
