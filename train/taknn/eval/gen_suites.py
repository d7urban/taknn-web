#!/usr/bin/env python3
"""Generate tactical test suite positions from shard data.

Scans existing 5x5 (and optionally 4x4) shard positions, identifies
those with specific tactical properties, and saves them as JSONL files
for use by the TacticalSuiteRunner.

Usage:
    python -m taknn.eval.gen_suites --shards shards/ --out taknn/eval/suites/
"""

import argparse
import json
import os
import random

import tak_python
from ..data.shard import ShardDataset


def find_winning_moves(game):
    """Return list of move indices that win by road."""
    winners = []
    n = game.legal_move_count()
    for i in range(n):
        probe = tak_python.PyGameState(game.size(), game.to_tps())
        probe.apply_move(i)
        if probe.is_terminal():
            r = probe.result()
            stm = game.side_to_move()
            # Road wins: 0 = white road, 1 = black road
            if (stm == 0 and r == 0) or (stm == 1 and r == 1):
                winners.append(i)
    return winners


def find_blocking_moves(game):
    """Return list of move indices that prevent opponent road-in-1.

    A blocking move is one where, after playing it, the opponent
    no longer has a road win in 1.
    """
    blockers = []
    opp = 1 - game.side_to_move()
    n = game.legal_move_count()
    for i in range(n):
        probe = tak_python.PyGameState(game.size(), game.to_tps())
        probe.apply_move(i)
        if probe.is_terminal():
            # If this move itself wins, it's also a valid "block" (winning > blocking)
            blockers.append(i)
            continue
        # Check if opponent still has road-in-1 after our move
        labels = probe.get_tactical_labels()
        opp_has_road = labels["road_in_1_white"] if opp == 0 else labels["road_in_1_black"]
        if not opp_has_road:
            blockers.append(i)
    return blockers


def find_capstone_flatten_moves(game):
    """Return list of move indices that involve capstone flattening."""
    descs = game.get_move_descriptors()
    return [i for i, d in enumerate(descs) if d["capstone_flatten"] > 0.5]


def generate_suites(shard_paths, out_dir, target_per_category=60, board_sizes=None):
    """Scan shards and generate tactical test suites."""
    if board_sizes is None:
        board_sizes = {4, 5}

    road_win = []
    forced_block = []
    capstone_flatten = []

    ds = ShardDataset(shard_paths, load_descriptors=False, load_aux_labels=False)
    print(f"Scanning {len(ds)} positions...")

    indices = list(range(len(ds)))
    random.shuffle(indices)

    for count, idx in enumerate(indices):
        # Stop early if we have enough
        if (len(road_win) >= target_per_category and
                len(forced_block) >= target_per_category and
                len(capstone_flatten) >= target_per_category):
            break

        rec = ds.records[idx]
        board_size = rec["board_size"]
        if board_size not in board_sizes:
            continue

        tps = rec["tps"]
        game = tak_python.PyGameState(board_size, tps)

        if game.is_terminal() or game.legal_move_count() < 2:
            continue

        labels = game.get_tactical_labels()
        stm = game.side_to_move()

        # Road win in 1 for STM
        stm_road = labels["road_in_1_white"] if stm == 0 else labels["road_in_1_black"]
        if stm_road and len(road_win) < target_per_category:
            winners = find_winning_moves(game)
            if winners:
                road_win.append({
                    "tps": tps,
                    "size": board_size,
                    "category": "road_win_in_1",
                    "expected_move_indices": winners,
                })

        # Forced road block: opponent has road-in-1, STM must defend
        opp = 1 - stm
        opp_road = labels["road_in_1_white"] if opp == 0 else labels["road_in_1_black"]
        if opp_road and not stm_road and len(forced_block) < target_per_category:
            blockers = find_blocking_moves(game)
            n_legal = game.legal_move_count()
            # Only interesting if not all moves block (i.e., there's a wrong answer)
            if 0 < len(blockers) < n_legal:
                forced_block.append({
                    "tps": tps,
                    "size": board_size,
                    "category": "forced_road_block",
                    "expected_move_indices": blockers,
                })

        # Capstone flatten
        if labels["capstone_flatten"] and len(capstone_flatten) < target_per_category:
            flatten_moves = find_capstone_flatten_moves(game)
            if flatten_moves:
                capstone_flatten.append({
                    "tps": tps,
                    "size": board_size,
                    "category": "capstone_flatten",
                    "expected_move_indices": flatten_moves,
                })

        if (count + 1) % 5000 == 0:
            print(f"  Scanned {count + 1}: road_win={len(road_win)}, "
                  f"forced_block={len(forced_block)}, cap_flatten={len(capstone_flatten)}")

    os.makedirs(out_dir, exist_ok=True)

    for name, positions in [("road_win_in_1", road_win),
                            ("forced_road_block", forced_block),
                            ("capstone_flatten", capstone_flatten)]:
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, 'w') as f:
            for pos in positions:
                f.write(json.dumps(pos) + "\n")
        print(f"Wrote {len(positions)} positions to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate tactical test suites")
    parser.add_argument("--shards", default="shards", help="Directory with .tknn files")
    parser.add_argument("--out", default="taknn/eval/suites", help="Output directory for JSONL files")
    parser.add_argument("--target", type=int, default=60, help="Target positions per category")
    args = parser.parse_args()

    import glob
    shard_files = sorted(glob.glob(os.path.join(args.shards, "*.tknn")))
    if not shard_files:
        print(f"No .tknn files found in {args.shards}")
        return
    print(f"Found {len(shard_files)} shard files")

    generate_suites(shard_files, args.out, target_per_category=args.target)


if __name__ == "__main__":
    main()
