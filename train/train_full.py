#!/usr/bin/env python3
"""Train a TakNN teacher or student model with policy + value + auxiliary heads.

Usage:
    python train_full.py --shards shards/ --epochs 50
    python train_full.py --shards shards/ --model student --channels 64 --blocks 6
"""

import argparse
import glob
import os

import torch
from torch.utils.data import DataLoader, random_split

from taknn.models.teacher import TeacherModel
from taknn.models.student import StudentModel
from taknn.data.shard import ShardDataset
from taknn.data.dataset import collate_with_descriptors
from taknn.train.config import TrainConfig
from taknn.train.loop import train


def main():
    parser = argparse.ArgumentParser(description="Train TakNN model")
    parser.add_argument("--shards", default="shards", help="Directory with .tknn files")
    parser.add_argument("--model", default="teacher", choices=["teacher", "student"])
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--no-policy", action="store_true", help="Skip policy training (value+aux only)")
    parser.add_argument("--no-aux", action="store_true", help="Skip auxiliary head training")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config
    config = TrainConfig()
    config.model_type = args.model
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.checkpoint_dir = args.checkpoint_dir
    config.val_split = args.val_split
    config.mixed_precision = not args.no_amp

    if args.model == "teacher":
        config.channels = args.channels or 128
        config.num_blocks = args.blocks or 10
        config.film_embed_dim = 32
    else:
        config.channels = args.channels or 64
        config.num_blocks = args.blocks or 6
        config.film_embed_dim = 16

    # If skipping policy/aux, zero out the weights
    if args.no_policy:
        config.w_policy = 0.0
    if args.no_aux:
        config.w_road = 0.0
        config.w_block = 0.0
        config.w_cap = 0.0
        config.w_endgame = 0.0

    # Load data
    shard_files = sorted(glob.glob(os.path.join(args.shards, "*.tknn")))
    if not shard_files:
        print(f"No .tknn files found in {args.shards}")
        return
    print(f"Found {len(shard_files)} shard files")

    load_descs = config.w_policy > 0
    load_aux = (config.w_road + config.w_block + config.w_cap + config.w_endgame) > 0

    dataset = ShardDataset(shard_files, load_descriptors=load_descs, load_aux_labels=load_aux)
    print(f"Loaded {len(dataset)} records (descriptors={load_descs}, aux_labels={load_aux})")

    # Split
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}, Val: {val_size}")

    collate_fn = collate_with_descriptors if load_descs else None
    use_cuda = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=use_cuda, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=use_cuda, collate_fn=collate_fn,
    )

    # Model
    if args.model == "teacher":
        model = TeacherModel(config.channels, config.num_blocks, config.film_embed_dim)
    else:
        model = StudentModel(config.channels, config.num_blocks, config.film_embed_dim)
    model = model.to(device)

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Resumed from {args.resume}")

    train(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
