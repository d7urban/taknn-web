#!/usr/bin/env python3
"""Distill a trained teacher model into a smaller student.

Usage:
    python distill.py --teacher checkpoints/teacher_best.pt --shards shards/ --epochs 30
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
from taknn.train.distill import distill


def main():
    parser = argparse.ArgumentParser(description="Distill teacher → student")
    parser.add_argument("--teacher", required=True, help="Path to teacher checkpoint")
    parser.add_argument("--shards", default="shards", help="Directory with .tknn files")
    parser.add_argument("--student-channels", type=int, default=64)
    parser.add_argument("--student-blocks", type=int, default=6)
    parser.add_argument("--student-film-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load teacher
    t_ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    t_config = t_ckpt.get("config", {})
    teacher = TeacherModel(
        channels=t_config.get("channels", 128),
        num_blocks=t_config.get("num_blocks", 10),
        film_embed_dim=t_config.get("film_embed_dim", 32),
    ).to(device)
    teacher.load_state_dict(t_ckpt["model_state_dict"])
    teacher.eval()
    print(f"Teacher: {t_config.get('channels', 128)}ch x {t_config.get('num_blocks', 10)} blocks")

    # Config for student training
    config = TrainConfig()
    config.model_type = "student"
    config.channels = args.student_channels
    config.num_blocks = args.student_blocks
    config.film_embed_dim = args.student_film_dim
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.checkpoint_dir = args.checkpoint_dir
    config.val_split = args.val_split
    config.mixed_precision = not args.no_amp

    # Load data with descriptors for policy distillation
    shard_files = sorted(glob.glob(os.path.join(args.shards, "*.tknn")))
    if not shard_files:
        print(f"No .tknn files found in {args.shards}")
        return
    print(f"Found {len(shard_files)} shard files")

    dataset = ShardDataset(shard_files, load_descriptors=True, load_aux_labels=False)
    print(f"Loaded {len(dataset)} records")

    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}, Val: {val_size}")

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=use_cuda, collate_fn=collate_with_descriptors,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=use_cuda, collate_fn=collate_with_descriptors,
    )

    # Create student
    student = StudentModel(config.channels, config.num_blocks, config.film_embed_dim).to(device)
    s_params = sum(p.numel() for p in student.parameters())
    print(f"Student: {config.channels}ch x {config.num_blocks} blocks, {s_params:,} params")

    distill(teacher, student, train_loader, val_loader, config, device, temperature=args.temperature)


if __name__ == "__main__":
    main()
