#!/usr/bin/env python3
"""Train a teacher value model on self-play data.

Trains WDL + margin heads. Policy is deferred for now.
The trained model can replace HeuristicEval in PVS for stronger play.
"""

import argparse
import glob
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import tak_python


class ValueDataset(Dataset):
    """Load shard records, convert game_result to WDL target."""

    def __init__(self, shard_paths):
        self.records = []
        for path in shard_paths:
            reader = tak_python.PyShardReader(path)
            while True:
                rec = reader.next()
                if rec is None:
                    break
                # Skip ongoing games
                if rec["game_result"] == 255:
                    continue
                self.records.append(rec)
        print(f"Loaded {len(self.records)} records from {len(shard_paths)} shards")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        board_tensor = torch.from_numpy(rec["board_tensor"].copy())
        size_id = rec["board_size"] - 3  # 0-indexed: 4x4 -> 1

        # WDL target: blend search-derived teacher_wdl with game outcome.
        # teacher_wdl is from STM perspective (set by selfplay engine).
        result = rec["game_result"]
        stm = rec["side_to_move"]  # 0=White, 1=Black

        # Game outcome WDL
        white_wins = result in (0, 2)
        black_wins = result in (1, 3)
        draw = result == 4
        if draw:
            game_wdl = [0.0, 1.0, 0.0]
        elif (white_wins and stm == 0) or (black_wins and stm == 1):
            game_wdl = [1.0, 0.0, 0.0]
        else:
            game_wdl = [0.0, 0.0, 1.0]

        # Search-derived WDL (from teacher_wdl field)
        search_wdl = list(rec["teacher_wdl"])
        # Check if search_wdl is placeholder (all ~0.33)
        is_placeholder = all(abs(w - 0.33) < 0.01 for w in search_wdl)

        if is_placeholder:
            wdl = game_wdl
        else:
            # Blend: 70% search-derived, 30% game outcome
            wdl = [0.7 * s + 0.3 * g for s, g in zip(search_wdl, game_wdl)]

        # Margin: prefer search-derived teacher_margin
        margin_norm = rec["teacher_margin"]
        if is_placeholder:
            margin = rec["flat_margin"]
            if stm == 1:
                margin = -margin
            margin_norm = max(-1.0, min(1.0, margin / 16.0))

        return {
            "board_tensor": board_tensor,
            "size_id": torch.tensor(size_id, dtype=torch.long),
            "wdl": torch.tensor(wdl, dtype=torch.float32),
            "margin": torch.tensor([margin_norm], dtype=torch.float32),
        }


class ValueModel(nn.Module):
    """Simplified teacher: residual trunk + WDL/margin heads."""

    def __init__(self, channels=128, num_blocks=10, film_dim=32):
        super().__init__()
        self.size_embed = nn.Embedding(6, film_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(31, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResBlock(channels, film_dim))

        self.v_hidden = nn.Linear(channels + film_dim, 256)
        self.wdl_head = nn.Linear(256, 3)
        self.margin_head = nn.Linear(256, 1)

    def forward(self, x, size_id):
        e = self.size_embed(size_id)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x, e)
        g = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        v = F.relu(self.v_hidden(torch.cat([g, e], dim=1)))
        wdl = F.softmax(self.wdl_head(v), dim=1)
        margin = torch.tanh(self.margin_head(v))
        return wdl, margin


class ResBlock(nn.Module):
    def __init__(self, ch, film_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
        self.gamma1 = nn.Linear(film_dim, ch)
        self.beta1 = nn.Linear(film_dim, ch)
        self.gamma2 = nn.Linear(film_dim, ch)
        self.beta2 = nn.Linear(film_dim, ch)

    def forward(self, x, e):
        res = x
        g1 = self.gamma1(e).unsqueeze(-1).unsqueeze(-1)
        b1 = self.beta1(e).unsqueeze(-1).unsqueeze(-1)
        out = F.relu(self.bn1(self.conv1(x)) * g1 + b1)
        g2 = self.gamma2(e).unsqueeze(-1).unsqueeze(-1)
        b2 = self.beta2(e).unsqueeze(-1).unsqueeze(-1)
        out = self.bn2(self.conv2(out)) * g2 + b2
        return F.relu(out + res)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_wdl_loss = 0
    total_margin_loss = 0
    n = 0

    for batch in loader:
        x = batch["board_tensor"].to(device)
        sid = batch["size_id"].to(device)
        target_wdl = batch["wdl"].to(device)
        target_margin = batch["margin"].to(device)

        wdl, margin = model(x, sid)

        # WDL: cross-entropy (using soft targets via NLL)
        loss_wdl = -torch.sum(target_wdl * torch.log(wdl + 1e-8)) / x.size(0)
        loss_margin = F.mse_loss(margin, target_margin)

        loss = loss_wdl + 0.5 * loss_margin

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_wdl_loss += loss_wdl.item() * x.size(0)
        total_margin_loss += loss_margin.item() * x.size(0)
        n += x.size(0)

    return total_loss / n, total_wdl_loss / n, total_margin_loss / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    n = 0

    for batch in loader:
        x = batch["board_tensor"].to(device)
        sid = batch["size_id"].to(device)
        target_wdl = batch["wdl"].to(device)
        target_margin = batch["margin"].to(device)

        wdl, margin = model(x, sid)

        loss_wdl = -torch.sum(target_wdl * torch.log(wdl + 1e-8)) / x.size(0)
        loss_margin = F.mse_loss(margin, target_margin)
        loss = loss_wdl + 0.5 * loss_margin

        total_loss += loss.item() * x.size(0)

        # Accuracy: predicted WDL class vs actual
        pred = wdl.argmax(dim=1)
        target = target_wdl.argmax(dim=1)
        correct += (pred == target).sum().item()
        n += x.size(0)

    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default="shards", help="Directory with .tknn files")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=64, help="Model channels (64 for small, 128 for full)")
    parser.add_argument("--blocks", type=int, default=6, help="Residual blocks (6 for small, 10 for full)")
    parser.add_argument("--output", default="checkpoints/teacher_4x4.pt")
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    shard_files = sorted(glob.glob(os.path.join(args.shards, "*.tknn")))
    if not shard_files:
        print(f"No .tknn files in {args.shards}")
        return
    print(f"Found {len(shard_files)} shard files")

    dataset = ValueDataset(shard_files)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Train: {train_size}, Val: {val_size}")

    use_cuda = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=use_cuda)

    model = ValueModel(channels=args.channels, num_blocks=args.blocks).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.channels}ch x {args.blocks} blocks, {param_count:,} params")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, wdl_loss, margin_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        scheduler.step()
        dt = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.4f} (wdl {wdl_loss:.4f} margin {margin_loss:.4f}) | "
            f"val {val_loss:.4f} acc {val_acc:.1%} | "
            f"lr {lr:.1e} | {dt:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "channels": args.channels,
                "blocks": args.blocks,
            }, args.output)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
