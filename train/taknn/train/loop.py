"""Full training loop for TakNN teacher/student models.

Supports policy + value + auxiliary head training with mixed precision,
cosine LR with warmup, and periodic validation/checkpointing.
"""

import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import TrainConfig


def cosine_warmup_lr(step, warmup_steps, total_steps, base_lr):
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_policy_loss(logits, policy_target, num_moves):
    """Cross-entropy between model policy and sparse teacher policy target.

    Args:
        logits: [B, M] raw logits from score_moves (padding = -inf)
        policy_target: [B, M] dense target distribution (from collate)
        num_moves: [B] actual move counts

    Returns:
        scalar loss
    """
    # Only count positions with valid policy targets (sum > 0)
    has_target = policy_target.sum(dim=1) > 0  # [B]
    if not has_target.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Masked log-softmax — replace -inf with large negative finite to avoid 0 * -inf = NaN
    safe_logits = logits.clone()
    safe_logits[safe_logits == float("-inf")] = -1e9
    log_probs = F.log_softmax(safe_logits, dim=1)  # [B, M]

    loss = -(policy_target * log_probs).sum(dim=1)  # [B]
    return loss[has_target].mean()


def compute_losses(model, batch, config, device):
    """Compute all losses for a single batch.

    Returns (total_loss, loss_dict) where loss_dict has per-component losses.
    """
    board_tensor = batch["board_tensor"].to(device)
    size_id = batch["size_id"].to(device)
    target_wdl = batch["wdl"].to(device)
    target_margin = batch["margin"].to(device)

    # Forward pass: trunk + value/aux heads
    outputs = model(board_tensor, size_id)

    # WDL loss: soft cross-entropy
    wdl_pred = outputs["wdl"]
    loss_wdl = -torch.sum(target_wdl * torch.log(wdl_pred + 1e-8), dim=1).mean()

    # Margin loss: MSE
    loss_margin = F.mse_loss(outputs["margin"], target_margin)

    losses = {"wdl": loss_wdl, "margin": loss_margin}
    total = config.w_wdl * loss_wdl + config.w_margin * loss_margin

    # Policy loss (if descriptors are present)
    if "descriptors" in batch:
        descs = {k: v.to(device) for k, v in batch["descriptors"].items()}
        num_moves = batch["num_moves"].to(device)
        policy_target = batch["policy_target"].to(device)

        logits = model.score_moves(outputs, descs, num_moves)
        loss_policy = compute_policy_loss(logits, policy_target, num_moves)
        losses["policy"] = loss_policy
        total = total + config.w_policy * loss_policy

    # Auxiliary losses (if labels are present)
    if "road_threat_label" in batch:
        rt_label = batch["road_threat_label"].to(device)
        bt_label = batch["block_threat_label"].to(device)
        cf_label = batch["cap_flatten_label"].to(device)
        eg_label = batch["endgame_label"].to(device)

        loss_road = F.binary_cross_entropy(outputs["road"], rt_label)
        loss_block = F.binary_cross_entropy(outputs["block"], bt_label)
        loss_cap = F.binary_cross_entropy(outputs["cap"], cf_label)
        loss_endgame = F.binary_cross_entropy(outputs["endgame"], eg_label)

        losses["road"] = loss_road
        losses["block"] = loss_block
        losses["cap"] = loss_cap
        losses["endgame"] = loss_endgame
        total = (total
                 + config.w_road * loss_road
                 + config.w_block * loss_block
                 + config.w_cap * loss_cap
                 + config.w_endgame * loss_endgame)

    losses["total"] = total
    return total, losses


def train_one_epoch(model, loader, optimizer, scaler, config, device, global_step, total_steps):
    """Train for one epoch, returning (avg_loss, loss_dict, global_step)."""
    model.train()
    running_loss = 0.0
    running_counts = {}
    n_batches = 0

    for batch in loader:
        # LR scheduling
        lr = cosine_warmup_lr(global_step, config.warmup_steps, total_steps, config.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()

        if config.mixed_precision and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                loss, loss_dict = compute_losses(model, batch, config, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, loss_dict = compute_losses(model, batch, config, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

        running_loss += loss.item()
        for k, v in loss_dict.items():
            running_counts[k] = running_counts.get(k, 0.0) + v.item()
        n_batches += 1
        global_step += 1

    avg_loss = running_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in running_counts.items()}
    return avg_loss, avg_components, global_step


@torch.no_grad()
def validate(model, loader, config, device):
    """Run validation, returning (avg_loss, loss_dict, wdl_accuracy)."""
    model.eval()
    running_loss = 0.0
    running_counts = {}
    correct = 0
    total = 0
    n_batches = 0

    for batch in loader:
        if config.mixed_precision and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                loss, loss_dict = compute_losses(model, batch, config, device)
        else:
            loss, loss_dict = compute_losses(model, batch, config, device)

        running_loss += loss.item()
        for k, v in loss_dict.items():
            running_counts[k] = running_counts.get(k, 0.0) + v.item()
        n_batches += 1

        # WDL accuracy
        outputs = model(batch["board_tensor"].to(device), batch["size_id"].to(device))
        pred = outputs["wdl"].argmax(dim=1).cpu()
        target = batch["wdl"].argmax(dim=1)
        correct += (pred == target).sum().item()
        total += pred.shape[0]

    avg_loss = running_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in running_counts.items()}
    accuracy = correct / max(total, 1)
    return avg_loss, avg_components, accuracy


def save_checkpoint(model, optimizer, scaler, epoch, global_step, val_loss, config, path):
    """Save training checkpoint."""
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "val_loss": val_loss,
        "config": {
            "channels": config.channels,
            "num_blocks": config.num_blocks,
            "film_embed_dim": config.film_embed_dim,
            "model_type": config.model_type,
        },
    }, path)


def train(model, train_loader, val_loader, config, device):
    """Full training loop."""
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if (config.mixed_precision and device.type == "cuda") else None

    total_steps = config.epochs * len(train_loader)
    global_step = 0
    best_val_loss = float("inf")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.channels}ch x {config.num_blocks} blocks, {param_count:,} params")
    print(f"Training: {config.epochs} epochs, {len(train_loader)} batches/epoch, {total_steps} total steps")

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        train_loss, train_components, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, config, device, global_step, total_steps
        )

        val_loss, val_components, val_acc = validate(model, val_loader, config, device)

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Log
        parts = [f"Epoch {epoch:3d}/{config.epochs}"]
        parts.append(f"train {train_loss:.4f}")
        for k in ["policy", "wdl", "margin"]:
            if k in train_components:
                parts.append(f"{k} {train_components[k]:.4f}")
        parts.append(f"| val {val_loss:.4f} acc {val_acc:.1%}")
        parts.append(f"| lr {lr:.1e} | {dt:.1f}s")
        print(" | ".join(parts))

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(config.checkpoint_dir, f"{config.model_type}_best.pt")
            save_checkpoint(model, optimizer, scaler, epoch, global_step, val_loss, config, path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    return best_val_loss
