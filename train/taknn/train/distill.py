"""Knowledge distillation: teacher → student.

Loss: KL(student_policy, teacher_policy)
    + 0.7 * KL(student_wdl, teacher_wdl)
    + 0.3 * MSE(student_margin, teacher_margin)
    + 0.3 * CE(student_wdl, game_result)

Teacher soft targets are generated on-the-fly from the teacher model.
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
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_distill_losses(teacher, student, batch, config, device, temperature=2.0):
    """Compute distillation losses for a single batch.

    Returns (total_loss, loss_dict).
    """
    board_tensor = batch["board_tensor"].to(device)
    size_id = batch["size_id"].to(device)

    # Teacher forward (no grad)
    with torch.no_grad():
        t_out = teacher(board_tensor, size_id)

    # Student forward
    s_out = student(board_tensor, size_id)

    losses = {}

    # WDL distillation: KL divergence with temperature
    t_wdl_logits = torch.log(t_out["wdl"] + 1e-8) / temperature
    s_wdl_logits = torch.log(s_out["wdl"] + 1e-8) / temperature
    t_wdl_soft = F.softmax(t_wdl_logits, dim=1)
    s_wdl_log_soft = F.log_softmax(s_wdl_logits, dim=1)
    loss_wdl_kl = F.kl_div(s_wdl_log_soft, t_wdl_soft, reduction="batchmean") * (temperature ** 2)
    losses["wdl_kl"] = loss_wdl_kl

    # Hard WDL target from game result (if available)
    loss_wdl_hard = torch.tensor(0.0, device=device, requires_grad=True)
    if "wdl" in batch:
        target_wdl = batch["wdl"].to(device)
        loss_wdl_hard = -torch.sum(target_wdl * torch.log(s_out["wdl"] + 1e-8), dim=1).mean()
        losses["wdl_hard"] = loss_wdl_hard

    # Margin distillation: MSE to teacher
    loss_margin = F.mse_loss(s_out["margin"], t_out["margin"].detach())
    losses["margin"] = loss_margin

    total = 0.7 * loss_wdl_kl + 0.3 * loss_wdl_hard + 0.3 * loss_margin

    # Policy distillation (if descriptors present)
    if "descriptors" in batch:
        descs = {k: v.to(device) for k, v in batch["descriptors"].items()}
        num_moves = batch["num_moves"].to(device)

        with torch.no_grad():
            t_logits = teacher.score_moves(t_out, descs, num_moves)

        s_logits = student.score_moves(s_out, descs, num_moves)

        # Mask for valid moves
        M = s_logits.shape[1]
        move_mask = torch.arange(M, device=device).unsqueeze(0) < num_moves.unsqueeze(1)

        # KL divergence on policy with temperature
        safe_t = t_logits.clone()
        safe_s = s_logits.clone()
        safe_t[~move_mask] = -1e9
        safe_s[~move_mask] = -1e9

        t_policy = F.softmax(safe_t / temperature, dim=1)
        s_log_policy = F.log_softmax(safe_s / temperature, dim=1)

        # Per-sample KL, averaged over samples that have valid targets
        kl_per_sample = F.kl_div(s_log_policy, t_policy, reduction="none").sum(dim=1)
        has_moves = num_moves > 0
        if has_moves.any():
            loss_policy = kl_per_sample[has_moves].mean() * (temperature ** 2)
        else:
            loss_policy = torch.tensor(0.0, device=device, requires_grad=True)

        losses["policy_kl"] = loss_policy
        total = total + loss_policy

    losses["total"] = total
    return total, losses


def distill(teacher, student, train_loader, val_loader, config, device, temperature=2.0):
    """Full distillation training loop."""
    teacher.eval()
    optimizer = optim.AdamW(student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if (config.mixed_precision and device.type == "cuda") else None

    total_steps = config.epochs * len(train_loader)
    global_step = 0
    best_val_loss = float("inf")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher: {t_params:,} params, Student: {s_params:,} params ({s_params/t_params:.1%})")
    print(f"Distillation: {config.epochs} epochs, {len(train_loader)} batches/epoch, T={temperature}")

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        student.train()
        running_loss = 0.0
        running_counts = {}
        n_batches = 0

        for batch in train_loader:
            lr = cosine_warmup_lr(global_step, config.warmup_steps, total_steps, config.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()

            if config.mixed_precision and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    loss, loss_dict = compute_distill_losses(
                        teacher, student, batch, config, device, temperature
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, loss_dict = compute_distill_losses(
                    teacher, student, batch, config, device, temperature
                )
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip)
                optimizer.step()

            running_loss += loss.item()
            for k, v in loss_dict.items():
                running_counts[k] = running_counts.get(k, 0.0) + v.item()
            n_batches += 1
            global_step += 1

        avg_loss = running_loss / max(n_batches, 1)
        avg_components = {k: v / max(n_batches, 1) for k, v in running_counts.items()}

        # Validation
        student.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                if config.mixed_precision and device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        loss, _ = compute_distill_losses(
                            teacher, student, batch, config, device, temperature
                        )
                else:
                    loss, _ = compute_distill_losses(
                        teacher, student, batch, config, device, temperature
                    )
                val_loss += loss.item()
                val_n += 1
        val_avg = val_loss / max(val_n, 1)

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        parts = [f"Epoch {epoch:3d}/{config.epochs}"]
        parts.append(f"train {avg_loss:.4f}")
        for k in ["policy_kl", "wdl_kl", "margin"]:
            if k in avg_components:
                parts.append(f"{k} {avg_components[k]:.4f}")
        parts.append(f"| val {val_avg:.4f}")
        parts.append(f"| lr {lr:.1e} | {dt:.1f}s")
        print(" | ".join(parts))

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            path = os.path.join(config.checkpoint_dir, "student_best.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "val_loss": val_avg,
                "config": {
                    "channels": config.channels,
                    "num_blocks": config.num_blocks,
                    "film_embed_dim": config.film_embed_dim,
                    "model_type": "student",
                },
            }, path)
            print(f"  -> Saved best student (val_loss={val_avg:.4f})")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    return best_val_loss
