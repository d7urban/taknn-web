/*!
Trainer — full training loop with AdamW, cosine warmup LR, multi-head loss.

Follows HexZero's `trainer.rs` pattern: `Trainer` struct owns VarStore +
optimizer, `train_step()` computes all losses and calls `backward_step_clip`.
*/

use std::f64::consts::PI;
use std::time::Instant;

use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::checkpoint::{self, TrainingState};
use crate::config::TrainConfig;
use crate::data::{ShardLoader, TrainingBatch};
use crate::net::TakNet;
use crate::policy::PolicyScorer;

// ---------------------------------------------------------------------------
// LR schedule
// ---------------------------------------------------------------------------

/// Cosine annealing with linear warmup.
pub fn cosine_warmup_lr(
    step: usize,
    warmup_steps: usize,
    total_steps: usize,
    base_lr: f64,
) -> f64 {
    if step < warmup_steps {
        return base_lr * step as f64 / warmup_steps.max(1) as f64;
    }
    let progress =
        (step - warmup_steps) as f64 / (total_steps.saturating_sub(warmup_steps)).max(1) as f64;
    base_lr * 0.5 * (1.0 + (PI * progress).cos())
}

// ---------------------------------------------------------------------------
// Loss computation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LossDict {
    pub total: f64,
    pub policy: f64,
    pub wdl: f64,
    pub margin: f64,
    pub road: f64,
    pub block: f64,
    pub cap: f64,
    pub endgame: f64,
}

/// Compute all losses for a training batch. Returns `(total_loss_tensor, LossDict)`.
fn compute_losses(
    net: &TakNet,
    policy: &PolicyScorer,
    batch: &TrainingBatch,
    config: &TrainConfig,
    train: bool,
) -> (Tensor, LossDict) {
    let out = net.forward_t(&batch.board_tensor, &batch.size_id, train);

    // Policy loss: cross-entropy with dense target
    let logits = policy.score_moves(
        &out.spatial,
        &out.global_pool,
        &batch.descriptors,
        &batch.num_moves,
    );
    let loss_policy = compute_policy_loss(&logits, &batch.policy_target, &batch.num_moves);

    // WDL loss: soft cross-entropy -Σ target * log(pred)
    let loss_wdl = -(&batch.wdl * (out.wdl + 1e-8).log())
        .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
        .mean(Kind::Float);

    // Margin loss: MSE
    let loss_margin = out.margin.mse_loss(&batch.margin, tch::Reduction::Mean);

    // Aux losses: binary cross-entropy (explicit f32, no autocast issues)
    let loss_road = binary_cross_entropy(&out.road, &batch.road_threat);
    let loss_block = binary_cross_entropy(&out.block, &batch.block_threat);
    let loss_cap = binary_cross_entropy(&out.cap, &batch.cap_flatten);
    let loss_endgame = binary_cross_entropy(&out.endgame, &batch.endgame);

    let total = config.w_policy * &loss_policy
        + config.w_wdl * &loss_wdl
        + config.w_margin * &loss_margin
        + config.w_road * &loss_road
        + config.w_block * &loss_block
        + config.w_cap * &loss_cap
        + config.w_endgame * &loss_endgame;

    let dict = LossDict {
        total: total.double_value(&[]),
        policy: loss_policy.double_value(&[]),
        wdl: loss_wdl.double_value(&[]),
        margin: loss_margin.double_value(&[]),
        road: loss_road.double_value(&[]),
        block: loss_block.double_value(&[]),
        cap: loss_cap.double_value(&[]),
        endgame: loss_endgame.double_value(&[]),
    };

    (total, dict)
}

/// Policy cross-entropy: -Σ target * log_softmax(logits), averaged over
/// samples that have valid targets.
fn compute_policy_loss(logits: &Tensor, policy_target: &Tensor, _num_moves: &Tensor) -> Tensor {
    // Check if any sample has a valid target
    let has_target = policy_target
        .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
        .gt(0.0);

    if !bool::try_from(has_target.any()).unwrap_or(false) {
        return Tensor::from(0.0f32)
            .to(logits.device())
            .set_requires_grad(true);
    }

    // Safe logits: replace -inf with -1e9 for log_softmax stability
    let safe_logits = logits
        .to_kind(Kind::Float)
        .clamp(f64::from(-1e9f32), f64::from(1e9f32));
    let log_probs = safe_logits.log_softmax(1, Kind::Float);

    // Per-sample CE: -Σ_m target_m * log_prob_m
    let loss_per_sample = -(policy_target.to_kind(Kind::Float) * &log_probs)
        .sum_dim_intlist([1i64].as_slice(), false, Kind::Float);

    // Average only over samples with valid targets
    loss_per_sample.masked_select(&has_target).mean(Kind::Float)
}

/// Binary cross-entropy: -[t*log(p) + (1-t)*log(1-p)], element-wise mean.
fn binary_cross_entropy(pred: &Tensor, target: &Tensor) -> Tensor {
    let p = pred.to_kind(Kind::Float).clamp(1e-7, 1.0 - 1e-7);
    let t = target.to_kind(Kind::Float);
    let one_minus_t: Tensor = 1.0 - &t;
    let one_minus_p: Tensor = 1.0 - &p;
    -(&t * p.log() + one_minus_t * one_minus_p.log()).mean(Kind::Float)
}

// ---------------------------------------------------------------------------
// EpochStats
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EpochStats {
    pub avg_loss: LossDict,
    pub n_batches: usize,
}

impl EpochStats {
    fn new() -> Self {
        EpochStats {
            avg_loss: LossDict {
                total: 0.0,
                policy: 0.0,
                wdl: 0.0,
                margin: 0.0,
                road: 0.0,
                block: 0.0,
                cap: 0.0,
                endgame: 0.0,
            },
            n_batches: 0,
        }
    }

    fn accumulate(&mut self, d: &LossDict) {
        self.avg_loss.total += d.total;
        self.avg_loss.policy += d.policy;
        self.avg_loss.wdl += d.wdl;
        self.avg_loss.margin += d.margin;
        self.avg_loss.road += d.road;
        self.avg_loss.block += d.block;
        self.avg_loss.cap += d.cap;
        self.avg_loss.endgame += d.endgame;
        self.n_batches += 1;
    }

    fn finalize(&mut self) {
        let n = self.n_batches.max(1) as f64;
        self.avg_loss.total /= n;
        self.avg_loss.policy /= n;
        self.avg_loss.wdl /= n;
        self.avg_loss.margin /= n;
        self.avg_loss.road /= n;
        self.avg_loss.block /= n;
        self.avg_loss.cap /= n;
        self.avg_loss.endgame /= n;
    }
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

pub struct Trainer {
    pub vs: nn::VarStore,
    pub net: TakNet,
    pub policy: PolicyScorer,
    pub optimizer: nn::Optimizer,
    pub config: TrainConfig,
    pub global_step: usize,
    pub best_val_loss: f64,
    pub device: Device,
}

impl Trainer {
    /// Build a fresh trainer with randomly-initialized weights.
    pub fn new(config: TrainConfig, device: Device) -> anyhow::Result<Self> {
        let vs = nn::VarStore::new(device);
        let net = TakNet::new(&vs, &config.net);
        let policy_scorer = PolicyScorer::new(&vs, &config.net);
        let optimizer = nn::Adam {
            wd: config.weight_decay,
            ..Default::default()
        }
        .build(&vs, config.learning_rate)?;

        let param_count: i64 = vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel() as i64)
            .sum();
        println!(
            "Model: {}ch x {} blocks, {param_count} params",
            config.net.channels, config.net.num_blocks
        );

        Ok(Trainer {
            vs,
            net,
            policy: policy_scorer,
            optimizer,
            config,
            global_step: 0,
            best_val_loss: f64::INFINITY,
            device,
        })
    }

    /// Run the full training loop.
    pub fn train(
        &mut self,
        train_loader: &mut ShardLoader,
        val_loader: &mut ShardLoader,
    ) -> f64 {
        let total_steps = self.config.epochs * train_loader.num_batches();
        println!(
            "Training: {} epochs, {} batches/epoch, {total_steps} total steps",
            self.config.epochs,
            train_loader.num_batches()
        );

        for epoch in 1..=self.config.epochs {
            let t0 = Instant::now();

            let train_stats = self.train_epoch(train_loader, total_steps);
            let val_stats = self.validate(val_loader);

            let dt = t0.elapsed().as_secs_f64();
            let lr = cosine_warmup_lr(
                self.global_step,
                self.config.warmup_steps,
                total_steps,
                self.config.learning_rate,
            );

            println!(
                "Epoch {:3}/{} | train {:.4} (policy {:.4} wdl {:.4} margin {:.4}) \
                 | val {:.4} | lr {:.1e} | {:.1}s",
                epoch,
                self.config.epochs,
                train_stats.avg_loss.total,
                train_stats.avg_loss.policy,
                train_stats.avg_loss.wdl,
                train_stats.avg_loss.margin,
                val_stats.avg_loss.total,
                lr,
                dt
            );

            // Checkpoint on improvement
            if val_stats.avg_loss.total < self.best_val_loss {
                self.best_val_loss = val_stats.avg_loss.total;
                let state = TrainingState {
                    epoch,
                    global_step: self.global_step,
                    best_val_loss: self.best_val_loss,
                    net_config: Some(self.config.net.clone()),
                    model_type: if self.config.net.channels >= 128 {
                        "teacher".into()
                    } else {
                        "student".into()
                    },
                };
                if let Err(e) = checkpoint::save_checkpoint(
                    &self.vs,
                    &state,
                    &self.config.checkpoint_dir,
                    "best",
                ) {
                    eprintln!("  Warning: checkpoint save failed: {e}");
                } else {
                    println!("  -> Saved best model (val_loss={:.4})", self.best_val_loss);
                }
            }
        }

        println!("\nBest val loss: {:.4}", self.best_val_loss);
        self.best_val_loss
    }

    fn train_epoch(
        &mut self,
        loader: &mut ShardLoader,
        total_steps: usize,
    ) -> EpochStats {
        loader.shuffle();
        let mut stats = EpochStats::new();

        while let Some(batch) = loader.next_batch(self.device) {
            // Update LR
            let lr = cosine_warmup_lr(
                self.global_step,
                self.config.warmup_steps,
                total_steps,
                self.config.learning_rate,
            );
            self.optimizer.set_lr(lr);

            // Forward + loss
            let (loss, loss_dict) =
                compute_losses(&self.net, &self.policy, &batch, &self.config, true);

            // Backward + clip + step
            self.optimizer
                .backward_step_clip(&loss, self.config.gradient_clip);
            self.global_step += 1;

            stats.accumulate(&loss_dict);
        }

        stats.finalize();
        stats
    }

    fn validate(&self, loader: &mut ShardLoader) -> EpochStats {
        loader.reset();
        let mut stats = EpochStats::new();

        tch::no_grad(|| {
            while let Some(batch) = loader.next_batch(self.device) {
                let (_, loss_dict) =
                    compute_losses(&self.net, &self.policy, &batch, &self.config, false);
                stats.accumulate(&loss_dict);
            }
        });

        stats.finalize();
        stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_warmup_lr_at_zero_is_zero() {
        let lr = cosine_warmup_lr(0, 100, 1000, 1e-3);
        assert!((lr - 0.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_warmup_lr_at_warmup_end_is_base() {
        let lr = cosine_warmup_lr(100, 100, 1000, 1e-3);
        assert!((lr - 1e-3).abs() < 1e-10);
    }

    #[test]
    fn cosine_warmup_lr_at_total_is_zero() {
        let lr = cosine_warmup_lr(1000, 100, 1000, 1e-3);
        assert!(lr < 1e-10);
    }

    #[test]
    fn cosine_warmup_lr_midpoint() {
        // At midpoint between warmup and total: cos(π/2) = 0 → base_lr/2
        let lr = cosine_warmup_lr(550, 100, 1000, 1e-3);
        let expected = 1e-3 * 0.5;
        assert!((lr - expected).abs() < 1e-10);
    }

    #[test]
    fn cosine_warmup_lr_monotone_after_warmup() {
        let mut prev = cosine_warmup_lr(100, 100, 1000, 1e-3);
        for step in 101..=1000 {
            let cur = cosine_warmup_lr(step, 100, 1000, 1e-3);
            assert!(cur <= prev + 1e-15, "LR increased at step {step}");
            prev = cur;
        }
    }
}
