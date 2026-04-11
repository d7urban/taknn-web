/*!
Knowledge distillation: teacher → student.

Loss:
- `0.7 * KL(student_wdl / T, teacher_wdl / T) * T²`
- `0.3 * CE(student_wdl, game_result)`
- `0.3 * MSE(student_margin, teacher_margin)`
- `1.0 * KL(student_policy / T, teacher_policy / T) * T²`

The teacher is frozen — only the student trains.
*/

use std::time::Instant;

use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::checkpoint::{self, TrainingState};
use crate::config::{NetConfig, TrainConfig};
use crate::data::{ShardLoader, TrainingBatch};
use crate::net::TakNet;
use crate::policy::PolicyScorer;
use crate::trainer::cosine_warmup_lr;

// ---------------------------------------------------------------------------
// DistillTrainer
// ---------------------------------------------------------------------------

pub struct DistillTrainer {
    // Teacher (frozen)
    pub teacher_vs: nn::VarStore,
    pub teacher: TakNet,
    pub teacher_policy: PolicyScorer,
    // Student (trainable)
    pub student_vs: nn::VarStore,
    pub student: TakNet,
    pub student_policy: PolicyScorer,
    pub optimizer: nn::Optimizer,
    // Config
    pub config: TrainConfig,
    pub temperature: f64,
    pub global_step: usize,
    pub best_val_loss: f64,
    pub device: Device,
}

impl DistillTrainer {
    /// Build a distillation trainer.
    ///
    /// * `teacher_cfg` — teacher network config
    /// * `student_cfg` — student network config (passed via `config.net`)
    /// * `config` — training hyperparameters (uses `config.net` for student dims)
    pub fn new(
        teacher_cfg: &NetConfig,
        config: TrainConfig,
        device: Device,
        temperature: f64,
    ) -> anyhow::Result<Self> {
        // Teacher
        let teacher_vs = nn::VarStore::new(device);
        let teacher = TakNet::new(&teacher_vs, teacher_cfg);
        let teacher_policy = PolicyScorer::new(&teacher_vs, teacher_cfg);

        // Student
        let student_vs = nn::VarStore::new(device);
        let student = TakNet::new(&student_vs, &config.net);
        let student_policy = PolicyScorer::new(&student_vs, &config.net);
        let optimizer = nn::Adam {
            wd: config.weight_decay,
            ..Default::default()
        }
        .build(&student_vs, config.learning_rate)?;

        let t_params: i64 = teacher_vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel() as i64)
            .sum();
        let s_params: i64 = student_vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel() as i64)
            .sum();
        println!(
            "Teacher: {t_params} params, Student: {s_params} params ({:.1}%)",
            100.0 * s_params as f64 / t_params as f64
        );

        Ok(DistillTrainer {
            teacher_vs,
            teacher,
            teacher_policy,
            student_vs,
            student,
            student_policy,
            optimizer,
            config,
            temperature,
            global_step: 0,
            best_val_loss: f64::INFINITY,
            device,
        })
    }

    /// Load teacher weights from a checkpoint.
    pub fn load_teacher(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        let missing = self.teacher_vs.load_partial(path)?;
        if !missing.is_empty() {
            eprintln!(
                "Warning: {} missing keys when loading teacher",
                missing.len()
            );
        }
        // Freeze teacher by setting requires_grad = false
        self.teacher_vs.freeze();
        Ok(())
    }

    /// Run the full distillation loop.
    pub fn distill(
        &mut self,
        train_loader: &mut ShardLoader,
        val_loader: &mut ShardLoader,
    ) -> f64 {
        let total_steps = self.config.epochs * train_loader.num_batches();
        println!(
            "Distillation: {} epochs, {} batches/epoch, T={}",
            self.config.epochs,
            train_loader.num_batches(),
            self.temperature
        );

        for epoch in 1..=self.config.epochs {
            let t0 = Instant::now();
            train_loader.shuffle();

            let mut train_loss = 0.0;
            let mut n_batches = 0;

            while let Some(batch) = train_loader.next_batch(self.device) {
                let lr = cosine_warmup_lr(
                    self.global_step,
                    self.config.warmup_steps,
                    total_steps,
                    self.config.learning_rate,
                );
                self.optimizer.set_lr(lr);

                let loss = self.distill_step(&batch);
                train_loss += loss;
                n_batches += 1;
                self.global_step += 1;
            }

            train_loss /= n_batches.max(1) as f64;

            // Validation
            val_loader.reset();
            let mut val_loss = 0.0;
            let mut val_n = 0;
            tch::no_grad(|| {
                while let Some(batch) = val_loader.next_batch(self.device) {
                    val_loss += self.compute_distill_loss(&batch, false).0;
                    val_n += 1;
                }
            });
            val_loss /= val_n.max(1) as f64;

            let dt = t0.elapsed().as_secs_f64();
            let lr = cosine_warmup_lr(
                self.global_step,
                self.config.warmup_steps,
                total_steps,
                self.config.learning_rate,
            );

            println!(
                "Epoch {:3}/{} | train {:.4} | val {:.4} | lr {:.1e} | {:.1}s",
                epoch, self.config.epochs, train_loss, val_loss, lr, dt
            );

            if val_loss < self.best_val_loss {
                self.best_val_loss = val_loss;
                let state = TrainingState {
                    epoch,
                    global_step: self.global_step,
                    best_val_loss: self.best_val_loss,
                    net_config: Some(self.config.net.clone()),
                    model_type: "student".into(),
                };
                if let Err(e) = checkpoint::save_checkpoint(
                    &self.student_vs,
                    &state,
                    &self.config.checkpoint_dir,
                    "student_best",
                ) {
                    eprintln!("  Warning: checkpoint save failed: {e}");
                } else {
                    println!("  -> Saved best student (val_loss={:.4})", self.best_val_loss);
                }
            }
        }

        println!("\nBest val loss: {:.4}", self.best_val_loss);
        self.best_val_loss
    }

    fn distill_step(&mut self, batch: &TrainingBatch) -> f64 {
        let (loss_val, loss_tensor) = self.compute_distill_loss(batch, true);
        self.optimizer
            .backward_step_clip(&loss_tensor, self.config.gradient_clip);
        loss_val
    }

    fn compute_distill_loss(&self, batch: &TrainingBatch, train: bool) -> (f64, Tensor) {
        let t = self.temperature;

        // Teacher forward (no grad — already frozen, but be explicit)
        let t_out = tch::no_grad(|| {
            self.teacher
                .forward_t(&batch.board_tensor, &batch.size_id, false)
        });

        // Student forward
        let s_out = self.student.forward_t(&batch.board_tensor, &batch.size_id, train);

        // WDL distillation: KL divergence with temperature
        let t_wdl_logits = (&t_out.wdl + 1e-8).log() / t;
        let s_wdl_logits = (&s_out.wdl + 1e-8).log() / t;
        let t_wdl_soft = t_wdl_logits.softmax(1, Kind::Float);
        let s_wdl_log_soft = s_wdl_logits.log_softmax(1, Kind::Float);
        let loss_wdl_kl = Tensor::kl_div(&s_wdl_log_soft, &t_wdl_soft, tch::Reduction::Mean, false)
            * (t * t);

        // Hard WDL target from game result
        let loss_wdl_hard = -(&batch.wdl * (&s_out.wdl + 1e-8).log())
            .sum_dim_intlist([1i64].as_slice(), false, Kind::Float)
            .mean(Kind::Float);

        // Margin distillation: MSE to teacher
        let loss_margin = s_out.margin.mse_loss(
            &t_out.margin.detach(),
            tch::Reduction::Mean,
        );

        let mut total: Tensor = 0.7 * &loss_wdl_kl + 0.3 * &loss_wdl_hard + 0.3 * &loss_margin;

        // Policy distillation (if descriptors present with moves)
        let has_moves = bool::try_from(batch.num_moves.gt(0).any()).unwrap_or(false);
        if has_moves {
            let t_logits = tch::no_grad(|| {
                self.teacher_policy.score_moves(
                    &t_out.spatial,
                    &t_out.global_pool,
                    &batch.descriptors,
                    &batch.num_moves,
                )
            });
            let s_logits = self.student_policy.score_moves(
                &s_out.spatial,
                &s_out.global_pool,
                &batch.descriptors,
                &batch.num_moves,
            );

            // Mask invalid positions
            let m = s_logits.size()[1];
            let move_mask = Tensor::arange(m, (Kind::Int64, s_logits.device()))
                .unsqueeze(0)
                .lt_tensor(&batch.num_moves.unsqueeze(1));

            let safe_t = t_logits.masked_fill(&move_mask.logical_not(), -1e9) / t;
            let safe_s = s_logits.masked_fill(&move_mask.logical_not(), -1e9) / t;

            let t_policy = safe_t.softmax(1, Kind::Float);
            let s_log_policy = safe_s.log_softmax(1, Kind::Float);

            let kl_per_sample = Tensor::kl_div(&s_log_policy, &t_policy, tch::Reduction::None, false)
                .sum_dim_intlist([1i64].as_slice(), false, Kind::Float);

            let has_moves_mask = batch.num_moves.gt(0);
            let loss_policy = kl_per_sample.masked_select(&has_moves_mask).mean(Kind::Float) * (t * t);

            total += &loss_policy;
        }

        let val = total.double_value(&[]);
        (val, total)
    }
}
