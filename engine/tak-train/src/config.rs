use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Network architecture configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetConfig {
    /// Convolutional channels (128 teacher, 64 student).
    pub channels: i64,
    /// Number of residual blocks (10 teacher, 6 student).
    pub num_blocks: i64,
    /// FiLM embedding dimension (32 teacher, 16 student).
    pub film_embed_dim: i64,
    /// Number of board sizes supported (3..=8 → 6).
    pub num_sizes: i64,
}

impl NetConfig {
    pub fn teacher() -> Self {
        NetConfig {
            channels: 128,
            num_blocks: 10,
            film_embed_dim: 32,
            num_sizes: 6,
        }
    }

    pub fn student() -> Self {
        NetConfig {
            channels: 64,
            num_blocks: 6,
            film_embed_dim: 16,
            num_sizes: 6,
        }
    }

    /// Total discrete embedding dimensions (same for teacher and student).
    /// move_type(8) + piece_type(8) + direction(8) + pickup(16) + template(16) + travel(8) = 64
    pub fn discrete_embed_dim(&self) -> i64 {
        64
    }

    /// Policy MLP input dimension: global(C) + h_src(C) + h_dst(C) + path_pool(C) + discrete(64) + flags(3).
    pub fn policy_input_dim(&self) -> i64 {
        self.channels * 4 + self.discrete_embed_dim() + 3
    }

    /// Hidden size for the value head FC.
    pub fn value_hidden(&self) -> i64 {
        if self.channels >= 128 { 128 } else { 64 }
    }

    /// Hidden size for the policy MLP.
    pub fn policy_hidden(&self) -> i64 {
        if self.channels >= 128 { 256 } else { 128 }
    }
}

/// Training hyperparameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub net: NetConfig,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub warmup_steps: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub gradient_clip: f64,
    pub checkpoint_dir: PathBuf,
    pub val_split: f64,
    // Loss weights
    pub w_policy: f64,
    pub w_wdl: f64,
    pub w_margin: f64,
    pub w_road: f64,
    pub w_block: f64,
    pub w_cap: f64,
    pub w_endgame: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            net: NetConfig::teacher(),
            learning_rate: 2e-3,
            weight_decay: 1e-4,
            warmup_steps: 1000,
            epochs: 50,
            batch_size: 1024,
            gradient_clip: 1.0,
            checkpoint_dir: PathBuf::from("checkpoints"),
            val_split: 0.1,
            w_policy: 1.0,
            w_wdl: 1.0,
            w_margin: 0.5,
            w_road: 0.2,
            w_block: 0.2,
            w_cap: 0.1,
            w_endgame: 0.1,
        }
    }
}
