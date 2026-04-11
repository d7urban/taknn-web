/*!
TakNet — Size-conditioned ResNet with FiLM for Tak.

Mirrors the Python architecture in `train/taknn/models/teacher.py` exactly,
so weights can be cross-validated between Python and Rust.

**Architecture**
```text
Input   (B, 31, 8, 8)
Stem    Conv(31→C, 3×3, pad=1, no bias) → BN → ReLU
Tower   N × FiLMResBlock(C, film_embed_dim)
        Size embedding: Embedding(6 → film_embed_dim), shared
Value   global_avg_pool → cat(size_embed)
        → Linear(C+E→H) → ReLU → Linear(H→3) → Softmax  [WDL]
        → Linear(H→1) → tanh                              [margin]
Aux     Conv(C→2, 1×1) → sigmoid  [road_threat]
        Conv(C→2, 1×1) → sigmoid  [block_threat]
        Conv(C→1, 1×1) → sigmoid  [cap_flatten]
        cat(g, e) → Linear(C+E→1) → sigmoid  [endgame]
```

Default teacher: C=128, N=10, E=32, H=128.
Default student: C=64, N=6, E=16, H=64.
*/

use tak_core::tensor::C_IN;
use tch::{
    nn::{self, Module, ModuleT},
    Kind, Tensor,
};

use crate::config::NetConfig;

// ---------------------------------------------------------------------------
// FiLMResBlock
// ---------------------------------------------------------------------------

/// Residual block with FiLM conditioning on board size.
///
/// Each sub-layer (conv1→bn1, conv2→bn2) gets its own (γ, β) from linear
/// projections of the size embedding.
struct FiLMResBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    gamma1: nn::Linear,
    beta1: nn::Linear,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    gamma2: nn::Linear,
    beta2: nn::Linear,
}

impl FiLMResBlock {
    fn new(p: &nn::Path<'_>, channels: i64, film_embed_dim: i64) -> Self {
        let conv_cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        FiLMResBlock {
            conv1: nn::conv2d(p / "conv1", channels, channels, 3, conv_cfg),
            bn1: nn::batch_norm2d(p / "bn1", channels, Default::default()),
            gamma1: nn::linear(p / "gamma1", film_embed_dim, channels, Default::default()),
            beta1: nn::linear(p / "beta1", film_embed_dim, channels, Default::default()),
            conv2: nn::conv2d(p / "conv2", channels, channels, 3, conv_cfg),
            bn2: nn::batch_norm2d(p / "bn2", channels, Default::default()),
            gamma2: nn::linear(p / "gamma2", film_embed_dim, channels, Default::default()),
            beta2: nn::linear(p / "beta2", film_embed_dim, channels, Default::default()),
        }
    }

    /// Forward with FiLM conditioning.
    ///
    /// * `x` — `[B, C, 8, 8]`
    /// * `e` — `[B, film_embed_dim]` size embedding
    /// * `train` — batch norm mode
    fn forward_t(&self, x: &Tensor, e: &Tensor, train: bool) -> Tensor {
        let residual = x.shallow_clone();

        // Sub-layer 1: conv → BN → FiLM → ReLU
        let g1 = self.gamma1.forward(e); // [B, C]
        let b1 = self.beta1.forward(e);
        let h = self.conv1.forward(x);
        let h = self.bn1.forward_t(&h, train);
        let h = film(&h, &g1, &b1).relu();

        // Sub-layer 2: conv → BN → FiLM → add residual → ReLU
        let g2 = self.gamma2.forward(e);
        let b2 = self.beta2.forward(e);
        let h = self.conv2.forward(&h);
        let h = self.bn2.forward_t(&h, train);
        let h = film(&h, &g2, &b2);

        (h + residual).relu()
    }
}

/// FiLM: `x * gamma + beta` with spatial broadcast.
///
/// * `x` — `[B, C, H, W]`
/// * `gamma`, `beta` — `[B, C]`
fn film(x: &Tensor, gamma: &Tensor, beta: &Tensor) -> Tensor {
    let g = gamma.unsqueeze(-1).unsqueeze(-1); // [B, C, 1, 1]
    let b = beta.unsqueeze(-1).unsqueeze(-1);
    x * g + b
}

// ---------------------------------------------------------------------------
// TakNet output
// ---------------------------------------------------------------------------

pub struct TakNetOutput {
    /// Trunk spatial features `[B, C, 8, 8]`.
    pub spatial: Tensor,
    /// Global average pool `[B, C]`.
    pub global_pool: Tensor,
    /// Win/draw/loss probabilities `[B, 3]` (softmax applied).
    pub wdl: Tensor,
    /// Predicted flat margin `[B, 1]` (tanh applied).
    pub margin: Tensor,
    /// Road threat per square `[B, 2, 8, 8]` (sigmoid applied).
    pub road: Tensor,
    /// Block threat per square `[B, 2, 8, 8]` (sigmoid applied).
    pub block: Tensor,
    /// Capstone flatten per square `[B, 1, 8, 8]` (sigmoid applied).
    pub cap: Tensor,
    /// Endgame scalar `[B, 1]` (sigmoid applied).
    pub endgame: Tensor,
}

// ---------------------------------------------------------------------------
// TakNet
// ---------------------------------------------------------------------------

pub struct TakNet {
    // Stem
    stem_conv: nn::Conv2D,
    stem_bn: nn::BatchNorm,
    // Size conditioning
    size_embed: nn::Embedding,
    // Trunk
    blocks: Vec<FiLMResBlock>,
    // Value head
    v_hidden: nn::Linear,
    wdl_head: nn::Linear,
    margin_head: nn::Linear,
    // Aux heads
    road_threat: nn::Conv2D,
    block_threat: nn::Conv2D,
    cap_flatten: nn::Conv2D,
    endgame_head: nn::Linear,
    // Config (for computing dims)
    config: NetConfig,
}

impl TakNet {
    /// Build the network, registering all parameters in `vs`.
    ///
    /// Layer paths match the Python `state_dict` naming:
    /// `stem.0.weight`, `blocks.0.conv1.weight`, etc.
    pub fn new(vs: &nn::VarStore, cfg: &NetConfig) -> Self {
        let p = vs.root();
        let c = cfg.channels;
        let e = cfg.film_embed_dim;

        let no_bias_pad1 = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };

        // Stem: Conv → BN (ReLU applied in forward)
        let stem_conv = nn::conv2d(&p / "stem" / "0", C_IN as i64, c, 3, no_bias_pad1);
        let stem_bn = nn::batch_norm2d(&p / "stem" / "1", c, Default::default());

        // Size embedding
        let size_embed = nn::embedding(&p / "size_embed", cfg.num_sizes, e, Default::default());

        // Residual tower
        let blocks_p = &p / "blocks";
        let blocks = (0..cfg.num_blocks)
            .map(|i| FiLMResBlock::new(&(&blocks_p / &i.to_string()), c, e))
            .collect();

        // Value head
        let vh = cfg.value_hidden();
        let v_hidden = nn::linear(&p / "v_hidden", c + e, vh, Default::default());
        let wdl_head = nn::linear(&p / "wdl_head", vh, 3, Default::default());
        let margin_head = nn::linear(&p / "margin_head", vh, 1, Default::default());

        // Aux heads
        let road_threat = nn::conv2d(&p / "road_threat", c, 2, 1, Default::default());
        let block_threat = nn::conv2d(&p / "block_threat", c, 2, 1, Default::default());
        let cap_flatten = nn::conv2d(&p / "cap_flatten", c, 1, 1, Default::default());
        let endgame_head = nn::linear(&p / "endgame_head", c + e, 1, Default::default());

        TakNet {
            stem_conv,
            stem_bn,
            size_embed,
            blocks,
            v_hidden,
            wdl_head,
            margin_head,
            road_threat,
            block_threat,
            cap_flatten,
            endgame_head,
            config: cfg.clone(),
        }
    }

    /// Forward pass.
    ///
    /// * `board` — `[B, 31, 8, 8]` float32
    /// * `size_id` — `[B]` int64 (0..5 for board sizes 3..8)
    /// * `train` — `true` during training (BN uses batch stats)
    pub fn forward_t(
        &self,
        board: &Tensor,
        size_id: &Tensor,
        train: bool,
    ) -> TakNetOutput {
        // Size embedding
        let e = self.size_embed.forward(size_id); // [B, E]

        // Stem
        let x = self.stem_conv.forward(board);
        let mut x = self.stem_bn.forward_t(&x, train).relu();

        // Trunk
        for block in &self.blocks {
            x = block.forward_t(&x, &e, train);
        }

        let h = x; // spatial: [B, C, 8, 8]
        let g = h.adaptive_avg_pool2d([1, 1]).flatten(1, -1); // [B, C]

        // Value head: cat(g, e) → FC → ReLU → heads
        let v_in = Tensor::cat(&[&g, &e], 1); // [B, C+E]
        let v_hid = self.v_hidden.forward(&v_in).relu();
        let wdl = self.wdl_head.forward(&v_hid).softmax(1, Kind::Float);
        let margin = self.margin_head.forward(&v_hid).tanh();

        // Aux heads
        let road = self.road_threat.forward(&h).sigmoid();
        let block = self.block_threat.forward(&h).sigmoid();
        let cap = self.cap_flatten.forward(&h).sigmoid();
        let endgame = self.endgame_head.forward(&v_in).sigmoid();

        TakNetOutput {
            spatial: h,
            global_pool: g,
            wdl,
            margin,
            road,
            block,
            cap,
            endgame,
        }
    }

    /// Inference convenience: wraps `forward_t(train=false)` in `no_grad`.
    pub fn infer(&self, board: &Tensor, size_id: &Tensor) -> TakNetOutput {
        tch::no_grad(|| self.forward_t(board, size_id, false))
    }

    pub fn config(&self) -> &NetConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    fn cpu_net(cfg: &NetConfig) -> (nn::VarStore, TakNet) {
        let vs = nn::VarStore::new(Device::Cpu);
        let net = TakNet::new(&vs, cfg);
        (vs, net)
    }

    #[test]
    fn build_teacher_without_panic() {
        let _ = cpu_net(&NetConfig::teacher());
    }

    #[test]
    fn build_student_without_panic() {
        let _ = cpu_net(&NetConfig::student());
    }

    #[test]
    fn teacher_forward_shapes() {
        let (_, net) = cpu_net(&NetConfig::teacher());
        let board = Tensor::zeros([2, C_IN as i64, 8, 8], (Kind::Float, Device::Cpu));
        let size_id = Tensor::zeros([2], (Kind::Int64, Device::Cpu));
        let out = net.forward_t(&board, &size_id, false);

        assert_eq!(out.spatial.size(), [2, 128, 8, 8]);
        assert_eq!(out.global_pool.size(), [2, 128]);
        assert_eq!(out.wdl.size(), [2, 3]);
        assert_eq!(out.margin.size(), [2, 1]);
        assert_eq!(out.road.size(), [2, 2, 8, 8]);
        assert_eq!(out.block.size(), [2, 2, 8, 8]);
        assert_eq!(out.cap.size(), [2, 1, 8, 8]);
        assert_eq!(out.endgame.size(), [2, 1]);
    }

    #[test]
    fn student_forward_shapes() {
        let (_, net) = cpu_net(&NetConfig::student());
        let board = Tensor::zeros([4, C_IN as i64, 8, 8], (Kind::Float, Device::Cpu));
        let size_id = Tensor::from_slice(&[0i64, 1, 2, 3]);
        let out = net.forward_t(&board, &size_id, false);

        assert_eq!(out.spatial.size(), [4, 64, 8, 8]);
        assert_eq!(out.global_pool.size(), [4, 64]);
        assert_eq!(out.wdl.size(), [4, 3]);
        assert_eq!(out.margin.size(), [4, 1]);
    }

    #[test]
    fn wdl_sums_to_one() {
        let (_, net) = cpu_net(&NetConfig::teacher());
        let board = Tensor::randn([1, C_IN as i64, 8, 8], (Kind::Float, Device::Cpu));
        let size_id = Tensor::zeros([1], (Kind::Int64, Device::Cpu));
        let out = net.forward_t(&board, &size_id, false);
        let sum: f64 = out.wdl.sum(Kind::Float).double_value(&[]);
        assert!((sum - 1.0).abs() < 1e-4, "WDL sum = {sum}");
    }

    #[test]
    fn margin_in_range() {
        let (_, net) = cpu_net(&NetConfig::teacher());
        let board = Tensor::randn([4, C_IN as i64, 8, 8], (Kind::Float, Device::Cpu));
        let size_id = Tensor::zeros([4], (Kind::Int64, Device::Cpu));
        let out = net.forward_t(&board, &size_id, false);
        let min: f64 = out.margin.min().double_value(&[]);
        let max: f64 = out.margin.max().double_value(&[]);
        assert!(min >= -1.0 - 1e-6, "margin below -1: {min}");
        assert!(max <= 1.0 + 1e-6, "margin above  1: {max}");
    }

    #[test]
    fn aux_heads_in_01() {
        let (_, net) = cpu_net(&NetConfig::student());
        let board = Tensor::randn([2, C_IN as i64, 8, 8], (Kind::Float, Device::Cpu));
        let size_id = Tensor::zeros([2], (Kind::Int64, Device::Cpu));
        let out = net.forward_t(&board, &size_id, false);

        for (name, t) in [
            ("road", &out.road),
            ("block", &out.block),
            ("cap", &out.cap),
            ("endgame", &out.endgame),
        ] {
            let min: f64 = t.min().double_value(&[]);
            let max: f64 = t.max().double_value(&[]);
            assert!(min >= -1e-6, "{name} min below 0: {min}");
            assert!(max <= 1.0 + 1e-6, "{name} max above 1: {max}");
        }
    }

    #[test]
    fn eval_mode_is_deterministic() {
        let (_, net) = cpu_net(&NetConfig::teacher());
        let board = Tensor::randn([1, C_IN as i64, 8, 8], (Kind::Float, Device::Cpu));
        let size_id = Tensor::zeros([1], (Kind::Int64, Device::Cpu));
        let o1 = net.forward_t(&board, &size_id, false);
        let o2 = net.forward_t(&board, &size_id, false);
        let diff: f64 = (&o1.wdl - &o2.wdl).abs().max().double_value(&[]);
        assert!(diff < 1e-6, "not deterministic: diff={diff}");
    }

    #[test]
    fn teacher_param_count() {
        let (vs, _) = cpu_net(&NetConfig::teacher());
        let count: i64 = vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel() as i64)
            .sum();
        // Sanity: teacher should have ~3.3M params
        assert!(count > 3_000_000, "teacher too small: {count}");
        assert!(count < 4_000_000, "teacher too large: {count}");
    }

    #[test]
    fn student_param_count() {
        let (vs, _) = cpu_net(&NetConfig::student());
        let count: i64 = vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel() as i64)
            .sum();
        // Sanity: student should have ~540K params
        assert!(count > 400_000, "student too small: {count}");
        assert!(count < 700_000, "student too large: {count}");
    }
}
