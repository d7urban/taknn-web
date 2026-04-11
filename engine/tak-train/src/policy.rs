/*!
PolicyScorer — legal-move scorer for the policy head.

Separate from [`TakNet`] so the trunk can be exported to ONNX without the
variable-length policy logic.

The scorer takes trunk outputs (spatial features, global pool) plus batched
move descriptors and produces per-move logits.

## MLP input layout

For each move `m` in sample `b`:
```text
[global_pool(C), h_src(C), h_dst(C), path_pool(C), discrete(64), flags(3)]
```
Total: `4*C + 64 + 3` (teacher: 579, student: 323).
*/

use tch::{
    nn::{self, Module},
    Kind, Tensor,
};

use crate::config::NetConfig;

// ---------------------------------------------------------------------------
// Descriptor batch (padded tensors)
// ---------------------------------------------------------------------------

/// Batched, padded move descriptors ready for GPU.
///
/// All tensors have shape `[B, M]` (or `[B, M, 7]` for path) where
/// `M = max_moves` in the batch.
pub struct DescriptorBatch {
    pub src: Tensor,               // [B, M] u8 square indices (0..63)
    pub dst: Tensor,               // [B, M]
    pub path: Tensor,              // [B, M, 7] square indices; 255 = padding
    pub move_type: Tensor,         // [B, M] 0=place, 1=spread
    pub piece_type: Tensor,        // [B, M] 0..3
    pub direction: Tensor,         // [B, M] 0..4
    pub pickup_count: Tensor,      // [B, M] 0..8
    pub drop_template_id: Tensor,  // [B, M] 0..255
    pub travel_length: Tensor,     // [B, M] 0..7
    pub capstone_flatten: Tensor,  // [B, M] float 0/1
    pub enters_occupied: Tensor,   // [B, M] float 0/1
    pub opening_phase: Tensor,     // [B, M] float 0/1
}

// ---------------------------------------------------------------------------
// PolicyScorer
// ---------------------------------------------------------------------------

pub struct PolicyScorer {
    move_type_emb: nn::Embedding,
    piece_type_emb: nn::Embedding,
    direction_emb: nn::Embedding,
    pickup_count_emb: nn::Embedding,
    drop_template_emb: nn::Embedding,
    travel_length_emb: nn::Embedding,
    policy_mlp: nn::Sequential,
}

impl PolicyScorer {
    /// Build the policy scorer, registering parameters under `vs`.
    pub fn new(vs: &nn::VarStore, cfg: &NetConfig) -> Self {
        let p = vs.root();

        // Discrete embeddings (dims match Python exactly)
        let move_type_emb = nn::embedding(&p / "move_type_emb", 2, 8, Default::default());
        let piece_type_emb = nn::embedding(&p / "piece_type_emb", 4, 8, Default::default());
        let direction_emb = nn::embedding(&p / "direction_emb", 5, 8, Default::default());
        let pickup_count_emb = nn::embedding(&p / "pickup_count_emb", 9, 16, Default::default());
        let drop_template_emb =
            nn::embedding(&p / "drop_template_emb", 256, 16, Default::default());
        let travel_length_emb = nn::embedding(&p / "travel_length_emb", 8, 8, Default::default());

        // Policy MLP: input_dim → hidden → 1
        let input_dim = cfg.policy_input_dim();
        let hidden = cfg.policy_hidden();
        let mlp_p = &p / "policy_mlp";
        let policy_mlp = nn::seq()
            .add(nn::linear(&mlp_p / "0", input_dim, hidden, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(&mlp_p / "2", hidden, 1, Default::default()));

        PolicyScorer {
            move_type_emb,
            piece_type_emb,
            direction_emb,
            pickup_count_emb,
            drop_template_emb,
            travel_length_emb,
            policy_mlp,
        }
    }

    /// Score legal moves.
    ///
    /// * `spatial` — `[B, C, 8, 8]` trunk features
    /// * `global_pool` — `[B, C]` global average pool
    /// * `descs` — padded descriptor batch
    /// * `num_moves` — `[B]` actual move count per sample
    ///
    /// Returns `[B, M]` logits with `-inf` at padding positions.
    pub fn score_moves(
        &self,
        spatial: &Tensor,
        global_pool: &Tensor,
        descs: &DescriptorBatch,
        num_moves: &Tensor,
    ) -> Tensor {
        let (b, c, _, _) = spatial.size4().unwrap();
        let m = descs.src.size()[1]; // max_moves

        // Flatten spatial: [B, C, 64] → [B, 64, C]
        let h_flat = spatial.reshape([b, c, 64]).permute([0, 2, 1]);

        // Gather h_src, h_dst: [B, M, C]
        let src_idx = descs.src.to_kind(Kind::Int64).clamp(0, 63);
        let dst_idx = descs.dst.to_kind(Kind::Int64).clamp(0, 63);
        let h_src = h_flat.gather(1, &src_idx.unsqueeze(-1).expand([-1, -1, c], false), false);
        let h_dst = h_flat.gather(1, &dst_idx.unsqueeze(-1).expand([-1, -1, c], false), false);

        // Path pooling: mean of h at path squares, masked
        let path = descs.path.to_kind(Kind::Int64).clamp(0, 63); // [B, M, 7]
        let path_mask = descs.path.ne(255); // [B, M, 7] bool

        let path_flat = path.reshape([b, m * 7]); // [B, M*7]
        let h_path_flat =
            h_flat.gather(1, &path_flat.unsqueeze(-1).expand([-1, -1, c], false), false); // [B, M*7, C]
        let h_path = h_path_flat.reshape([b, m, 7, c]); // [B, M, 7, C]

        let path_mask_f = path_mask
            .unsqueeze(-1)
            .to_kind(Kind::Float); // [B, M, 7, 1]
        let path_sum = (h_path * &path_mask_f).sum_dim_intlist([2].as_slice(), false, Kind::Float); // [B, M, C]
        let path_count = path_mask_f
            .sum_dim_intlist([2].as_slice(), false, Kind::Float)
            .clamp_min(1.0); // [B, M, 1]
        let path_pool = path_sum / path_count; // [B, M, C]

        // Discrete embeddings: each [B, M, dim_i] → concat → [B, M, 64]
        let e_move = self.move_type_emb.forward(&descs.move_type.to_kind(Kind::Int64));
        let e_piece = self.piece_type_emb.forward(&descs.piece_type.to_kind(Kind::Int64));
        let e_dir = self.direction_emb.forward(&descs.direction.to_kind(Kind::Int64));
        let e_pickup = self.pickup_count_emb.forward(&descs.pickup_count.to_kind(Kind::Int64));
        let e_template = self.drop_template_emb.forward(&descs.drop_template_id.to_kind(Kind::Int64));
        let e_travel = self.travel_length_emb.forward(&descs.travel_length.to_kind(Kind::Int64));
        let discrete = Tensor::cat(
            &[&e_move, &e_piece, &e_dir, &e_pickup, &e_template, &e_travel],
            -1,
        ); // [B, M, 64]

        // Float flags: [B, M, 3]
        let flags = Tensor::stack(
            &[
                &descs.capstone_flatten,
                &descs.enters_occupied,
                &descs.opening_phase,
            ],
            -1,
        )
        .to_kind(Kind::Float);

        // Expand global pool: [B, C] → [B, M, C]
        let g_exp = global_pool.unsqueeze(1).expand([-1, m, -1], false);

        // Concatenate: [B, M, 4*C + 64 + 3]
        let mlp_input = Tensor::cat(
            &[&g_exp, &h_src, &h_dst, &path_pool, &discrete, &flags],
            -1,
        );

        // MLP: [B, M, input_dim] → [B, M, 1] → [B, M]
        let logits = self.policy_mlp.forward(&mlp_input).squeeze_dim(-1);

        // Mask padding positions to -inf
        let move_mask = Tensor::arange(m, (Kind::Int64, logits.device()))
            .unsqueeze(0)
            .lt_tensor(&num_moves.unsqueeze(1));
        logits.masked_fill(&move_mask.logical_not(), f64::NEG_INFINITY)
    }
}

// ---------------------------------------------------------------------------
// Descriptor batching utility
// ---------------------------------------------------------------------------

use tak_core::descriptor::MoveDescriptor;

/// Pad and batch descriptors from multiple samples into GPU-ready tensors.
///
/// Returns `(DescriptorBatch, num_moves_tensor)` on the given device.
pub fn batch_descriptors(
    samples: &[Vec<MoveDescriptor>],
    device: tch::Device,
) -> (DescriptorBatch, Tensor) {
    let b = samples.len();
    let max_moves = samples.iter().map(|s| s.len()).max().unwrap_or(0);

    let mut src = vec![0u8; b * max_moves];
    let mut dst = vec![0u8; b * max_moves];
    let mut path = vec![255u8; b * max_moves * 7];
    let mut move_type = vec![0u8; b * max_moves];
    let mut piece_type = vec![0u8; b * max_moves];
    let mut direction = vec![0u8; b * max_moves];
    let mut pickup_count = vec![0u8; b * max_moves];
    let mut drop_template_id = vec![0u8; b * max_moves]; // capped at 255
    let mut travel_length = vec![0u8; b * max_moves];
    let mut capstone_flatten = vec![0.0f32; b * max_moves];
    let mut enters_occupied = vec![0.0f32; b * max_moves];
    let mut opening_phase = vec![0.0f32; b * max_moves];
    let mut num_moves = vec![0i64; b];

    for (i, descs) in samples.iter().enumerate() {
        num_moves[i] = descs.len() as i64;
        for (j, d) in descs.iter().enumerate() {
            let idx = i * max_moves + j;
            src[idx] = d.src;
            dst[idx] = d.dst;
            for (k, &sq) in d.path.iter().enumerate() {
                path[idx * 7 + k] = sq;
            }
            move_type[idx] = d.move_type;
            piece_type[idx] = d.piece_type;
            direction[idx] = d.direction;
            pickup_count[idx] = d.pickup_count;
            drop_template_id[idx] = (d.drop_template_id & 0xFF) as u8;
            travel_length[idx] = d.travel_length;
            capstone_flatten[idx] = if d.capstone_flatten { 1.0 } else { 0.0 };
            enters_occupied[idx] = if d.enters_occupied { 1.0 } else { 0.0 };
            opening_phase[idx] = if d.opening_phase { 1.0 } else { 0.0 };
        }
    }

    let bm = [b as i64, max_moves as i64];
    let bm7 = [b as i64, max_moves as i64, 7];

    let to = |data: &[u8], shape: &[i64]| -> Tensor {
        Tensor::from_slice(data)
            .to_kind(Kind::Int64)
            .reshape(shape)
            .to(device)
    };
    let to_f = |data: &[f32], shape: &[i64]| -> Tensor {
        Tensor::from_slice(data).reshape(shape).to(device)
    };

    let batch = DescriptorBatch {
        src: to(&src, &bm),
        dst: to(&dst, &bm),
        path: to(&path, &bm7),
        move_type: to(&move_type, &bm),
        piece_type: to(&piece_type, &bm),
        direction: to(&direction, &bm),
        pickup_count: to(&pickup_count, &bm),
        drop_template_id: to(&drop_template_id, &bm),
        travel_length: to(&travel_length, &bm),
        capstone_flatten: to_f(&capstone_flatten, &bm),
        enters_occupied: to_f(&enters_occupied, &bm),
        opening_phase: to_f(&opening_phase, &bm),
    };

    let nm = Tensor::from_slice(&num_moves).to(device);
    (batch, nm)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NetConfig;
    use crate::net::TakNet;
    use tak_core::descriptor::MoveDescriptor;
    use tch::{Device, Kind, Tensor};

    fn cpu_policy(cfg: &NetConfig) -> (nn::VarStore, TakNet, PolicyScorer) {
        let vs = nn::VarStore::new(Device::Cpu);
        let net = TakNet::new(&vs, cfg);
        let policy = PolicyScorer::new(&vs, cfg);
        (vs, net, policy)
    }

    #[test]
    fn teacher_score_moves_shape() {
        let cfg = NetConfig::teacher();
        let (_, net, policy) = cpu_policy(&cfg);

        let b = 2i64;
        let board = Tensor::randn([b, 31, 8, 8], (Kind::Float, Device::Cpu));
        let size_id = Tensor::zeros([b], (Kind::Int64, Device::Cpu));
        let out = net.forward_t(&board, &size_id, false);

        // Fake descriptors: 5 moves for sample 0, 3 for sample 1
        let fake_desc = |n: usize| -> Vec<MoveDescriptor> {
            (0..n)
                .map(|_| MoveDescriptor {
                    src: 0,
                    dst: 1,
                    path: Default::default(),
                    move_type: 0,
                    piece_type: 0,
                    direction: 4,
                    pickup_count: 0,
                    drop_template_id: 0,
                    travel_length: 0,
                    capstone_flatten: false,
                    enters_occupied: false,
                    opening_phase: true,
                })
                .collect()
        };

        let samples = vec![fake_desc(5), fake_desc(3)];
        let (descs, num_moves) = batch_descriptors(&samples, Device::Cpu);

        let logits = policy.score_moves(&out.spatial, &out.global_pool, &descs, &num_moves);
        assert_eq!(logits.size(), [2, 5]); // max_moves = 5

        // Padding positions should be -inf
        let l: Vec<f64> = logits.get(1).try_into().unwrap();
        assert!(l[3].is_infinite() && l[3] < 0.0, "padding should be -inf");
        assert!(l[4].is_infinite() && l[4] < 0.0, "padding should be -inf");
        // Valid positions should be finite
        assert!(l[0].is_finite(), "valid move should be finite");
    }

    #[test]
    fn student_score_moves_shape() {
        let cfg = NetConfig::student();
        let (_, net, policy) = cpu_policy(&cfg);

        let b = 1i64;
        let board = Tensor::randn([b, 31, 8, 8], (Kind::Float, Device::Cpu));
        let size_id = Tensor::zeros([b], (Kind::Int64, Device::Cpu));
        let out = net.forward_t(&board, &size_id, false);

        let fake_desc = |n: usize| -> Vec<MoveDescriptor> {
            (0..n)
                .map(|_| MoveDescriptor {
                    src: 10,
                    dst: 18,
                    path: {
                        let mut p = arrayvec::ArrayVec::new();
                        p.push(18);
                        p
                    },
                    move_type: 1,
                    piece_type: 3,
                    direction: 0,
                    pickup_count: 1,
                    drop_template_id: 1,
                    travel_length: 1,
                    capstone_flatten: false,
                    enters_occupied: true,
                    opening_phase: false,
                })
                .collect()
        };

        let samples = vec![fake_desc(10)];
        let (descs, num_moves) = batch_descriptors(&samples, Device::Cpu);

        let logits = policy.score_moves(&out.spatial, &out.global_pool, &descs, &num_moves);
        assert_eq!(logits.size(), [1, 10]);
    }

    #[test]
    fn batch_descriptors_empty_moves() {
        let samples: Vec<Vec<MoveDescriptor>> = vec![vec![], vec![]];
        let (descs, nm) = batch_descriptors(&samples, Device::Cpu);
        // max_moves = 0 → [B, 0]
        assert_eq!(descs.src.size(), [2, 0]);
        let nm_vec: Vec<i64> = nm.try_into().unwrap();
        assert_eq!(nm_vec, vec![0, 0]);
    }
}
