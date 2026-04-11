/*!
Data pipeline: load `.tknn` shards → collate into GPU-ready batches.

## Label strategy

This pipeline trains on **game outcomes** (ground-truth), not on the
search-derived `teacher_wdl` / `teacher_margin` stored in each shard
record.  WDL comes from the game result and margin from the actual flat
count difference.  The shard's search-score fields are preserved in the
schema for external consumers (e.g. the Python training loop) but are
intentionally unused here.

For **soft teacher labels**, use the distillation path (`distill.rs`),
which runs a frozen teacher network at training time instead of reading
pre-computed labels from shards.

## Flow

1. `load_shards()` reads all records from `.tknn` shard files via [`ShardReader`].
2. `ShardLoader` iterates over records in batches, shuffling each epoch.
3. `collate()` converts a slice of [`TrainingRecord`]s into a [`TrainingBatch`]:
   - Encodes boards via [`BoardTensor::encode`]
   - Builds descriptors via [`build_descriptors`]
   - Computes spatial labels via [`SpatialLabels::compute`]
   - Pads and stacks into tensors
*/

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use rayon::prelude::*;
use tch::{Device, Tensor};

use tak_core::descriptor::{build_descriptors, MoveDescriptor};
use tak_core::moves::MoveGen;
use tak_core::state::{GameResult, GameState};
use tak_core::tactical::SpatialLabels;
use tak_core::tensor::{BoardTensor, C_IN};
use tak_data::shard::{ShardReader, TrainingRecord};

use crate::policy::{batch_descriptors, DescriptorBatch};

// ---------------------------------------------------------------------------
// TrainingBatch
// ---------------------------------------------------------------------------

/// A collated batch ready for the GPU.
pub struct TrainingBatch {
    pub board_tensor: Tensor, // [B, 31, 8, 8]
    pub size_id: Tensor,      // [B] i64
    pub wdl: Tensor,          // [B, 3]
    pub margin: Tensor,       // [B, 1]
    pub policy_target: Tensor, // [B, M] dense distribution (0 at padding)
    pub num_moves: Tensor,     // [B] i64
    pub descriptors: DescriptorBatch,
    // Aux labels
    pub road_threat: Tensor,  // [B, 2, 8, 8]
    pub block_threat: Tensor, // [B, 2, 8, 8]
    pub cap_flatten: Tensor,  // [B, 1, 8, 8]
    pub endgame: Tensor,      // [B, 1]
}

// ---------------------------------------------------------------------------
// Per-record encoding (CPU, parallelized with Rayon)
// ---------------------------------------------------------------------------

struct EncodedRecord {
    board_data: [f32; C_IN * 64],
    size_id: i64,
    wdl: [f32; 3],
    margin: f32,
    descriptors: Vec<MoveDescriptor>,
    policy_dense: Vec<f32>, // length = descriptors.len()
    road_threat: [[f32; 64]; 2],
    block_threat: [[f32; 64]; 2],
    cap_flatten: [f32; 64],
    endgame_label: f32,
}

fn encode_record(rec: &TrainingRecord) -> Option<EncodedRecord> {
    let state = rec.unpack_board().ok()?;
    let bt = BoardTensor::encode(&state);

    // WDL: use the game outcome (hard 1/0 labels), not the shard's
    // teacher_wdl (search-score logistic). Ground-truth outcomes give
    // cleaner gradients for the initial teacher; distillation (distill.rs)
    // uses live teacher inference for soft labels instead.
    let wdl = result_to_wdl(&rec.game_result, &state);

    // Margin: use the actual flat-count difference at game end, not the
    // shard's teacher_margin (search-score based). Same rationale as WDL.
    let margin = (rec.flat_margin as f32 / 50.0).clamp(-1.0, 1.0);

    // Descriptors + policy target
    let moves = MoveGen::legal_moves_for(
        &state.board,
        &state.config,
        state.side_to_move,
        state.ply,
        &state.reserves,
        &state.templates,
    );
    let descriptors = build_descriptors(&state, &moves);

    // Build dense policy target over legal moves
    let n_moves = descriptors.len();
    let mut policy_dense = vec![0.0f32; n_moves];
    for &(idx, prob) in &rec.policy_target {
        let i = idx as usize;
        if i < n_moves {
            policy_dense[i] = f32::from(prob);
        }
    }
    // Normalize if sum > 0
    let sum: f32 = policy_dense.iter().sum();
    if sum > 0.0 {
        for p in &mut policy_dense {
            *p /= sum;
        }
    }

    // Spatial labels
    let labels = SpatialLabels::compute(&state);

    Some(EncodedRecord {
        board_data: bt.data,
        size_id: bt.size_id as i64,
        wdl,
        margin,
        descriptors,
        policy_dense,
        road_threat: labels.road_threat,
        block_threat: labels.block_threat,
        cap_flatten: labels.cap_flatten,
        endgame_label: labels.endgame,
    })
}

fn result_to_wdl(result: &GameResult, state: &GameState) -> [f32; 3] {
    let stm = state.side_to_move;
    match result {
        GameResult::RoadWin(c) | GameResult::FlatWin(c) => {
            if *c == stm {
                [1.0, 0.0, 0.0] // win
            } else {
                [0.0, 0.0, 1.0] // loss
            }
        }
        GameResult::Draw => [0.0, 1.0, 0.0],
        GameResult::Ongoing => [0.33, 0.34, 0.33], // fallback
    }
}

// ---------------------------------------------------------------------------
// Collation
// ---------------------------------------------------------------------------

/// Collate a batch of encoded records into GPU tensors.
pub fn collate(records: &[TrainingRecord], device: Device) -> Option<TrainingBatch> {
    if records.is_empty() {
        return None;
    }

    // Encode in parallel
    let encoded: Vec<EncodedRecord> = records
        .par_iter()
        .filter_map(encode_record)
        .collect();

    if encoded.is_empty() {
        return None;
    }

    let b = encoded.len();
    let max_moves = encoded.iter().map(|e| e.descriptors.len()).max().unwrap_or(0);

    // Board tensors: [B, 31, 8, 8]
    let mut board_flat = vec![0.0f32; b * C_IN * 64];
    let mut size_ids = vec![0i64; b];
    let mut wdl_flat = vec![0.0f32; b * 3];
    let mut margin_flat = vec![0.0f32; b];

    // Policy target: [B, max_moves] dense
    let mut policy_flat = vec![0.0f32; b * max_moves];

    // Aux labels
    let mut road_flat = vec![0.0f32; b * 2 * 64];
    let mut block_flat = vec![0.0f32; b * 2 * 64];
    let mut cap_flat = vec![0.0f32; b * 64];
    let mut endgame_flat = vec![0.0f32; b];

    for (i, enc) in encoded.iter().enumerate() {
        board_flat[i * C_IN * 64..(i + 1) * C_IN * 64].copy_from_slice(&enc.board_data);
        size_ids[i] = enc.size_id;
        wdl_flat[i * 3..(i + 1) * 3].copy_from_slice(&enc.wdl);
        margin_flat[i] = enc.margin;

        // Copy policy into padded buffer
        for (j, &p) in enc.policy_dense.iter().enumerate() {
            policy_flat[i * max_moves + j] = p;
        }

        // Aux labels
        road_flat[i * 128..i * 128 + 64].copy_from_slice(&enc.road_threat[0]);
        road_flat[i * 128 + 64..(i + 1) * 128].copy_from_slice(&enc.road_threat[1]);
        block_flat[i * 128..i * 128 + 64].copy_from_slice(&enc.block_threat[0]);
        block_flat[i * 128 + 64..(i + 1) * 128].copy_from_slice(&enc.block_threat[1]);
        cap_flat[i * 64..(i + 1) * 64].copy_from_slice(&enc.cap_flatten);
        endgame_flat[i] = enc.endgame_label;
    }

    // Descriptor batch
    let desc_samples: Vec<Vec<MoveDescriptor>> =
        encoded.iter().map(|e| e.descriptors.clone()).collect();
    let (descriptors, num_moves) = batch_descriptors(&desc_samples, device);

    let board_tensor = Tensor::from_slice(&board_flat)
        .reshape([b as i64, C_IN as i64, 8, 8])
        .to(device);
    let size_id = Tensor::from_slice(&size_ids).to(device);
    let wdl = Tensor::from_slice(&wdl_flat)
        .reshape([b as i64, 3])
        .to(device);
    let margin = Tensor::from_slice(&margin_flat)
        .reshape([b as i64, 1])
        .to(device);
    let policy_target = Tensor::from_slice(&policy_flat)
        .reshape([b as i64, max_moves as i64])
        .to(device);
    let road_threat = Tensor::from_slice(&road_flat)
        .reshape([b as i64, 2, 8, 8])
        .to(device);
    let block_threat = Tensor::from_slice(&block_flat)
        .reshape([b as i64, 2, 8, 8])
        .to(device);
    let cap_flatten = Tensor::from_slice(&cap_flat)
        .reshape([b as i64, 1, 8, 8])
        .to(device);
    let endgame = Tensor::from_slice(&endgame_flat)
        .reshape([b as i64, 1])
        .to(device);

    Some(TrainingBatch {
        board_tensor,
        size_id,
        wdl,
        margin,
        policy_target,
        num_moves,
        descriptors,
        road_threat,
        block_threat,
        cap_flatten,
        endgame,
    })
}

// ---------------------------------------------------------------------------
// ShardLoader
// ---------------------------------------------------------------------------

/// Load all records from a list of `.tknn` shard files.
pub fn load_shards(paths: &[PathBuf]) -> Vec<TrainingRecord> {
    let mut records = Vec::new();
    for path in paths {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Warning: cannot open {}: {e}", path.display());
                continue;
            }
        };
        let reader = BufReader::new(file);
        let mut shard = match ShardReader::new(reader) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Warning: invalid shard {}: {e}", path.display());
                continue;
            }
        };
        while let Ok(Some(rec)) = shard.next_record() {
            records.push(rec);
        }
    }
    records
}

/// Iterable batch loader that shuffles each epoch.
pub struct ShardLoader {
    records: Vec<TrainingRecord>,
    batch_size: usize,
    offset: usize,
}

impl ShardLoader {
    pub fn new(records: Vec<TrainingRecord>, batch_size: usize) -> Self {
        ShardLoader {
            records,
            batch_size,
            offset: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn num_batches(&self) -> usize {
        self.records.len().div_ceil(self.batch_size)
    }

    /// Shuffle records for a new epoch.
    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::rngs::ThreadRng::default();
        self.records.shuffle(&mut rng);
        self.offset = 0;
    }

    /// Reset iterator to the start without shuffling.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Get the next batch, or `None` if the epoch is done.
    pub fn next_batch(&mut self, device: Device) -> Option<TrainingBatch> {
        if self.offset >= self.records.len() {
            return None;
        }
        let end = (self.offset + self.batch_size).min(self.records.len());
        let batch_records = &self.records[self.offset..end];
        self.offset = end;
        collate(batch_records, device)
    }
}

// ---------------------------------------------------------------------------
// Train/val split
// ---------------------------------------------------------------------------

/// Split records into (train, val) with the given validation fraction.
pub fn train_val_split(
    mut records: Vec<TrainingRecord>,
    val_fraction: f64,
) -> (Vec<TrainingRecord>, Vec<TrainingRecord>) {
    use rand::seq::SliceRandom;
    let mut rng = rand::rng();
    records.shuffle(&mut rng);

    let val_size = (records.len() as f64 * val_fraction) as usize;
    let val = records.split_off(records.len() - val_size);
    (records, val)
}
