//! WASM bindings for the Tak engine, exposing a JS-friendly API.

use js_sys::Float32Array;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

use tak_core::board::Square;
use tak_core::descriptor::{self, MoveDescriptor};
use tak_core::moves::Move;
use tak_core::piece::PieceType;
use tak_core::ptn;
use tak_core::rules::GameConfig;
use tak_core::state::{GameResult, GameState};
use tak_core::tensor::BoardTensor;
use tak_core::tps;

/// Install panic hook so WASM panics produce readable console errors.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

// ---------------------------------------------------------------------------
// Neural FFI bindings
// ---------------------------------------------------------------------------

struct Blob {
    shape: Vec<usize>,
    data: Vec<f32>,
}

struct Embedding {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Embedding {
    fn from_blob(name: &str, blob: Blob) -> Result<Self, String> {
        if blob.shape.len() != 2 {
            return Err(format!("{name} must be rank-2, got {:?}", blob.shape));
        }

        let rows = blob.shape[0];
        let cols = blob.shape[1];
        if rows * cols != blob.data.len() {
            return Err(format!(
                "{name} shape {:?} does not match data length {}",
                blob.shape,
                blob.data.len()
            ));
        }

        Ok(Self {
            rows,
            cols,
            data: blob.data,
        })
    }

    fn append(&self, index: usize, out: &mut Vec<f32>) -> Result<(), String> {
        if index >= self.rows {
            return Err(format!(
                "embedding index {} out of range for {} rows",
                index, self.rows
            ));
        }

        let start = index * self.cols;
        out.extend_from_slice(&self.data[start..start + self.cols]);
        Ok(())
    }
}

struct LinearLayer {
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearLayer {
    fn from_blobs(name: &str, weight: Blob, bias: Blob) -> Result<Self, String> {
        if weight.shape.len() != 2 {
            return Err(format!(
                "{name} weight must be rank-2, got {:?}",
                weight.shape
            ));
        }
        if bias.shape.len() != 1 {
            return Err(format!("{name} bias must be rank-1, got {:?}", bias.shape));
        }

        let out_dim = weight.shape[0];
        let in_dim = weight.shape[1];
        if bias.shape[0] != out_dim {
            return Err(format!(
                "{name} bias shape {:?} does not match weight output dim {}",
                bias.shape, out_dim
            ));
        }
        if weight.data.len() != out_dim * in_dim {
            return Err(format!(
                "{name} weight shape {:?} does not match data length {}",
                weight.shape,
                weight.data.len()
            ));
        }

        Ok(Self {
            in_dim,
            out_dim,
            weight: weight.data,
            bias: bias.data,
        })
    }

    fn apply(&self, input: &[f32], out: &mut [f32]) -> Result<(), String> {
        if input.len() != self.in_dim {
            return Err(format!(
                "linear input length {} does not match expected {}",
                input.len(),
                self.in_dim
            ));
        }
        if out.len() != self.out_dim {
            return Err(format!(
                "linear output length {} does not match expected {}",
                out.len(),
                self.out_dim
            ));
        }

        for (row_idx, dst) in out.iter_mut().enumerate() {
            let row = &self.weight[row_idx * self.in_dim..(row_idx + 1) * self.in_dim];
            let mut acc = self.bias[row_idx];
            for (&w, &x) in row.iter().zip(input.iter()) {
                acc += w * x;
            }
            *dst = acc;
        }

        Ok(())
    }

    fn apply_scalar(&self, input: &[f32]) -> Result<f32, String> {
        if self.out_dim != 1 {
            return Err(format!(
                "apply_scalar requires out_dim=1, got {}",
                self.out_dim
            ));
        }

        let mut out = [0.0f32; 1];
        self.apply(input, &mut out)?;
        Ok(out[0])
    }
}

#[wasm_bindgen]
pub struct NeuralPolicy {
    channels: usize,
    move_type_emb: Embedding,
    piece_type_emb: Embedding,
    direction_emb: Embedding,
    pickup_count_emb: Embedding,
    drop_template_emb: Embedding,
    travel_length_emb: Embedding,
    linear1: LinearLayer,
    linear2: LinearLayer,
}

#[wasm_bindgen]
impl NeuralPolicy {
    #[wasm_bindgen(constructor)]
    pub fn new(buffer: &[u8]) -> Result<NeuralPolicy, JsError> {
        Self::from_tpol(buffer).map_err(|err| JsError::new(&err))
    }

    /// Evaluates the policy MLP for the given move descriptors using the spatial and global pools.
    /// `spatial_pool`: [64, 8, 8] (or similar) from the ONNX trunk.
    /// `global_pool`: [64] from the ONNX trunk.
    #[wasm_bindgen(js_name = "scoreMoves")]
    pub fn score_moves(
        &self,
        game: &TakGame,
        spatial_pool: &[f32],
        global_pool: &[f32],
    ) -> Result<Float32Array, JsError> {
        let probs = self
            .score_probabilities(&game.state, spatial_pool, global_pool)
            .map_err(|err| JsError::new(&err))?;
        Ok(float32_array_from_slice(&probs))
    }
}

impl NeuralPolicy {
    fn from_tpol(buffer: &[u8]) -> Result<Self, String> {
        let mut blobs = parse_tpol(buffer)?;

        let move_type_emb = Embedding::from_blob(
            "move_type_emb.weight",
            take_blob(&mut blobs, "move_type_emb.weight")?,
        )?;
        let piece_type_emb = Embedding::from_blob(
            "piece_type_emb.weight",
            take_blob(&mut blobs, "piece_type_emb.weight")?,
        )?;
        let direction_emb = Embedding::from_blob(
            "direction_emb.weight",
            take_blob(&mut blobs, "direction_emb.weight")?,
        )?;
        let pickup_count_emb = Embedding::from_blob(
            "pickup_count_emb.weight",
            take_blob(&mut blobs, "pickup_count_emb.weight")?,
        )?;
        let drop_template_emb = Embedding::from_blob(
            "drop_template_emb.weight",
            take_blob(&mut blobs, "drop_template_emb.weight")?,
        )?;
        let travel_length_emb = Embedding::from_blob(
            "travel_length_emb.weight",
            take_blob(&mut blobs, "travel_length_emb.weight")?,
        )?;

        let linear1 = LinearLayer::from_blobs(
            "policy_mlp.0",
            take_blob(&mut blobs, "policy_mlp.0.weight")?,
            take_blob(&mut blobs, "policy_mlp.0.bias")?,
        )?;
        let linear2 = LinearLayer::from_blobs(
            "policy_mlp.2",
            take_blob(&mut blobs, "policy_mlp.2.weight")?,
            take_blob(&mut blobs, "policy_mlp.2.bias")?,
        )?;

        let discrete_dim = move_type_emb.cols
            + piece_type_emb.cols
            + direction_emb.cols
            + pickup_count_emb.cols
            + drop_template_emb.cols
            + travel_length_emb.cols;
        if linear1.in_dim < discrete_dim + 3 {
            return Err(format!(
                "policy input dim {} is too small for discrete dim {}",
                linear1.in_dim, discrete_dim
            ));
        }

        let trunk_dim = linear1.in_dim - discrete_dim - 3;
        if trunk_dim % 4 != 0 {
            return Err(format!(
                "policy input dim {} does not match 4*C + discrete + 3 layout",
                linear1.in_dim
            ));
        }

        let channels = trunk_dim / 4;
        if channels == 0 {
            return Err("policy scorer inferred zero channels".into());
        }
        if linear2.in_dim != linear1.out_dim || linear2.out_dim != 1 {
            return Err(format!(
                "policy MLP shapes are inconsistent: hidden={} second=({}, {})",
                linear1.out_dim, linear2.out_dim, linear2.in_dim
            ));
        }

        Ok(Self {
            channels,
            move_type_emb,
            piece_type_emb,
            direction_emb,
            pickup_count_emb,
            drop_template_emb,
            travel_length_emb,
            linear1,
            linear2,
        })
    }

    fn score_probabilities(
        &self,
        state: &GameState,
        spatial_pool: &[f32],
        global_pool: &[f32],
    ) -> Result<Vec<f32>, String> {
        if spatial_pool.len() != self.channels * 64 {
            return Err(format!(
                "spatial pool length {} does not match expected {}",
                spatial_pool.len(),
                self.channels * 64
            ));
        }
        if global_pool.len() != self.channels {
            return Err(format!(
                "global pool length {} does not match expected {}",
                global_pool.len(),
                self.channels
            ));
        }

        let moves = state.legal_moves();
        if moves.is_empty() {
            return Ok(Vec::new());
        }

        let descriptors = descriptor::build_descriptors(state, &moves);
        let mut logits = Vec::with_capacity(descriptors.len());
        let mut input = Vec::with_capacity(self.linear1.in_dim);
        let mut hidden = vec![0.0f32; self.linear1.out_dim];

        for desc in &descriptors {
            self.encode_move_input(desc, spatial_pool, global_pool, &mut input)?;
            self.linear1.apply(&input, &mut hidden)?;
            for value in &mut hidden {
                *value = value.max(0.0);
            }
            logits.push(self.linear2.apply_scalar(&hidden)?);
        }

        Ok(softmax(&logits))
    }

    fn encode_move_input(
        &self,
        desc: &MoveDescriptor,
        spatial_pool: &[f32],
        global_pool: &[f32],
        out: &mut Vec<f32>,
    ) -> Result<(), String> {
        out.clear();
        out.extend_from_slice(global_pool);
        append_square_features(spatial_pool, self.channels, desc.src, out);
        append_square_features(spatial_pool, self.channels, desc.dst, out);
        append_path_pool(spatial_pool, self.channels, desc, out);

        self.move_type_emb
            .append(usize::from(desc.move_type), out)?;
        self.piece_type_emb
            .append(usize::from(desc.piece_type), out)?;
        self.direction_emb
            .append(usize::from(desc.direction), out)?;
        self.pickup_count_emb
            .append(usize::from(desc.pickup_count), out)?;
        self.drop_template_emb
            .append(usize::from(desc.drop_template_id & 0x00FF), out)?;
        self.travel_length_emb
            .append(usize::from(desc.travel_length), out)?;

        out.push(bool_to_f32(desc.capstone_flatten));
        out.push(bool_to_f32(desc.enters_occupied));
        out.push(bool_to_f32(desc.opening_phase));

        if out.len() != self.linear1.in_dim {
            return Err(format!(
                "constructed policy input length {} does not match expected {}",
                out.len(),
                self.linear1.in_dim
            ));
        }

        Ok(())
    }
}

fn parse_tpol(buffer: &[u8]) -> Result<HashMap<String, Blob>, String> {
    if buffer.len() < 12 {
        return Err("TPOL buffer too small for header".into());
    }
    if &buffer[0..4] != b"TPOL" {
        return Err("invalid TPOL magic header".into());
    }

    let version = u32::from_le_bytes(buffer[4..8].try_into().unwrap());
    if version != 1 {
        return Err(format!("unsupported TPOL version: {}", version));
    }

    let num_blobs = u32::from_le_bytes(buffer[8..12].try_into().unwrap()) as usize;
    let mut offset = 12;
    let mut blobs = HashMap::new();

    for _ in 0..num_blobs {
        if offset + 2 > buffer.len() {
            return Err("TPOL blob truncated (name_len)".into());
        }
        let name_len = u16::from_le_bytes(buffer[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;

        if offset + name_len > buffer.len() {
            return Err("TPOL blob truncated (name)".into());
        }
        let name = std::str::from_utf8(&buffer[offset..offset + name_len])
            .map_err(|_| "TPOL invalid UTF-8 in blob name".to_string())?
            .to_string();
        offset += name_len;

        if offset + 1 > buffer.len() {
            return Err("TPOL blob truncated (ndims)".into());
        }
        let ndims = buffer[offset] as usize;
        offset += 1;

        let mut shape = Vec::with_capacity(ndims);
        let mut numel = 1usize;
        for _ in 0..ndims {
            if offset + 4 > buffer.len() {
                return Err("TPOL blob truncated (dims)".into());
            }
            let dim = u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
            shape.push(dim);
            numel = numel
                .checked_mul(dim)
                .ok_or_else(|| format!("TPOL blob {} has overflowing shape {:?}", name, shape))?;
            offset += 4;
        }

        if offset + numel * 4 > buffer.len() {
            return Err("TPOL blob truncated (data)".into());
        }
        let mut data = Vec::with_capacity(numel);
        for _ in 0..numel {
            let value = f32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap());
            data.push(value);
            offset += 4;
        }

        blobs.insert(name, Blob { shape, data });
    }

    if offset != buffer.len() {
        return Err(format!(
            "TPOL buffer has {} trailing byte(s)",
            buffer.len() - offset
        ));
    }

    Ok(blobs)
}

fn take_blob(blobs: &mut HashMap<String, Blob>, suffix: &str) -> Result<Blob, String> {
    let candidates: Vec<String> = blobs
        .keys()
        .filter(|name| *name == suffix || name.ends_with(suffix))
        .cloned()
        .collect();

    match candidates.len() {
        0 => Err(format!("missing TPOL blob {}", suffix)),
        1 => Ok(blobs.remove(&candidates[0]).unwrap()),
        _ => Err(format!(
            "multiple TPOL blobs matched {}: {:?}",
            suffix, candidates
        )),
    }
}

fn append_square_features(spatial_pool: &[f32], channels: usize, square: u8, out: &mut Vec<f32>) {
    let square = usize::from(square);
    for channel in 0..channels {
        out.push(spatial_pool[channel * 64 + square]);
    }
}

fn append_path_pool(
    spatial_pool: &[f32],
    channels: usize,
    desc: &MoveDescriptor,
    out: &mut Vec<f32>,
) {
    if desc.path.is_empty() {
        out.resize(out.len() + channels, 0.0);
        return;
    }

    for channel in 0..channels {
        let mut sum = 0.0;
        for &square in &desc.path {
            sum += spatial_pool[channel * 64 + usize::from(square)];
        }
        out.push(sum / desc.path.len() as f32);
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0;

    for &logit in logits {
        let value = (logit - max_logit).exp();
        exps.push(value);
        sum += value;
    }

    if sum == 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }

    exps.into_iter().map(|value| value / sum).collect()
}

fn float32_array_from_slice(data: &[f32]) -> Float32Array {
    let array = Float32Array::new_with_length(data.len() as u32);
    array.copy_from(data);
    array
}

fn bool_to_f32(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// JS-facing game wrapper
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct TakGame {
    state: GameState,
    move_history: Vec<Move>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveInfo {
    pub index: usize,
    pub ptn: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameInfo {
    pub size: u8,
    pub ply: u16,
    pub side_to_move: String,
    pub result: String,
    pub tps: String,
    pub white_stones: u8,
    pub white_caps: u8,
    pub black_stones: u8,
    pub black_caps: u8,
}

#[wasm_bindgen]
impl TakGame {
    /// Create a new game with the given board size (3-8).
    #[wasm_bindgen(constructor)]
    pub fn new(size: u8) -> Result<TakGame, JsError> {
        if !(3..=8).contains(&size) {
            return Err(JsError::new(&format!("unsupported board size: {}", size)));
        }
        Ok(TakGame {
            state: GameState::new(GameConfig::standard(size)),
            move_history: Vec::new(),
        })
    }

    /// Create a game from a TPS string.
    #[wasm_bindgen(js_name = "fromTps")]
    pub fn from_tps(tps_str: &str) -> Result<TakGame, JsError> {
        let state = tps::from_tps(tps_str).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(TakGame {
            state,
            move_history: Vec::new(),
        })
    }

    /// Get game info as a JSON object.
    #[wasm_bindgen(js_name = "getInfo")]
    pub fn get_info(&self) -> Result<JsValue, JsError> {
        let info = GameInfo {
            size: self.state.config.size,
            ply: self.state.ply,
            side_to_move: match self.state.side_to_move {
                tak_core::piece::Color::White => "white".into(),
                tak_core::piece::Color::Black => "black".into(),
            },
            result: format_result(self.state.result),
            tps: tps::to_tps(&self.state),
            white_stones: self.state.reserves[0],
            white_caps: self.state.reserves[1],
            black_stones: self.state.reserves[2],
            black_caps: self.state.reserves[3],
        };
        serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get list of legal moves as JSON array of {index, ptn}.
    #[wasm_bindgen(js_name = "legalMoves")]
    pub fn legal_moves(&self) -> Result<JsValue, JsError> {
        let moves = self.state.legal_moves();
        let infos: Vec<MoveInfo> = moves
            .iter()
            .enumerate()
            .map(|(i, &mv)| MoveInfo {
                index: i,
                ptn: ptn::format_move(mv, &self.state),
            })
            .collect();
        serde_wasm_bindgen::to_value(&infos).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Apply a move by its index in the legal moves list.
    #[wasm_bindgen(js_name = "applyMoveIndex")]
    pub fn apply_move_index(&mut self, index: usize) -> Result<(), JsError> {
        let moves = self.state.legal_moves();
        let mv = moves
            .get(index)
            .ok_or_else(|| JsError::new(&format!("move index {} out of range", index)))?;
        let mv = *mv;
        self.state.apply_move(mv);
        self.move_history.push(mv);
        Ok(())
    }

    /// Apply a move by PTN string.
    #[wasm_bindgen(js_name = "applyMovePtn")]
    pub fn apply_move_ptn(&mut self, ptn_str: &str) -> Result<(), JsError> {
        let mv = ptn::parse_move(ptn_str, &self.state).map_err(|e| JsError::new(&e.to_string()))?;
        // Verify the move is legal.
        let legal = self.state.legal_moves();
        if !legal.contains(&mv) {
            return Err(JsError::new(&format!("illegal move: {}", ptn_str)));
        }
        self.state.apply_move(mv);
        self.move_history.push(mv);
        Ok(())
    }

    /// Undo the last move. Returns false if no moves to undo.
    pub fn undo(&mut self) -> bool {
        if self.move_history.is_empty() {
            return false;
        }
        // Rebuild from scratch (simpler than maintaining undo stack through WASM boundary).
        let size = self.state.config.size;
        let moves = self.move_history.clone();
        self.state = GameState::new(GameConfig::standard(size));
        self.move_history.clear();
        for &mv in &moves[..moves.len() - 1] {
            self.state.apply_move(mv);
            self.move_history.push(mv);
        }
        true
    }

    /// Get the board state as a flat array for rendering.
    /// Returns a JSON array of 64 square objects (8x8 grid, row-major).
    /// Each square: { pieces: [{color, type}], active: bool }
    #[wasm_bindgen(js_name = "getBoard")]
    pub fn get_board(&self) -> Result<JsValue, JsError> {
        let size = self.state.config.size;
        let mut squares: Vec<SquareInfo> = Vec::with_capacity(64);

        for r in 0..8u8 {
            for c in 0..8u8 {
                let active = r < size && c < size;
                let sq = Square::from_rc(r, c);
                let stack = self.state.board.get(sq);
                let mut pieces = Vec::new();

                if active && !stack.is_empty() {
                    // Build pieces bottom to top.
                    // Buried pieces (order unknown).
                    for _ in 0..stack.buried_white {
                        pieces.push(PieceInfo {
                            color: "white".into(),
                            piece_type: "flat".into(),
                        });
                    }
                    for _ in 0..stack.buried_black {
                        pieces.push(PieceInfo {
                            color: "black".into(),
                            piece_type: "flat".into(),
                        });
                    }
                    // Explicit interior layers (bottom to top = reversed below array).
                    for &color in stack.below.iter().rev() {
                        pieces.push(PieceInfo {
                            color: match color {
                                tak_core::piece::Color::White => "white".into(),
                                tak_core::piece::Color::Black => "black".into(),
                            },
                            piece_type: "flat".into(),
                        });
                    }
                    // Top piece.
                    if let Some(top) = stack.top {
                        pieces.push(PieceInfo {
                            color: match top.color() {
                                tak_core::piece::Color::White => "white".into(),
                                tak_core::piece::Color::Black => "black".into(),
                            },
                            piece_type: match top.piece_type() {
                                PieceType::Flat => "flat".into(),
                                PieceType::Wall => "wall".into(),
                                PieceType::Cap => "cap".into(),
                            },
                        });
                    }
                }

                squares.push(SquareInfo { pieces, active });
            }
        }

        serde_wasm_bindgen::to_value(&squares).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get TPS string for the current position.
    #[wasm_bindgen(js_name = "getTps")]
    pub fn get_tps(&self) -> String {
        tps::to_tps(&self.state)
    }

    /// Get the move history as PTN.
    #[wasm_bindgen(js_name = "getMoveHistory")]
    pub fn get_move_history(&self) -> String {
        ptn::format_game(&self.state.config, &self.move_history)
    }

    /// Check if the game is over.
    #[wasm_bindgen(js_name = "isGameOver")]
    pub fn is_game_over(&self) -> bool {
        self.state.result.is_terminal()
    }

    /// Get the board size.
    pub fn size(&self) -> u8 {
        self.state.config.size
    }

    /// Get current ply.
    pub fn ply(&self) -> u16 {
        self.state.ply
    }

    /// Encode the current position as the NN board tensor.
    #[wasm_bindgen(js_name = "encodeBoard")]
    pub fn encode_board(&self) -> Float32Array {
        let tensor = BoardTensor::encode(&self.state);
        float32_array_from_slice(&tensor.data)
    }

    /// Get the size-id expected by the NN trunk input (3x3 -> 0, ..., 8x8 -> 5).
    #[wasm_bindgen(js_name = "sizeId")]
    pub fn size_id(&self) -> u8 {
        self.state.config.size - 3
    }

    /// Run heuristic search and return the best move + info.
    /// `max_depth` and `time_ms` control search limits.
    #[wasm_bindgen(js_name = "searchMove")]
    pub fn search_move(&mut self, max_depth: u8, time_ms: u32) -> Result<JsValue, JsError> {
        use tak_search::eval::HeuristicEval;
        use tak_search::pvs::{PvsSearch, SearchConfig};

        if self.state.result.is_terminal() {
            return Err(JsError::new("game is already over"));
        }

        let config = SearchConfig {
            max_depth,
            max_time_ms: time_ms as u64,
            tt_size_mb: 4,
        };
        let mut search = PvsSearch::new(config, HeuristicEval);
        let result = search.search(&mut self.state);

        let best_ptn = result.best_move.map(|mv| ptn::format_move(mv, &self.state));
        let pv_ptn: Vec<String> = result
            .pv
            .iter()
            .map(|mv| ptn::format_move(*mv, &self.state))
            .collect();

        let info = SearchResultInfo {
            best_move: best_ptn.unwrap_or_default(),
            score: result.score,
            depth: result.depth,
            nodes: result.nodes as u32,
            pv: pv_ptn,
            tt_hits: result.tt_hits as u32,
        };

        serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
    }

    /// Run search and apply the best move. Returns search info.
    #[wasm_bindgen(js_name = "botMove")]
    pub fn bot_move(&mut self, max_depth: u8, time_ms: u32) -> Result<JsValue, JsError> {
        use tak_search::eval::HeuristicEval;
        use tak_search::pvs::{PvsSearch, SearchConfig};

        if self.state.result.is_terminal() {
            return Err(JsError::new("game is already over"));
        }

        // Cap depth for WASM to limit stack usage.
        let effective_depth = max_depth.min(12);
        let config = SearchConfig {
            max_depth: effective_depth,
            max_time_ms: time_ms as u64,
            tt_size_mb: 4,
        };
        let mut search = PvsSearch::new(config, HeuristicEval);
        let result = search.search(&mut self.state);

        let mv = result
            .best_move
            .ok_or_else(|| JsError::new("no move found"))?;

        let best_ptn = ptn::format_move(mv, &self.state);
        let pv_ptn: Vec<String> = result
            .pv
            .iter()
            .map(|m| ptn::format_move(*m, &self.state))
            .collect();

        // Apply the move.
        self.state.apply_move(mv);
        self.move_history.push(mv);

        let info = SearchResultInfo {
            best_move: best_ptn,
            score: result.score,
            depth: result.depth,
            nodes: result.nodes as u32,
            pv: pv_ptn,
            tt_hits: result.tt_hits as u32,
        };

        serde_wasm_bindgen::to_value(&info).map_err(|e| JsError::new(&e.to_string()))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchResultInfo {
    best_move: String,
    score: i32,
    depth: u8,
    nodes: u32,
    pv: Vec<String>,
    tt_hits: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PieceInfo {
    color: String,
    piece_type: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SquareInfo {
    pieces: Vec<PieceInfo>,
    active: bool,
}

fn format_result(result: GameResult) -> String {
    match result {
        GameResult::Ongoing => "ongoing".into(),
        GameResult::RoadWin(c) => format!("road_win_{}", color_str(c)),
        GameResult::FlatWin(c) => format!("flat_win_{}", color_str(c)),
        GameResult::Draw => "draw".into(),
    }
}

fn color_str(c: tak_core::piece::Color) -> &'static str {
    match c {
        tak_core::piece::Color::White => "white",
        tak_core::piece::Color::Black => "black",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_blob(buffer: &mut Vec<u8>, name: &str, shape: &[usize], data: &[f32]) {
        buffer.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buffer.extend_from_slice(name.as_bytes());
        buffer.push(shape.len() as u8);
        for &dim in shape {
            buffer.extend_from_slice(&(dim as u32).to_le_bytes());
        }
        for &value in data {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn zero_policy_buffer(channels: usize, hidden: usize) -> Vec<u8> {
        let input_dim = channels * 4 + 64 + 3;
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"TPOL");
        buffer.extend_from_slice(&1u32.to_le_bytes());
        buffer.extend_from_slice(&10u32.to_le_bytes());

        push_blob(
            &mut buffer,
            "move_type_emb.weight",
            &[2, 8],
            &vec![0.0; 2 * 8],
        );
        push_blob(
            &mut buffer,
            "piece_type_emb.weight",
            &[4, 8],
            &vec![0.0; 4 * 8],
        );
        push_blob(
            &mut buffer,
            "direction_emb.weight",
            &[5, 8],
            &vec![0.0; 5 * 8],
        );
        push_blob(
            &mut buffer,
            "pickup_count_emb.weight",
            &[9, 16],
            &vec![0.0; 9 * 16],
        );
        push_blob(
            &mut buffer,
            "drop_template_emb.weight",
            &[256, 16],
            &vec![0.0; 256 * 16],
        );
        push_blob(
            &mut buffer,
            "travel_length_emb.weight",
            &[8, 8],
            &vec![0.0; 8 * 8],
        );
        push_blob(
            &mut buffer,
            "policy_mlp.0.weight",
            &[hidden, input_dim],
            &vec![0.0; hidden * input_dim],
        );
        push_blob(
            &mut buffer,
            "policy_mlp.0.bias",
            &[hidden],
            &vec![0.0; hidden],
        );
        push_blob(
            &mut buffer,
            "policy_mlp.2.weight",
            &[1, hidden],
            &vec![0.0; hidden],
        );
        push_blob(&mut buffer, "policy_mlp.2.bias", &[1], &[0.0]);

        buffer
    }

    #[test]
    fn parses_tpol_and_inferrs_channel_count() {
        let policy = NeuralPolicy::from_tpol(&zero_policy_buffer(2, 3)).unwrap();

        assert_eq!(policy.channels, 2);
        assert_eq!(policy.linear1.in_dim, 75);
        assert_eq!(policy.linear1.out_dim, 3);
    }

    #[test]
    fn zero_weights_produce_uniform_policy() {
        let policy = NeuralPolicy::from_tpol(&zero_policy_buffer(1, 2)).unwrap();
        let state = GameState::new(GameConfig::standard(3));

        let probs = policy
            .score_probabilities(&state, &vec![0.0; 64], &[0.0])
            .unwrap();

        assert_eq!(probs.len(), 9);
        let expected = 1.0 / probs.len() as f32;
        for prob in &probs {
            assert!((prob - expected).abs() < 1e-6);
        }
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_invalid_feature_lengths() {
        let policy = NeuralPolicy::from_tpol(&zero_policy_buffer(2, 3)).unwrap();
        let state = GameState::new(GameConfig::standard(3));

        let err = policy
            .score_probabilities(&state, &vec![0.0; 64], &[0.0, 0.0])
            .unwrap_err();

        assert!(err.contains("spatial pool length"));
    }
}
