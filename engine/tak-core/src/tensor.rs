//! Board-to-tensor encoding for neural network input.
//!
//! Encodes a GameState into a fixed-shape tensor suitable for the residual
//! trunk. Shape: [C_IN, 8, 8] in CHW order, row-major.

use crate::piece::Color;
use crate::state::GameState;

/// Total input channels.
pub const C_IN: usize = 31;

/// Encoded board tensor.
pub struct BoardTensor {
    /// Shape: [C_IN, 8, 8] in CHW order, row-major. data[c * 64 + r * 8 + col].
    pub data: [f32; C_IN * 64],
    /// Board size index: 0..5 for sizes 3..8.
    pub size_id: u8,
}

impl BoardTensor {
    /// Encode a GameState into the NN input tensor.
    pub fn encode(state: &GameState) -> Self {
        let mut data = [0.0f32; C_IN * 64];
        let size = state.config.size;
        let size_id = size - 3;

        // --- Spatial channels (only active NxN region) ---
        for r in 0..size {
            for c in 0..size {
                let sq = crate::board::Square::from_rc(r, c);
                let si = (r as usize) * 8 + (c as usize); // spatial index in 8x8 grid
                let stack = state.board.get(sq);

                // Channels 0..6: top piece one-hot + is_occupied
                if let Some(top) = stack.top {
                    let piece_idx = top as u8 as usize; // 0..5
                    data[piece_idx * 64 + si] = 1.0;
                    data[6 * 64 + si] = 1.0; // is_occupied
                }

                // Channels 7..20: interior layers (7 layers × 2 channels each)
                for (layer_idx, &color) in stack.below.iter().enumerate() {
                    let base_ch = 7 + layer_idx * 2;
                    let color_offset = color as u8 as usize; // 0=white, 1=black
                    data[(base_ch + color_offset) * 64 + si] = 1.0;
                }

                // Channels 21..22: buried counts (normalized)
                data[21 * 64 + si] = stack.buried_white as f32 / 50.0;
                data[22 * 64 + si] = stack.buried_black as f32 / 50.0;
            }
        }

        // --- Global feature planes (broadcast to all 64 squares) ---
        let max_stones = state.config.stones as f32;
        let max_caps = if state.config.capstones > 0 {
            state.config.capstones as f32
        } else {
            1.0 // avoid div by zero; value will be 0.0 anyway
        };

        let globals: [(usize, f32); 8] = [
            (
                23,
                if state.side_to_move == Color::White {
                    1.0
                } else {
                    0.0
                },
            ),
            (24, if state.ply < 2 { 1.0 } else { 0.0 }),
            (25, state.reserves[0] as f32 / max_stones),
            (26, state.reserves[2] as f32 / max_stones),
            (
                27,
                if state.config.capstones > 0 {
                    state.reserves[1] as f32 / max_caps
                } else {
                    0.0
                },
            ),
            (
                28,
                if state.config.capstones > 0 {
                    state.reserves[3] as f32 / max_caps
                } else {
                    0.0
                },
            ),
            (
                29,
                (state.config.komi as f32 + if state.config.half_komi { 0.5 } else { 0.0 }) / 4.0,
            ),
            (30, state.ply as f32 / 200.0),
        ];

        for (ch, val) in globals {
            let base = ch * 64;
            for i in 0..64 {
                data[base + i] = val;
            }
        }

        BoardTensor { data, size_id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Square;
    use crate::moves::Move;
    use crate::piece::PieceType;
    use crate::rules::GameConfig;

    #[test]
    fn empty_board_globals() {
        let state = GameState::new(GameConfig::standard(6));
        let t = BoardTensor::encode(&state);
        assert_eq!(t.size_id, 3); // 6 - 3

        // Side to move = white = 1.0
        assert_eq!(t.data[23 * 64], 1.0);
        // Opening phase = true = 1.0
        assert_eq!(t.data[24 * 64], 1.0);
        // White stones = 30/30 = 1.0
        assert!((t.data[25 * 64] - 1.0).abs() < 1e-6);
        // Black stones = 30/30 = 1.0
        assert!((t.data[26 * 64] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn piece_encoding_channel() {
        let mut state = GameState::new(GameConfig::standard(5));
        // Place a black flat at (0,0) (opening move).
        state.apply_move(Move::Place {
            square: Square::from_rc(0, 0),
            piece_type: PieceType::Flat,
        });
        // Now (0,0) has a black flat (opening places opponent's piece).
        let t = BoardTensor::encode(&state);
        let si = 0; // row 0, col 0
                    // BlackFlat = piece variant 3, so channel 3 should be 1.0.
        assert_eq!(t.data[3 * 64 + si], 1.0);
        // is_occupied (channel 6)
        assert_eq!(t.data[6 * 64 + si], 1.0);
        // WhiteFlat (channel 0) should be 0.
        assert_eq!(t.data[si], 0.0);
    }

    #[test]
    fn non_active_squares_zero_spatial() {
        let state = GameState::new(GameConfig::standard(3));
        let t = BoardTensor::encode(&state);
        // Square (4, 4) is outside the 3x3 active region.
        let si = 4 * 8 + 4;
        for ch in 0..23 {
            assert_eq!(
                t.data[ch * 64 + si],
                0.0,
                "channel {} at (4,4) should be 0",
                ch
            );
        }
    }

    #[test]
    fn global_channels_broadcast() {
        let state = GameState::new(GameConfig::standard(5));
        let t = BoardTensor::encode(&state);
        // Channel 23 (side to move) should be same at all 64 positions.
        let val = t.data[23 * 64];
        for i in 1..64 {
            assert_eq!(t.data[23 * 64 + i], val);
        }
    }

    #[test]
    fn size_id_all_sizes() {
        for size in 3..=8u8 {
            let state = GameState::new(GameConfig::standard(size));
            let t = BoardTensor::encode(&state);
            assert_eq!(t.size_id, size - 3);
        }
    }
}
