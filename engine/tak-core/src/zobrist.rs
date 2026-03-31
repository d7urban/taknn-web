//! Zobrist hashing for Tak positions.
//!
//! The hash encodes:
//! - Board contents: piece types and colors at every stack layer (top piece,
//!   up to 7 interior layers, and buried counts per color per square).
//! - Side to move.
//! - Reserves remaining (white stones, white caps, black stones, black caps).
//!
//! It does NOT include ply count, komi, or hash_history.
//!
//! Hashing uses XOR of pre-generated random u64 keys. For incremental updates,
//! XOR in the old component and XOR in the new component (since XOR is its own
//! inverse).

use std::sync::LazyLock;

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

use crate::board::{Board, Square, Stack};
use crate::piece::{Color, Piece};

/// Maximum buried count per color per square that we pre-generate keys for.
/// In the worst case (8x8, 50 stones + 2 caps = 52 pieces per player, minus
/// the 8 explicit layers), a single square could have up to ~44 buried pieces
/// of one color. We round up to 50 for safety.
const MAX_BURIED: usize = 50;

/// Maximum reserve count for any single reserve type.
/// Largest is 50 stones on 8x8.
const MAX_RESERVE: usize = 51; // 0..=50

/// Pre-generated Zobrist keys. Initialized once, accessible globally.
pub static KEYS: LazyLock<ZobristKeys> = LazyLock::new(ZobristKeys::new);

/// Collection of all pre-generated random u64 keys used for Zobrist hashing.
pub struct ZobristKeys {
    /// Key for the top piece at each square.
    /// Indexed by `[square_index][piece_variant]` where piece_variant is
    /// `Piece as u8` (0..6: WhiteFlat, WhiteWall, WhiteCap, BlackFlat, BlackWall, BlackCap).
    pub top_piece: [[u64; 6]; 64],

    /// Key for each interior layer at each square.
    /// Indexed by `[square_index][layer_index][color]` where layer_index is 0..7
    /// (0 = just below top, 6 = deepest explicit layer) and color is 0=White, 1=Black.
    pub interior: [[[u64; 2]; 7]; 64],

    /// Key for each possible buried count per color per square.
    /// Indexed by `[square_index][color][count]` where count is 0..MAX_BURIED.
    /// Using count=0 contributes nothing (key is 0), so we only store 1..MAX_BURIED
    /// but index from 0 for simplicity (key at index 0 is still random but we skip it
    /// in compute_full by only hashing when count > 0).
    pub buried: [[[u64; MAX_BURIED]; 2]; 64],

    /// Key XORed in when it is Black's turn to move.
    /// When it is White's turn, this key is not included in the hash.
    pub side: u64,

    /// Keys for reserve counts.
    /// Indexed by `[reserve_type][count]` where reserve_type is:
    ///   0 = white stones, 1 = white caps, 2 = black stones, 3 = black caps.
    /// Count ranges from 0 to 50 inclusive.
    pub reserves: [[u64; MAX_RESERVE]; 4],
}

impl ZobristKeys {
    /// Generate all Zobrist keys using a deterministic PRNG with a fixed seed.
    fn new() -> Self {
        // Use a recognizable but arbitrary fixed seed.
        let mut rng = Xoshiro256StarStar::seed_from_u64(0x5441_4B5A_0B21_5700);

        let mut keys = ZobristKeys {
            top_piece: [[0; 6]; 64],
            interior: [[[0; 2]; 7]; 64],
            buried: [[[0; MAX_BURIED]; 2]; 64],
            side: 0,
            reserves: [[0; MAX_RESERVE]; 4],
        };

        // Top pieces: 64 squares x 6 piece variants
        for sq in 0..64 {
            for piece in 0..6 {
                keys.top_piece[sq][piece] = rng.random();
            }
        }

        // Interior layers: 64 squares x 7 layers x 2 colors
        for sq in 0..64 {
            for layer in 0..7 {
                for color in 0..2 {
                    keys.interior[sq][layer][color] = rng.random();
                }
            }
        }

        // Buried counts: 64 squares x 2 colors x MAX_BURIED counts
        for sq in 0..64 {
            for color in 0..2 {
                for count in 0..MAX_BURIED {
                    keys.buried[sq][color][count] = rng.random();
                }
            }
        }

        // Side to move
        keys.side = rng.random();

        // Reserves: 4 types x MAX_RESERVE counts
        for rtype in 0..4 {
            for count in 0..MAX_RESERVE {
                keys.reserves[rtype][count] = rng.random();
            }
        }

        keys
    }
}

// ---------------------------------------------------------------------------
// Full hash computation
// ---------------------------------------------------------------------------

/// Compute the full Zobrist hash from scratch for a given position.
///
/// `board` is the 8x8 board, `size` is the active NxN region,
/// `side_to_move` is whose turn it is, and `reserves` is
/// `[white_stones, white_caps, black_stones, black_caps]`.
pub fn compute_full(
    board: &Board,
    size: u8,
    side_to_move: Color,
    reserves: &[u8; 4],
) -> u64 {
    let keys = &*KEYS;
    let mut hash: u64 = 0;

    // Hash board contents within the active NxN region.
    for r in 0..size {
        for c in 0..size {
            let sq = Square::from_rc(r, c);
            let idx = sq.0 as usize;
            let stack = board.get(sq);

            hash ^= hash_stack(keys, idx, stack);
        }
    }

    // Side to move
    if side_to_move == Color::Black {
        hash ^= keys.side;
    }

    // Reserves
    for rtype in 0..4 {
        let count = reserves[rtype] as usize;
        debug_assert!(count < MAX_RESERVE, "reserve count {} out of range", count);
        hash ^= keys.reserves[rtype][count];
    }

    hash
}

/// Compute the hash contribution of a single stack.
#[inline]
fn hash_stack(keys: &ZobristKeys, sq: usize, stack: &Stack) -> u64 {
    let mut h: u64 = 0;

    // Top piece
    if let Some(top) = stack.top {
        h ^= keys.top_piece[sq][top as u8 as usize];
    }

    // Interior layers (below array, from index 0 = just below top)
    for (layer_idx, &color) in stack.below.iter().enumerate() {
        h ^= keys.interior[sq][layer_idx][color as u8 as usize];
    }

    // Buried counts
    let bw = stack.buried_white as usize;
    let bb = stack.buried_black as usize;
    if bw > 0 {
        debug_assert!(bw < MAX_BURIED, "buried_white {} out of range at sq {}", bw, sq);
        h ^= keys.buried[sq][0][bw];
    }
    if bb > 0 {
        debug_assert!(bb < MAX_BURIED, "buried_black {} out of range at sq {}", bb, sq);
        h ^= keys.buried[sq][1][bb];
    }

    h
}

// ---------------------------------------------------------------------------
// Incremental update helpers
// ---------------------------------------------------------------------------

/// Return the hash contribution of the top piece at `sq`.
/// Useful for XOR-ing out the old top and XOR-ing in the new top.
#[inline]
pub fn hash_top_piece(sq: usize, piece: Piece) -> u64 {
    KEYS.top_piece[sq][piece as u8 as usize]
}

/// Return the hash contribution of an interior layer at `sq`.
#[inline]
pub fn hash_interior(sq: usize, layer_idx: usize, color: Color) -> u64 {
    KEYS.interior[sq][layer_idx][color as u8 as usize]
}

/// Return the hash contribution of a buried count at `sq`.
/// Returns 0 for count == 0 (no contribution).
#[inline]
pub fn hash_buried(sq: usize, color: Color, count: usize) -> u64 {
    if count == 0 {
        0
    } else {
        debug_assert!(count < MAX_BURIED);
        KEYS.buried[sq][color as u8 as usize][count]
    }
}

/// Return the side-to-move toggle key.
/// XOR this to flip sides (works both ways since XOR is self-inverse).
#[inline]
pub fn hash_side() -> u64 {
    KEYS.side
}

/// Return the hash contribution for a reserve type and count.
/// `reserve_type`: 0 = white stones, 1 = white caps, 2 = black stones, 3 = black caps.
#[inline]
pub fn hash_reserve(reserve_type: usize, count: usize) -> u64 {
    debug_assert!(reserve_type < 4);
    debug_assert!(count < MAX_RESERVE);
    KEYS.reserves[reserve_type][count]
}

/// Compute the XOR diff to update the hash when a stack at `sq` changes.
///
/// Returns `old_contribution ^ new_contribution`. XOR this with the
/// current hash to update it.
///
/// This is useful for apply/undo of place and spread moves.
pub fn hash_stack_diff(sq: usize, old_stack: &Stack, new_stack: &Stack) -> u64 {
    let keys = &*KEYS;
    hash_stack(keys, sq, old_stack) ^ hash_stack(keys, sq, new_stack)
}

/// Compute the XOR diff when reserves change.
///
/// Returns the XOR of old and new reserve hash contributions for the
/// specified reserve type.
#[inline]
pub fn hash_reserve_diff(reserve_type: usize, old_count: u8, new_count: u8) -> u64 {
    hash_reserve(reserve_type, old_count as usize) ^ hash_reserve(reserve_type, new_count as usize)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::PieceType;

    /// Helper: build a board with one flat placed at (0,0).
    fn board_with_one_piece() -> Board {
        let mut board = Board::empty();
        board
            .get_mut(Square::from_rc(0, 0))
            .push(Piece::new(Color::White, PieceType::Flat));
        board
    }

    #[test]
    fn hash_is_deterministic() {
        // Computing the hash twice for the same position must yield the same value.
        let board = Board::empty();
        let reserves = [30u8, 1, 30, 1]; // 6x6 standard
        let h1 = compute_full(&board, 6, Color::White, &reserves);
        let h2 = compute_full(&board, 6, Color::White, &reserves);
        assert_eq!(h1, h2, "hash must be deterministic");
    }

    #[test]
    fn empty_board_hash_nonzero() {
        // Even an empty board has reserves, so the hash should be nonzero.
        let board = Board::empty();
        let reserves = [30u8, 1, 30, 1];
        let h = compute_full(&board, 6, Color::White, &reserves);
        assert_ne!(h, 0, "empty board hash should be nonzero (reserves contribute)");
    }

    #[test]
    fn hash_changes_when_piece_placed() {
        let reserves = [30u8, 1, 30, 1];
        let empty_hash = compute_full(&Board::empty(), 6, Color::White, &reserves);

        let board = board_with_one_piece();
        let placed_hash = compute_full(&board, 6, Color::White, &reserves);

        assert_ne!(
            empty_hash, placed_hash,
            "placing a piece must change the hash"
        );
    }

    #[test]
    fn hash_changes_with_side_to_move() {
        let board = Board::empty();
        let reserves = [30u8, 1, 30, 1];
        let h_white = compute_full(&board, 6, Color::White, &reserves);
        let h_black = compute_full(&board, 6, Color::Black, &reserves);
        assert_ne!(
            h_white, h_black,
            "hash must differ for different side to move"
        );
        // The difference should be exactly the side key.
        assert_eq!(h_white ^ h_black, KEYS.side);
    }

    #[test]
    fn hash_changes_with_reserves() {
        let board = Board::empty();
        let r1 = [30u8, 1, 30, 1];
        let r2 = [29u8, 1, 30, 1]; // white placed one stone
        let h1 = compute_full(&board, 6, Color::White, &r1);
        let h2 = compute_full(&board, 6, Color::White, &r2);
        assert_ne!(h1, h2, "different reserves must produce different hashes");
    }

    #[test]
    fn incremental_side_toggle() {
        let board = Board::empty();
        let reserves = [30u8, 1, 30, 1];
        let h = compute_full(&board, 6, Color::White, &reserves);
        // Toggle side: should match compute_full with Black.
        let h_toggled = h ^ hash_side();
        let h_black = compute_full(&board, 6, Color::Black, &reserves);
        assert_eq!(h_toggled, h_black);
    }

    #[test]
    fn incremental_place_piece() {
        let reserves_before = [30u8, 1, 30, 1];
        let reserves_after = [29u8, 1, 30, 1];

        // Start: empty board, white to move.
        let board_before = Board::empty();
        let h_before = compute_full(&board_before, 6, Color::White, &reserves_before);

        // Place white flat at (0,0), switch to black, decrement white stones.
        let board_after = board_with_one_piece();
        let h_expected = compute_full(&board_after, 6, Color::Black, &reserves_after);

        // Incremental: XOR out old stack, XOR in new stack, toggle side, update reserve.
        let sq = Square::from_rc(0, 0).0 as usize;
        let h_incremental = h_before
            ^ hash_stack_diff(sq, &Stack::empty(), board_after.get(Square::from_rc(0, 0)))
            ^ hash_side()
            ^ hash_reserve_diff(0, 30, 29);

        assert_eq!(h_incremental, h_expected, "incremental update must match full recompute");
    }

    #[test]
    fn stack_with_interior_layers() {
        let mut board = Board::empty();
        let sq = Square::from_rc(2, 2);
        {
            let stack = board.get_mut(sq);
            stack.push(Piece::new(Color::White, PieceType::Flat));
            stack.push(Piece::new(Color::Black, PieceType::Flat));
            stack.push(Piece::new(Color::White, PieceType::Wall));
        }

        let reserves = [28u8, 1, 29, 1];
        let h = compute_full(&board, 6, Color::White, &reserves);

        // Rebuild with a different interior arrangement to verify hash differs.
        let mut board2 = Board::empty();
        {
            let stack = board2.get_mut(sq);
            stack.push(Piece::new(Color::Black, PieceType::Flat));
            stack.push(Piece::new(Color::White, PieceType::Flat));
            stack.push(Piece::new(Color::White, PieceType::Wall));
        }
        let h2 = compute_full(&board2, 6, Color::White, &reserves);

        assert_ne!(h, h2, "different interior layer colors must hash differently");
    }

    #[test]
    fn different_top_pieces_hash_differently() {
        let reserves = [30u8, 1, 30, 1];
        let sq = Square::from_rc(0, 0);

        let mut board_flat = Board::empty();
        board_flat
            .get_mut(sq)
            .push(Piece::new(Color::White, PieceType::Flat));
        let h_flat = compute_full(&board_flat, 6, Color::White, &reserves);

        let mut board_wall = Board::empty();
        board_wall
            .get_mut(sq)
            .push(Piece::new(Color::White, PieceType::Wall));
        let h_wall = compute_full(&board_wall, 6, Color::White, &reserves);

        let mut board_cap = Board::empty();
        board_cap
            .get_mut(sq)
            .push(Piece::new(Color::White, PieceType::Cap));
        let h_cap = compute_full(&board_cap, 6, Color::White, &reserves);

        assert_ne!(h_flat, h_wall);
        assert_ne!(h_flat, h_cap);
        assert_ne!(h_wall, h_cap);
    }

    #[test]
    fn different_squares_hash_differently() {
        let reserves = [30u8, 1, 30, 1];

        let mut board1 = Board::empty();
        board1
            .get_mut(Square::from_rc(0, 0))
            .push(Piece::new(Color::White, PieceType::Flat));
        let h1 = compute_full(&board1, 6, Color::White, &reserves);

        let mut board2 = Board::empty();
        board2
            .get_mut(Square::from_rc(1, 1))
            .push(Piece::new(Color::White, PieceType::Flat));
        let h2 = compute_full(&board2, 6, Color::White, &reserves);

        assert_ne!(h1, h2, "same piece on different squares must hash differently");
    }

    #[test]
    fn buried_pieces_change_hash() {
        let reserves = [20u8, 1, 20, 1];
        let sq = Square::from_rc(0, 0);

        // Build a tall stack that overflows into buried layers.
        // Push 10 white flats then a black cap on top.
        let mut board = Board::empty();
        {
            let stack = board.get_mut(sq);
            for _ in 0..10 {
                stack.push(Piece::new(Color::White, PieceType::Flat));
            }
            stack.push(Piece::new(Color::Black, PieceType::Cap));
        }
        let h1 = compute_full(&board, 6, Color::White, &reserves);

        // Same but push 9 white + 1 black then black cap.
        let mut board2 = Board::empty();
        {
            let stack = board2.get_mut(sq);
            for _ in 0..9 {
                stack.push(Piece::new(Color::White, PieceType::Flat));
            }
            stack.push(Piece::new(Color::Black, PieceType::Flat));
            stack.push(Piece::new(Color::Black, PieceType::Cap));
        }
        let h2 = compute_full(&board2, 6, Color::White, &reserves);

        assert_ne!(h1, h2, "different buried compositions must hash differently");
    }

    #[test]
    fn no_collisions_in_key_tables() {
        // Verify that all generated keys are unique (no accidental collisions
        // in the PRNG output). This is a basic sanity check.
        let keys = &*KEYS;
        let mut all_keys = Vec::new();

        for sq in 0..64 {
            for p in 0..6 {
                all_keys.push(keys.top_piece[sq][p]);
            }
        }
        for sq in 0..64 {
            for layer in 0..7 {
                for color in 0..2 {
                    all_keys.push(keys.interior[sq][layer][color]);
                }
            }
        }
        for sq in 0..64 {
            for color in 0..2 {
                for count in 0..MAX_BURIED {
                    all_keys.push(keys.buried[sq][color][count]);
                }
            }
        }
        all_keys.push(keys.side);
        for rtype in 0..4 {
            for count in 0..MAX_RESERVE {
                all_keys.push(keys.reserves[rtype][count]);
            }
        }

        let total = all_keys.len();
        let mut sorted = all_keys.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            total,
            sorted.len(),
            "all Zobrist keys must be unique; found {} duplicates",
            total - sorted.len()
        );
    }

    // -------------------------------------------------------------------
    // AC 1.18: No Zobrist collisions in 10,000 random positions
    // -------------------------------------------------------------------

    #[test]
    fn no_collisions_in_10000_random_positions() {
        use rand::SeedableRng;
        use rand::Rng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use crate::rules::GameConfig;
        use crate::state::{GameState, GameResult};
        use crate::tps;
        use std::collections::HashMap;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12345);
        // Map from zobrist hash -> TPS of the first position with that hash.
        // A collision is when two DIFFERENT positions produce the same hash.
        let mut seen: HashMap<u64, String> = HashMap::new();
        let mut total = 0;

        for size in 3..=7u8 {
            let config = GameConfig::standard(size);
            let games = match size {
                3 => 100,
                4 => 80,
                5 => 60,
                6 => 40,
                _ => 30,
            };
            let max_plies = match size {
                3 => 18,
                4 => 32,
                5 => 50,
                6 => 72,
                _ => 98,
            };

            for _ in 0..games {
                let mut state = GameState::new(config);

                // Extract the position-relevant part of TPS (board + player,
                // excluding move number which isn't part of the hash).
                let pos_key = |state: &GameState| -> String {
                    let full = tps::to_tps(state);
                    // TPS = "board player move_number"; strip move_number.
                    let last_space = full.rfind(' ').unwrap();
                    full[..last_space].to_string()
                };

                let check = |state: &GameState, seen: &mut HashMap<u64, String>| {
                    let key = pos_key(state);
                    if let Some(prev) = seen.get(&state.zobrist) {
                        assert_eq!(
                            prev, &key,
                            "Zobrist collision: hash {:016x} maps to both '{}' and '{}'",
                            state.zobrist, prev, key
                        );
                    } else {
                        seen.insert(state.zobrist, key);
                    }
                };

                check(&state, &mut seen);
                total += 1;

                for _ in 0..max_plies {
                    if state.result != GameResult::Ongoing {
                        break;
                    }
                    let moves = state.legal_moves();
                    if moves.is_empty() {
                        break;
                    }
                    let idx = rng.random_range(0..moves.len());
                    state.apply_move(moves[idx]);
                    check(&state, &mut seen);
                    total += 1;
                }
            }
        }

        assert!(
            total >= 10_000,
            "expected 10,000+ positions, got {}",
            total
        );
    }
}
