//! TPS (Tak Positional System) parser and serializer.
//!
//! TPS is a notation for Tak board positions, analogous to FEN in chess.
//!
//! Format: `[board] [player] [move_number]`
//!
//! - `[board]` is a `/`-separated list of rows (top to bottom). Within a row,
//!   squares are comma-separated. Empty runs are a digit `1`–`8`; stacks are
//!   sequences of `1` (white) / `2` (black) with an optional `S` (wall) or `C`
//!   (capstone) suffix on the topmost piece. `x` is an alias for `1` (one empty
//!   square).
//! - `[player]` is `1` (White) or `2` (Black).
//! - `[move_number]` is the 1-based full-move counter.

use crate::board::{Board, Square};
use crate::piece::{Color, Piece, PieceType};
use crate::rules::GameConfig;
use crate::state::GameState;
use crate::templates::TemplateTable;
use crate::zobrist;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum TpsError {
    InvalidFormat(String),
    InvalidPiece(String),
    InvalidPlayer(String),
    InvalidMoveNumber(String),
    SizeMismatch(String),
}

impl std::fmt::Display for TpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TpsError::InvalidFormat(s) => write!(f, "invalid TPS format: {}", s),
            TpsError::InvalidPiece(s) => write!(f, "invalid piece in TPS: {}", s),
            TpsError::InvalidPlayer(s) => write!(f, "invalid player in TPS: {}", s),
            TpsError::InvalidMoveNumber(s) => write!(f, "invalid move number in TPS: {}", s),
            TpsError::SizeMismatch(s) => write!(f, "TPS size mismatch: {}", s),
        }
    }
}

impl std::error::Error for TpsError {}

// ---------------------------------------------------------------------------
// Serialization: GameState -> TPS
// ---------------------------------------------------------------------------

/// Serialize a `GameState` into TPS notation.
pub fn to_tps(state: &GameState) -> String {
    let size = state.config.size;
    let mut parts: Vec<String> = Vec::with_capacity(size as usize);

    // Rows from top (row size-1) to bottom (row 0).
    for r in (0..size).rev() {
        let mut row_parts: Vec<String> = Vec::new();
        let mut empty_run: u8 = 0;

        for c in 0..size {
            let sq = Square::from_rc(r, c);
            let stack = state.board.get(sq);

            if stack.is_empty() {
                empty_run += 1;
            } else {
                // Flush any accumulated empty squares.
                if empty_run > 0 {
                    if empty_run == 1 {
                        row_parts.push("x".to_string());
                    } else {
                        row_parts.push(format!("x{}", empty_run));
                    }
                    empty_run = 0;
                }
                row_parts.push(serialize_stack(stack));
            }
        }

        // Flush trailing empties.
        if empty_run > 0 {
            if empty_run == 1 {
                row_parts.push("x".to_string());
            } else {
                row_parts.push(format!("x{}", empty_run));
            }
        }

        parts.push(row_parts.join(","));
    }

    let board_str = parts.join("/");
    let player = match state.side_to_move {
        Color::White => '1',
        Color::Black => '2',
    };
    let move_number = state.ply / 2 + 1;

    format!("{} {} {}", board_str, player, move_number)
}

/// Serialize a single (non-empty) stack to TPS square notation.
fn serialize_stack(stack: &crate::board::Stack) -> String {
    let top = stack.top.expect("serialize_stack called on empty stack");
    let height = stack.height as usize;

    if height == 1 {
        // Single piece.
        let color_ch = match top.color() {
            Color::White => '1',
            Color::Black => '2',
        };
        return match top.piece_type() {
            PieceType::Flat => format!("{}", color_ch),
            PieceType::Wall => format!("{}S", color_ch),
            PieceType::Cap => format!("{}C", color_ch),
        };
    }

    // Multi-piece stack. We need to list pieces bottom-to-top.
    //
    // The stack stores:
    //   - `top`: the top piece (with full type info)
    //   - `below`: up to 7 colors, index 0 = just below top, index N = further down
    //   - `buried_white`, `buried_black`: counts of pieces below the explicit layers
    //
    // We reconstruct bottom-to-top order:
    //   buried pieces (order unknown, but we output white then black by convention),
    //   then `below` reversed, then `top`.
    let mut chars = String::with_capacity(height + 1);

    // Buried pieces: we don't know the true interleaving. Output white first,
    // then black. (This is a lossy area of the Stack representation, but for
    // positions that reach this deep it matches the data we have.)
    for _ in 0..stack.buried_white {
        chars.push('1');
    }
    for _ in 0..stack.buried_black {
        chars.push('2');
    }

    // Explicit interior layers, bottom-to-top (reverse the `below` array).
    for &color in stack.below.iter().rev() {
        chars.push(match color {
            Color::White => '1',
            Color::Black => '2',
        });
    }

    // Top piece color.
    chars.push(match top.color() {
        Color::White => '1',
        Color::Black => '2',
    });

    // Suffix for non-flat top.
    match top.piece_type() {
        PieceType::Flat => {}
        PieceType::Wall => chars.push('S'),
        PieceType::Cap => chars.push('C'),
    }

    chars
}

// ---------------------------------------------------------------------------
// Deserialization: TPS -> GameState
// ---------------------------------------------------------------------------

/// Parse a TPS string into a `GameState`.
///
/// The board size is inferred from the number of rows. A standard `GameConfig`
/// is used; reserves are computed by subtracting pieces on the board from the
/// standard totals.
pub fn from_tps(tps: &str) -> Result<GameState, TpsError> {
    let tps = tps.trim();
    let sections: Vec<&str> = tps.split_whitespace().collect();
    if sections.len() != 3 {
        return Err(TpsError::InvalidFormat(format!(
            "expected 3 space-separated sections, got {}",
            sections.len()
        )));
    }

    let board_str = sections[0];
    let player_str = sections[1];
    let move_str = sections[2];

    // --- Player to move ---
    let side_to_move = match player_str {
        "1" => Color::White,
        "2" => Color::Black,
        _ => {
            return Err(TpsError::InvalidPlayer(format!(
                "expected '1' or '2', got '{}'",
                player_str
            )))
        }
    };

    // --- Move number ---
    let move_number: u16 = move_str.parse().map_err(|_| {
        TpsError::InvalidMoveNumber(format!("'{}' is not a valid move number", move_str))
    })?;
    if move_number == 0 {
        return Err(TpsError::InvalidMoveNumber(
            "move number must be >= 1".to_string(),
        ));
    }

    // Ply = (move_number - 1) * 2 + (0 if White, 1 if Black)
    let ply: u16 = (move_number - 1) * 2
        + match side_to_move {
            Color::White => 0,
            Color::Black => 1,
        };

    // --- Board ---
    let rows: Vec<&str> = board_str.split('/').collect();
    let size = rows.len() as u8;
    if !(3..=8).contains(&size) {
        return Err(TpsError::SizeMismatch(format!(
            "board size {} is not supported (must be 3-8)",
            size
        )));
    }

    let config = GameConfig::standard(size);
    let mut board = Board::empty();

    // Rows in TPS go from top (row size-1) to bottom (row 0).
    for (row_idx, row_str) in rows.iter().enumerate() {
        let r = size - 1 - row_idx as u8; // board row
        let squares: Vec<&str> = row_str.split(',').collect();

        let mut c: u8 = 0;
        for token in &squares {
            if c >= size {
                return Err(TpsError::SizeMismatch(format!(
                    "row {} has too many squares",
                    row_idx
                )));
            }

            if let Some(count) = parse_empty_token(token) {
                // Empty squares.
                if c + count > size {
                    return Err(TpsError::SizeMismatch(format!(
                        "row {} overflows: empty run of {} at column {}",
                        row_idx, count, c
                    )));
                }
                c += count;
            } else {
                // Stack description.
                let pieces = parse_stack(token)?;
                let sq = Square::from_rc(r, c);
                let stack = board.get_mut(sq);
                for piece in pieces {
                    stack.push(piece);
                }
                c += 1;
            }
        }

        if c != size {
            return Err(TpsError::SizeMismatch(format!(
                "row {} has {} squares, expected {}",
                row_idx, c, size
            )));
        }
    }

    // --- Compute reserves ---
    // Start from standard totals and subtract pieces on the board.
    let mut reserves = [config.stones, config.capstones, config.stones, config.capstones];
    for r in 0..size {
        for c in 0..size {
            let sq = Square::from_rc(r, c);
            let stack = board.get(sq);

            if let Some(top) = stack.top {
                subtract_reserve(&mut reserves, top)?;
            }

            // Interior pieces (below array): all flats.
            for &color in stack.below.iter() {
                let piece = Piece::new(color, PieceType::Flat);
                subtract_reserve(&mut reserves, piece)?;
            }

            // Buried pieces.
            for _ in 0..stack.buried_white {
                subtract_reserve(&mut reserves, Piece::WhiteFlat)?;
            }
            for _ in 0..stack.buried_black {
                subtract_reserve(&mut reserves, Piece::BlackFlat)?;
            }
        }
    }

    // --- Zobrist hash ---
    let zobrist = zobrist::compute_full(&board, size, side_to_move, &reserves);

    // --- Assemble GameState ---
    let templates = TemplateTable::build(config.size);
    let state = GameState {
        board,
        config,
        side_to_move,
        ply,
        reserves,
        result: crate::state::GameResult::Ongoing,
        zobrist,
        hash_history: vec![zobrist],
        templates,
    };

    Ok(state)
}

/// Try to parse a token as an empty-square indicator.
/// Returns `Some(count)` if the token is `x`, `x1`..`x8`, or just a digit `1`..`8`.
fn parse_empty_token(token: &str) -> Option<u8> {
    if token == "x" {
        return Some(1);
    }
    if let Some(rest) = token.strip_prefix('x') {
        if let Ok(n) = rest.parse::<u8>() {
            if (1..=8).contains(&n) {
                return Some(n);
            }
        }
        return None;
    }
    // A bare digit 1-8 is only treated as empty squares if the token is
    // exactly one character that is a digit -- but this conflicts with `1`
    // or `2` meaning a single piece. In standard TPS, bare digits are NOT
    // used for empty squares; `x` or `xN` is used. So we don't treat bare
    // digits as empty counts here; they are parsed as stacks.
    None
}

/// Parse a stack token (e.g. `1`, `2S`, `12`, `21C`) into a sequence of
/// `Piece` values from bottom to top.
fn parse_stack(token: &str) -> Result<Vec<Piece>, TpsError> {
    if token.is_empty() {
        return Err(TpsError::InvalidPiece("empty square token".to_string()));
    }

    let bytes = token.as_bytes();
    let len = bytes.len();

    // Determine if the last character is a suffix (S or C).
    let (suffix, piece_end) = match bytes[len - 1] {
        b'S' => (Some(PieceType::Wall), len - 1),
        b'C' => (Some(PieceType::Cap), len - 1),
        _ => (None, len),
    };

    if piece_end == 0 {
        return Err(TpsError::InvalidPiece(format!(
            "stack '{}' has suffix but no piece",
            token
        )));
    }

    let mut pieces: Vec<Piece> = Vec::with_capacity(piece_end);

    for (i, &b) in bytes[..piece_end].iter().enumerate() {
        let color = match b {
            b'1' => Color::White,
            b'2' => Color::Black,
            _ => {
                return Err(TpsError::InvalidPiece(format!(
                    "unexpected character '{}' in stack '{}'",
                    b as char, token
                )))
            }
        };

        let is_top = i == piece_end - 1;
        let piece_type = if is_top {
            suffix.unwrap_or(PieceType::Flat)
        } else {
            PieceType::Flat
        };

        pieces.push(Piece::new(color, piece_type));
    }

    Ok(pieces)
}

/// Subtract one piece from the reserves array. Returns an error if the
/// reserve would go negative.
fn subtract_reserve(reserves: &mut [u8; 4], piece: Piece) -> Result<(), TpsError> {
    let idx = match piece.piece_type() {
        PieceType::Flat | PieceType::Wall => match piece.color() {
            Color::White => 0,
            Color::Black => 2,
        },
        PieceType::Cap => match piece.color() {
            Color::White => 1,
            Color::Black => 3,
        },
    };

    if reserves[idx] == 0 {
        return Err(TpsError::InvalidPiece(format!(
            "too many {:?} {:?} pieces on the board (reserves exhausted)",
            piece.color(),
            piece.piece_type()
        )));
    }
    reserves[idx] -= 1;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: to_tps(from_tps(s)) == s for empty boards of all sizes.
    #[test]
    fn roundtrip_empty_boards() {
        let cases = [
            ("x3/x3/x3 1 1", 3),
            ("x4/x4/x4/x4 1 1", 4),
            ("x5/x5/x5/x5/x5 1 1", 5),
            ("x6/x6/x6/x6/x6/x6 1 1", 6),
            ("x7/x7/x7/x7/x7/x7/x7 1 1", 7),
            ("x8/x8/x8/x8/x8/x8/x8/x8 1 1", 8),
        ];
        for (tps, expected_size) in &cases {
            let state = from_tps(tps).unwrap_or_else(|e| panic!("failed to parse '{}': {}", tps, e));
            assert_eq!(state.config.size, *expected_size, "size mismatch for '{}'", tps);
            assert_eq!(state.side_to_move, Color::White);
            assert_eq!(state.ply, 0);

            let serialized = to_tps(&state);
            assert_eq!(&serialized, tps, "round-trip failed for size {}", expected_size);
        }
    }

    /// Round-trip for a position with stacks, walls, and capstones.
    #[test]
    fn roundtrip_stacks_walls_caps() {
        // 5x5 board with various stack types.
        let tps = "x3,12,x/x,2S,x3/x5/x5/x,1,x3 2 3";
        let state = from_tps(tps).unwrap();
        assert_eq!(state.config.size, 5);
        assert_eq!(state.side_to_move, Color::Black);
        // ply = (3-1)*2 + 1 = 5
        assert_eq!(state.ply, 5);

        let serialized = to_tps(&state);
        assert_eq!(serialized, tps);
    }

    /// Round-trip for a position with a capstone and a multi-piece stack.
    #[test]
    fn roundtrip_capstone_stack() {
        // Row 4 (top): empty, empty, white cap, empty, empty
        // Row 2:       empty, stack(black flat + white flat), empty, empty, empty
        let tps = "x2,1C,x2/x5/x,21,x3/x5/x5 1 4";
        let state = from_tps(tps).unwrap();
        assert_eq!(state.config.size, 5);

        let serialized = to_tps(&state);
        assert_eq!(serialized, tps);

        // Verify the capstone at board row 4, col 2.
        let sq = Square::from_rc(4, 2);
        let stack = state.board.get(sq);
        assert_eq!(stack.height, 1);
        assert_eq!(stack.top, Some(Piece::WhiteCap));

        // Verify the stack at board row = 5-1-2 = 2, col = 1.
        let sq = Square::from_rc(2, 1);
        let stack = state.board.get(sq);
        assert_eq!(stack.height, 2);
        assert_eq!(stack.top, Some(Piece::WhiteFlat));
        assert_eq!(stack.below.len(), 1);
        assert_eq!(stack.below[0], Color::Black);
    }

    /// Round-trip for a position with capstone and larger multi-piece stack.
    #[test]
    fn roundtrip_capstone_larger_stack() {
        // 6x6 board with a 3-piece stack with capstone on top.
        let tps = "x3,1C,x2/x6/x,212C,x4/x6/x6/x6 1 4";
        let state = from_tps(tps).unwrap();
        assert_eq!(state.config.size, 6);

        let serialized = to_tps(&state);
        assert_eq!(serialized, tps);

        // Verify the stack at board row = 6-1-2 = 3, col=1.
        let sq = Square::from_rc(3, 1);
        let stack = state.board.get(sq);
        assert_eq!(stack.height, 3);
        assert_eq!(stack.top, Some(Piece::BlackCap));
        assert_eq!(stack.below.len(), 2);
        assert_eq!(stack.below[0], Color::White); // just below top
        assert_eq!(stack.below[1], Color::Black); // bottom
    }

    /// Parse error cases.
    #[test]
    fn parse_errors() {
        // Wrong number of sections.
        assert!(from_tps("x5/x5/x5/x5/x5 1").is_err());
        assert!(from_tps("x5/x5/x5/x5/x5").is_err());
        assert!(from_tps("").is_err());

        // Invalid player.
        assert!(from_tps("x5/x5/x5/x5/x5 3 1").is_err());
        assert!(from_tps("x5/x5/x5/x5/x5 0 1").is_err());

        // Invalid move number.
        assert!(from_tps("x5/x5/x5/x5/x5 1 0").is_err());
        assert!(from_tps("x5/x5/x5/x5/x5 1 abc").is_err());

        // Row overflow.
        assert!(from_tps("x6/x5/x5/x5/x5 1 1").is_err());

        // Invalid piece character.
        assert!(from_tps("x4,3/x5/x5/x5/x5 1 1").is_err());

        // Too few squares in a row.
        assert!(from_tps("x4/x5/x5/x5/x5 1 1").is_err());

        // Invalid board size (2 rows = size 2).
        assert!(from_tps("x2/x2 1 1").is_err());
    }

    /// Opening position TPS.
    #[test]
    fn opening_position() {
        let tps = "x5/x5/x5/x5/x5 1 1";
        let state = from_tps(tps).unwrap();

        assert_eq!(state.config.size, 5);
        assert_eq!(state.side_to_move, Color::White);
        assert_eq!(state.ply, 0);

        // All squares should be empty.
        for r in 0..5 {
            for c in 0..5 {
                assert!(state.board.get(Square::from_rc(r, c)).is_empty());
            }
        }

        // Reserves should be full.
        assert_eq!(state.reserves[0], 21); // white stones
        assert_eq!(state.reserves[1], 1);  // white caps
        assert_eq!(state.reserves[2], 21); // black stones
        assert_eq!(state.reserves[3], 1);  // black caps
    }

    /// A complex midgame position.
    #[test]
    fn complex_midgame() {
        // 5x5 with various piece types:
        //   Row 4 (top):    empty, white wall, stack 121S, empty, black cap
        //   Row 3:          all empty
        //   Row 2:          black flat, empty x3, white flat
        //   Row 1:          all empty
        //   Row 0 (bottom): white flat, 12, empty, 2, empty
        let tps = "x,1S,121S,x,2C/x5/2,x3,1/x5/1,12,x,2,x 2 5";
        let state = from_tps(tps).unwrap();

        assert_eq!(state.config.size, 5);
        assert_eq!(state.side_to_move, Color::Black);
        // ply = (5-1)*2 + 1 = 9
        assert_eq!(state.ply, 9);

        // Check specific squares.
        // Row 4, col 1: white wall
        let sq = Square::from_rc(4, 1);
        let stack = state.board.get(sq);
        assert_eq!(stack.height, 1);
        assert_eq!(stack.top, Some(Piece::WhiteWall));

        // Row 4, col 2: stack 121S (white flat, black flat, white wall)
        let sq = Square::from_rc(4, 2);
        let stack = state.board.get(sq);
        assert_eq!(stack.height, 3);
        assert_eq!(stack.top, Some(Piece::WhiteWall));
        assert_eq!(stack.below.len(), 2);
        // below[0] = just below top = black
        assert_eq!(stack.below[0], Color::Black);
        // below[1] = bottom = white
        assert_eq!(stack.below[1], Color::White);

        // Row 4, col 4: black capstone
        let sq = Square::from_rc(4, 4);
        let stack = state.board.get(sq);
        assert_eq!(stack.height, 1);
        assert_eq!(stack.top, Some(Piece::BlackCap));

        // Row 2, col 0: black flat
        let sq = Square::from_rc(2, 0);
        let stack = state.board.get(sq);
        assert_eq!(stack.height, 1);
        assert_eq!(stack.top, Some(Piece::BlackFlat));

        // Row 0, col 1: stack 12 (white flat bottom, black flat top)
        let sq = Square::from_rc(0, 1);
        let stack = state.board.get(sq);
        assert_eq!(stack.height, 2);
        assert_eq!(stack.top, Some(Piece::BlackFlat));
        assert_eq!(stack.below[0], Color::White);

        // Verify round-trip.
        let serialized = to_tps(&state);
        assert_eq!(serialized, tps);

        // Verify reserves are reduced correctly.
        // Pieces on board:
        //   White stones (flat/wall): 1S(r4c1) + 1(r4c2 bottom) + 1S(r4c2 top)
        //     + 1(r2c4) + 1(r0c0) + 1(r0c1 bottom) = 6 white stones used
        //   White caps: 0
        //   Black stones: 2(r4c2 middle) + 2(r2c0) + 2(r0c1 top) + 2(r0c3) = 4 black stones used
        //   Black caps: 2C(r4c4) = 1 black cap used
        // Standard 5x5: 21 stones, 1 cap each.
        assert_eq!(state.reserves[0], 15, "white stones"); // 21 - 6
        assert_eq!(state.reserves[1], 1, "white caps");    // 1 - 0
        assert_eq!(state.reserves[2], 17, "black stones"); // 21 - 4
        assert_eq!(state.reserves[3], 0, "black caps");    // 1 - 1
    }

    /// Verify that from_tps produces a consistent zobrist hash.
    #[test]
    fn zobrist_consistency() {
        let tps = "x3,12,x/x,2S,x3/x5/x5/x,1,x3 2 3";
        let state = from_tps(tps).unwrap();

        // Recompute the hash independently and compare.
        let expected = zobrist::compute_full(
            &state.board,
            state.config.size,
            state.side_to_move,
            &state.reserves,
        );
        assert_eq!(state.zobrist, expected);
    }

    /// Black to move at move 1 means ply = 1.
    #[test]
    fn ply_calculation() {
        let tps = "x5/x5/x5/x5/x5 2 1";
        let state = from_tps(tps).unwrap();
        assert_eq!(state.ply, 1);
        assert_eq!(state.side_to_move, Color::Black);

        let tps = "x5/x5/x5/x5/x5 1 2";
        let state = from_tps(tps).unwrap();
        assert_eq!(state.ply, 2);
        assert_eq!(state.side_to_move, Color::White);

        let tps = "x5/x5/x5/x5/x5 2 2";
        let state = from_tps(tps).unwrap();
        assert_eq!(state.ply, 3);
        assert_eq!(state.side_to_move, Color::Black);
    }

    /// A 6x6 position round-trips correctly.
    #[test]
    fn roundtrip_6x6() {
        let tps = "x6/x2,1,x3/x6/x3,2,x2/x6/x6 1 2";
        let state = from_tps(tps).unwrap();
        assert_eq!(state.config.size, 6);
        let serialized = to_tps(&state);
        assert_eq!(serialized, tps);
    }

    /// Verify to_tps on a programmatically constructed GameState.
    #[test]
    fn to_tps_new_game() {
        let config = GameConfig::standard(5);
        let state = GameState::new(config);
        let tps = to_tps(&state);
        assert_eq!(tps, "x5/x5/x5/x5/x5 1 1");
    }

    // -------------------------------------------------------------------
    // AC 1.8: TPS round-trip for 100+ positions
    // -------------------------------------------------------------------

    /// Generate 100+ positions by random play on various board sizes and
    /// verify `to_tps(from_tps(tps)) == tps` for each.
    #[test]
    fn tps_roundtrip_100_positions() {
        use rand::SeedableRng;
        use rand::Rng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        use crate::state::GameResult;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut total_positions = 0;

        for size in 3..=7u8 {
            let config = GameConfig::standard(size);
            // Run multiple games per size.
            let games = if size <= 4 { 8 } else { 5 };

            for game_idx in 0..games {
                let mut state = GameState::new(config);
                let max_plies = match size {
                    3 => 18,
                    4 => 32,
                    5 => 40,
                    _ => 50,
                };

                // Snapshot the initial position.
                let tps = to_tps(&state);
                let recovered = from_tps(&tps).unwrap_or_else(|e| {
                    panic!("from_tps failed on initial TPS '{}': {}", tps, e)
                });
                let re_serialized = to_tps(&recovered);
                assert_eq!(
                    tps, re_serialized,
                    "roundtrip failed for initial position (size={} game={})",
                    size, game_idx
                );
                total_positions += 1;

                // Play random moves and snapshot after each.
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

                    let tps = to_tps(&state);
                    let recovered = from_tps(&tps).unwrap_or_else(|e| {
                        panic!("from_tps failed on TPS '{}': {}", tps, e)
                    });
                    let re_serialized = to_tps(&recovered);
                    assert_eq!(
                        tps, re_serialized,
                        "roundtrip failed at ply {} (size={} game={}): {}",
                        state.ply, size, game_idx, tps
                    );
                    total_positions += 1;
                }
            }
        }

        assert!(
            total_positions >= 100,
            "expected 100+ TPS round-trip positions, got {}",
            total_positions
        );
    }
}
