//! Tactical detection for Tak positions.
//!
//! Used for training-data thinning: positions classified as Tactical are kept
//! at every ply, SemiTactical every 2 plies, and Quiet every 4 plies.

use crate::board::Square;
use crate::moves::{Move, MoveGen};
use crate::piece::Color;
use crate::state::{check_road, GameState};

// ---------------------------------------------------------------------------
// TacticalPhase
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TacticalPhase {
    /// Keep every ply.
    Tactical,
    /// Keep every 2 plies.
    SemiTactical,
    /// Keep every 4 plies.
    Quiet,
}

// ---------------------------------------------------------------------------
// TacticalFlags
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TacticalFlags {
    /// White can complete a road in one move.
    pub road_in_1_white: bool,
    /// Black can complete a road in one move.
    pub road_in_1_black: bool,
    /// The side to move has at most 2 legal moves that block an opponent road-in-1.
    pub forced_defense: bool,
    /// Any legal spread move for either side involves a capstone flattening a wall.
    pub capstone_flatten: bool,
    /// The position is in an endgame: empty squares <= 2*size or any player's
    /// total reserves (stones + caps) <= size.
    pub endgame: bool,
}

impl TacticalFlags {
    /// Compute tactical flags for the given game state.
    ///
    /// This function clones the state internally for road-in-1 probing so the
    /// caller's state is never mutated. Performance is acceptable because this
    /// is used only at training-data generation time, not during search.
    pub fn compute(state: &GameState) -> Self {
        let size = state.config.size;

        // -- Road in 1 -------------------------------------------------------
        let road_in_1_white = Self::has_road_in_1(state, Color::White);
        let road_in_1_black = Self::has_road_in_1(state, Color::Black);

        // -- Forced defense ---------------------------------------------------
        // If the opponent (non-side-to-move) has road-in-1, count how many of
        // the side-to-move's legal moves block ALL opponent road-in-1 threats.
        let forced_defense = {
            let stm = state.side_to_move;
            let opponent = stm.opposite();
            let opp_has_road_in_1 = match opponent {
                Color::White => road_in_1_white,
                Color::Black => road_in_1_black,
            };

            if opp_has_road_in_1 {
                let stm_moves = MoveGen::legal_moves_for(
                    &state.board,
                    &state.config,
                    stm,
                    state.ply,
                    &state.reserves,
                    &state.templates,
                );

                let mut blocking_count = 0u32;
                for mv in &stm_moves {
                    let mut clone = state.clone();
                    let undo = clone.apply_move(*mv);
                    // After stm's move, does the opponent still have road-in-1?
                    let still_has = Self::has_road_in_1(&clone, opponent);
                    clone.undo_move(*mv, &undo);
                    if !still_has {
                        blocking_count += 1;
                    }
                }
                blocking_count <= 2
            } else {
                false
            }
        };

        // -- Capstone flatten -------------------------------------------------
        let capstone_flatten = Self::any_capstone_flatten(state, Color::White)
            || Self::any_capstone_flatten(state, Color::Black);

        // -- Endgame ----------------------------------------------------------
        let empty = state.board.empty_count(size);
        let white_total = state.reserves[0] as u32 + state.reserves[1] as u32;
        let black_total = state.reserves[2] as u32 + state.reserves[3] as u32;
        let endgame =
            empty <= 2 * size as u32 || white_total <= size as u32 || black_total <= size as u32;

        TacticalFlags {
            road_in_1_white,
            road_in_1_black,
            forced_defense,
            capstone_flatten,
            endgame,
        }
    }

    /// Classify the tactical phase from the flags.
    pub fn phase(&self) -> TacticalPhase {
        if self.road_in_1_white
            || self.road_in_1_black
            || self.forced_defense
            || self.capstone_flatten
        {
            TacticalPhase::Tactical
        } else if self.endgame {
            TacticalPhase::SemiTactical
        } else {
            TacticalPhase::Quiet
        }
    }

    // -- Helpers --------------------------------------------------------------

    /// Check whether `color` can complete a road in a single move from the
    /// current position. We generate all legal moves for `color` (as if it
    /// were that color's turn), apply each one to a clone, and check for a
    /// road by that color.
    pub fn has_road_in_1(state: &GameState, color: Color) -> bool {
        let moves = MoveGen::legal_moves_for(
            &state.board,
            &state.config,
            color,
            state.ply,
            &state.reserves,
            &state.templates,
        );

        for mv in &moves {
            let mut clone = state.clone();
            // Force the clone's side_to_move so apply_move works correctly.
            clone.side_to_move = color;
            let undo = clone.apply_move(*mv);
            let has_road = check_road(&clone.board, state.config.size, color);
            clone.undo_move(*mv, &undo);
            if has_road {
                return true;
            }
        }

        false
    }

    /// Check whether `color` has any legal spread move that involves a capstone
    /// flattening a wall. We look at spread moves where the source top is a
    /// capstone and the final target square has a wall with a single-piece drop.
    fn any_capstone_flatten(state: &GameState, color: Color) -> bool {
        let size = state.config.size;
        let board = &state.board;

        let moves = MoveGen::legal_moves_for(
            board,
            &state.config,
            color,
            state.ply,
            &state.reserves,
            &state.templates,
        );

        for mv in &moves {
            if let Move::Spread {
                src,
                dir,
                pickup: _,
                template,
            } = mv
            {
                // The source top must be a capstone for a flatten to occur.
                match board.get(*src).top {
                    Some(p) if p.is_cap() && p.color() == color => {}
                    _ => continue,
                };

                // Walk along the direction to find the final target square.
                let seq = state.templates.get_sequence(*template);
                let num_steps = seq.drops.len();
                let last_drop = seq.drops[num_steps - 1];

                let (dr, dc) = dir.delta();
                let final_r = src.row() as i8 + dr * num_steps as i8;
                let final_c = src.col() as i8 + dc * num_steps as i8;

                if final_r < 0
                    || final_r >= size as i8
                    || final_c < 0
                    || final_c >= size as i8
                {
                    continue;
                }

                let final_sq = Square::from_rc(final_r as u8, final_c as u8);
                if let Some(target_top) = board.get(final_sq).top {
                    if target_top.is_wall() && last_drop == 1 {
                        return true;
                    }
                }
            }
        }

        false
    }
}

// ---------------------------------------------------------------------------
// SpatialLabels — per-square labels for auxiliary head training
// ---------------------------------------------------------------------------

/// Per-square labels for the auxiliary neural network heads.
/// All arrays are indexed as `[row * 8 + col]` with 8x8 padding.
#[derive(Clone, Debug)]
pub struct SpatialLabels {
    /// `road_threat[c][sq]` = 1.0 if color `c` (0=White, 1=Black) has a legal
    /// move that creates a road win and involves square `sq` (as src or path).
    pub road_threat: [[f32; 64]; 2],
    /// `block_threat[c][sq]` = 1.0 if a legal move by color `c` landing on `sq`
    /// blocks an opponent road-in-1.
    pub block_threat: [[f32; 64]; 2],
    /// `cap_flatten[sq]` = 1.0 if any legal capstone-flatten spread has source `sq`.
    pub cap_flatten: [f32; 64],
    /// 1.0 if the position is in endgame (few empty squares or low reserves).
    pub endgame: f32,
}

impl SpatialLabels {
    /// Compute spatial labels for the given game state.
    pub fn compute(state: &GameState) -> Self {
        let size = state.config.size;

        let mut road_threat = [[0.0f32; 64]; 2];
        let mut block_threat = [[0.0f32; 64]; 2];
        let mut cap_flatten = [0.0f32; 64];

        // --- Road threat: mark squares involved in road-winning moves ---
        for &color in &[Color::White, Color::Black] {
            let ci = color as usize;
            let moves = MoveGen::legal_moves_for(
                &state.board,
                &state.config,
                color,
                state.ply,
                &state.reserves,
                &state.templates,
            );

            for mv in &moves {
                let mut clone = state.clone();
                clone.side_to_move = color;
                let undo = clone.apply_move(*mv);
                let wins = check_road(&clone.board, size, color);
                clone.undo_move(*mv, &undo);

                if wins {
                    // Mark source and destination/path squares
                    match mv {
                        Move::Place { square, .. } => {
                            road_threat[ci][square.0 as usize] = 1.0;
                        }
                        Move::Spread { src, dir, template, .. } => {
                            road_threat[ci][src.0 as usize] = 1.0;
                            let seq = state.templates.get_sequence(*template);
                            let (dr, dc) = dir.delta();
                            let mut r = src.row() as i8;
                            let mut c = src.col() as i8;
                            for _ in 0..seq.drops.len() {
                                r += dr;
                                c += dc;
                                let sq = Square::from_rc(r as u8, c as u8);
                                road_threat[ci][sq.0 as usize] = 1.0;
                            }
                        }
                    }
                }
            }
        }

        // --- Block threat: mark squares where STM can block opponent road-in-1 ---
        for &color in &[Color::White, Color::Black] {
            let ci = color as usize;
            let opponent = color.opposite();

            // Check if opponent has road-in-1
            if !TacticalFlags::has_road_in_1(state, opponent) {
                continue;
            }

            let stm_moves = MoveGen::legal_moves_for(
                &state.board,
                &state.config,
                color,
                state.ply,
                &state.reserves,
                &state.templates,
            );

            for mv in &stm_moves {
                let mut clone = state.clone();
                clone.side_to_move = color;
                let undo = clone.apply_move(*mv);
                let still_has = TacticalFlags::has_road_in_1(&clone, opponent);
                clone.undo_move(*mv, &undo);

                if !still_has {
                    // This move blocks the opponent — mark the destination square
                    match mv {
                        Move::Place { square, .. } => {
                            block_threat[ci][square.0 as usize] = 1.0;
                        }
                        Move::Spread { src, dir, template, .. } => {
                            // Mark the final destination of the spread
                            let seq = state.templates.get_sequence(*template);
                            let (dr, dc) = dir.delta();
                            let final_r = src.row() as i8 + dr * seq.drops.len() as i8;
                            let final_c = src.col() as i8 + dc * seq.drops.len() as i8;
                            let dst = Square::from_rc(final_r as u8, final_c as u8);
                            block_threat[ci][dst.0 as usize] = 1.0;
                            // Also mark source (stack was moved from here)
                            block_threat[ci][src.0 as usize] = 1.0;
                        }
                    }
                }
            }
        }

        // --- Capstone flatten: mark source squares of cap-flatten spreads ---
        for &color in &[Color::White, Color::Black] {
            let moves = MoveGen::legal_moves_for(
                &state.board,
                &state.config,
                color,
                state.ply,
                &state.reserves,
                &state.templates,
            );

            for mv in &moves {
                if let Move::Spread { src, dir, template, .. } = mv {
                    // Check if source top is a capstone of this color
                    match state.board.get(*src).top {
                        Some(p) if p.is_cap() && p.color() == color => {}
                        _ => continue,
                    };

                    let seq = state.templates.get_sequence(*template);
                    let num_steps = seq.drops.len();
                    let last_drop = seq.drops[num_steps - 1];

                    let (dr, dc) = dir.delta();
                    let final_r = src.row() as i8 + dr * num_steps as i8;
                    let final_c = src.col() as i8 + dc * num_steps as i8;

                    if final_r < 0 || final_r >= size as i8
                        || final_c < 0 || final_c >= size as i8
                    {
                        continue;
                    }

                    let final_sq = Square::from_rc(final_r as u8, final_c as u8);
                    if let Some(target_top) = state.board.get(final_sq).top {
                        if target_top.is_wall() && last_drop == 1 {
                            cap_flatten[src.0 as usize] = 1.0;
                        }
                    }
                }
            }
        }

        // --- Endgame ---
        let empty = state.board.empty_count(size);
        let white_total = state.reserves[0] as u32 + state.reserves[1] as u32;
        let black_total = state.reserves[2] as u32 + state.reserves[3] as u32;
        let endgame = if empty <= 2 * size as u32
            || white_total <= size as u32
            || black_total <= size as u32
        {
            1.0
        } else {
            0.0
        };

        SpatialLabels {
            road_threat,
            block_threat,
            cap_flatten,
            endgame,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, Square};
    use crate::piece::{Color, Piece, PieceType};
    use crate::rules::GameConfig;
    use crate::state::GameState;

    /// Place a flat on the board at (r, c).
    fn place_flat(board: &mut Board, r: u8, c: u8, color: Color) {
        board
            .get_mut(Square::from_rc(r, c))
            .push(Piece::new(color, PieceType::Flat));
    }

    /// Place a wall on the board at (r, c).
    #[allow(dead_code)]
    fn place_wall(board: &mut Board, r: u8, c: u8, color: Color) {
        board
            .get_mut(Square::from_rc(r, c))
            .push(Piece::new(color, PieceType::Wall));
    }

    /// Place a capstone on the board at (r, c).
    #[allow(dead_code)]
    fn place_cap(board: &mut Board, r: u8, c: u8, color: Color) {
        board
            .get_mut(Square::from_rc(r, c))
            .push(Piece::new(color, PieceType::Cap));
    }

    /// Build a GameState with a custom board setup. The board modifications are
    /// applied after the opening phase, so `ply` starts at 2 and reserves are
    /// decremented according to the pieces placed.
    fn build_state_with(
        size: u8,
        setup: impl FnOnce(&mut Board) -> (u8, u8, u8, u8),
    ) -> GameState {
        let config = GameConfig::standard(size);
        let mut state = GameState::new(config);
        // Skip opening phase.
        state.ply = 2;
        let (w_stones_used, w_caps_used, b_stones_used, b_caps_used) = setup(&mut state.board);
        state.reserves[0] = config.stones.saturating_sub(w_stones_used);
        state.reserves[1] = config.capstones.saturating_sub(w_caps_used);
        state.reserves[2] = config.stones.saturating_sub(b_stones_used);
        state.reserves[3] = config.capstones.saturating_sub(b_caps_used);
        state
    }

    // -----------------------------------------------------------------------
    // 1. Quiet position has no tactical flags
    // -----------------------------------------------------------------------

    #[test]
    fn quiet_position_no_flags() {
        // 5x5 board with a few pieces, no roads close, plenty of reserves.
        let state = build_state_with(5, |board| {
            place_flat(board, 0, 0, Color::White);
            place_flat(board, 4, 4, Color::Black);
            // 1 white stone used, 0 white caps, 1 black stone, 0 black caps
            (1, 0, 1, 0)
        });

        let flags = TacticalFlags::compute(&state);
        assert!(!flags.road_in_1_white, "no road-in-1 for white");
        assert!(!flags.road_in_1_black, "no road-in-1 for black");
        assert!(!flags.forced_defense, "no forced defense");
        assert!(!flags.capstone_flatten, "no capstone flatten");
        assert!(!flags.endgame, "not endgame");
        assert_eq!(flags.phase(), TacticalPhase::Quiet);
    }

    // -----------------------------------------------------------------------
    // 2. Road-in-1 detection
    // -----------------------------------------------------------------------

    #[test]
    fn road_in_1_detected() {
        // 5x5 board: White has flats on column 0, rows 0..4, except row 3.
        // White has a flat reserve to place at (3,0) completing a N-S road.
        let state = build_state_with(5, |board| {
            for r in 0..5u8 {
                if r != 3 {
                    place_flat(board, r, 0, Color::White);
                }
            }
            // Also place a black flat somewhere so the board isn't trivial.
            place_flat(board, 2, 2, Color::Black);
            (4, 0, 1, 0)
        });

        let flags = TacticalFlags::compute(&state);
        assert!(flags.road_in_1_white, "white should have road-in-1");
        assert!(!flags.road_in_1_black, "black should not have road-in-1");
        assert_eq!(flags.phase(), TacticalPhase::Tactical);
    }

    // -----------------------------------------------------------------------
    // 3. Endgame detection
    // -----------------------------------------------------------------------

    #[test]
    fn endgame_few_empty_squares() {
        // 5x5 board: fill almost all squares so empty_count <= 2*5 = 10.
        // We need at most 10 empty squares out of 25. Let's fill 16, leaving 9.
        let state = build_state_with(5, |board| {
            let mut w = 0u8;
            let mut b = 0u8;
            for r in 0..4u8 {
                for c in 0..4u8 {
                    if (r + c) % 2 == 0 {
                        place_flat(board, r, c, Color::White);
                        w += 1;
                    } else {
                        place_flat(board, r, c, Color::Black);
                        b += 1;
                    }
                }
            }
            (w, 0, b, 0)
        });

        let flags = TacticalFlags::compute(&state);
        assert!(flags.endgame, "should be endgame with few empty squares");
        // No road-in-1 from this checkerboard pattern.
        if !flags.road_in_1_white
            && !flags.road_in_1_black
            && !flags.forced_defense
            && !flags.capstone_flatten
        {
            assert_eq!(flags.phase(), TacticalPhase::SemiTactical);
        }
    }

    #[test]
    fn endgame_low_reserves() {
        // 5x5 board: White has almost no reserves left.
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);
        state.ply = 2;
        // Place a couple pieces on the board.
        place_flat(&mut state.board, 0, 0, Color::White);
        place_flat(&mut state.board, 1, 1, Color::Black);
        // Set white reserves very low: stones=3, caps=1 => total=4 <= size=5.
        state.reserves[0] = 3;
        state.reserves[1] = 1;
        state.reserves[2] = config.stones;
        state.reserves[3] = config.capstones;

        let flags = TacticalFlags::compute(&state);
        assert!(flags.endgame, "white total reserves (4) <= size (5)");
    }

    // -----------------------------------------------------------------------
    // 4. Phase classification
    // -----------------------------------------------------------------------

    #[test]
    fn phase_tactical_from_road_in_1() {
        let flags = TacticalFlags {
            road_in_1_white: true,
            road_in_1_black: false,
            forced_defense: false,
            capstone_flatten: false,
            endgame: false,
        };
        assert_eq!(flags.phase(), TacticalPhase::Tactical);
    }

    #[test]
    fn phase_tactical_from_capstone_flatten() {
        let flags = TacticalFlags {
            road_in_1_white: false,
            road_in_1_black: false,
            forced_defense: false,
            capstone_flatten: true,
            endgame: false,
        };
        assert_eq!(flags.phase(), TacticalPhase::Tactical);
    }

    #[test]
    fn phase_tactical_from_forced_defense() {
        let flags = TacticalFlags {
            road_in_1_white: false,
            road_in_1_black: false,
            forced_defense: true,
            capstone_flatten: false,
            endgame: false,
        };
        assert_eq!(flags.phase(), TacticalPhase::Tactical);
    }

    #[test]
    fn phase_semi_tactical_from_endgame() {
        let flags = TacticalFlags {
            road_in_1_white: false,
            road_in_1_black: false,
            forced_defense: false,
            capstone_flatten: false,
            endgame: true,
        };
        assert_eq!(flags.phase(), TacticalPhase::SemiTactical);
    }

    #[test]
    fn phase_quiet_when_nothing() {
        let flags = TacticalFlags {
            road_in_1_white: false,
            road_in_1_black: false,
            forced_defense: false,
            capstone_flatten: false,
            endgame: false,
        };
        assert_eq!(flags.phase(), TacticalPhase::Quiet);
    }

    #[test]
    fn phase_tactical_trumps_endgame() {
        // When both tactical and endgame flags are set, Tactical wins.
        let flags = TacticalFlags {
            road_in_1_white: true,
            road_in_1_black: false,
            forced_defense: false,
            capstone_flatten: false,
            endgame: true,
        };
        assert_eq!(flags.phase(), TacticalPhase::Tactical);
    }
}
