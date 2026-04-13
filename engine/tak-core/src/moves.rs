use crate::board::{Board, Square};
use crate::piece::{Color, PieceType};
use crate::rules::GameConfig;
use crate::templates::{DropTemplateId, TemplateTable};

// ---------------------------------------------------------------------------
// Direction
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Direction {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
}

impl Direction {
    pub const ALL: [Direction; 4] = [
        Direction::North,
        Direction::East,
        Direction::South,
        Direction::West,
    ];

    /// Row/column delta for this direction.
    /// North = (-1, 0), East = (0, 1), South = (1, 0), West = (0, -1).
    #[inline]
    pub fn delta(self) -> (i8, i8) {
        match self {
            Direction::North => (-1, 0),
            Direction::East => (0, 1),
            Direction::South => (1, 0),
            Direction::West => (0, -1),
        }
    }
}

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Hash, Debug)]
pub enum Move {
    Place {
        square: Square,
        piece_type: PieceType,
    },
    Spread {
        src: Square,
        dir: Direction,
        pickup: u8,
        template: DropTemplateId,
    },
}

/// Move list. Grammar-maximum action count is 32,704 on 8x8.
/// Real positions will typically have 50-500 legal moves on 6x6.
pub type MoveList = Vec<Move>;

// ---------------------------------------------------------------------------
// MoveGen
// ---------------------------------------------------------------------------

pub struct MoveGen;

impl MoveGen {
    /// Generate all legal moves for the position.
    ///
    /// * `board` - the board state
    /// * `config` - game config (size, carry_limit, etc.)
    /// * `side_to_move` - whose turn
    /// * `ply` - current ply (for opening rule)
    /// * `reserves` - `[white_stones, white_caps, black_stones, black_caps]`
    /// * `templates` - precomputed template table for this board size
    pub fn legal_moves(
        board: &Board,
        config: &GameConfig,
        side_to_move: Color,
        ply: u16,
        reserves: &[u8; 4],
        templates: &TemplateTable,
    ) -> MoveList {
        Self::legal_moves_for(board, config, side_to_move, ply, reserves, templates)
    }

    /// Generate legal moves for a specific color.
    ///
    /// Generates moves as if it were that color's turn, regardless of whose
    /// actual turn it is. Useful for tactical detection.
    pub fn legal_moves_for(
        board: &Board,
        config: &GameConfig,
        color: Color,
        ply: u16,
        reserves: &[u8; 4],
        templates: &TemplateTable,
    ) -> MoveList {
        let size = config.size;
        let mut moves = MoveList::new();

        // --- Placements (always come first) ---
        Self::generate_placements(board, size, color, ply, reserves, &mut moves);

        // --- Movements (only after opening phase) ---
        if ply >= 2 {
            Self::generate_movements(board, config, color, templates, &mut moves);
        }

        moves
    }

    // -----------------------------------------------------------------------
    // Placement generation
    // -----------------------------------------------------------------------

    fn generate_placements(
        board: &Board,
        size: u8,
        color: Color,
        ply: u16,
        reserves: &[u8; 4],
        moves: &mut MoveList,
    ) {
        let opening = ply < 2;

        if opening {
            // Opening rule: only flat placements on empty squares.
            // Ply 0: White places a Black flat; ply 1: Black places a White flat.
            // The Move itself just records PieceType::Flat; the state module
            // handles the color swap.
            for r in 0..size {
                for c in 0..size {
                    let sq = Square::from_rc(r, c);
                    if board.get(sq).is_empty() {
                        moves.push(Move::Place {
                            square: sq,
                            piece_type: PieceType::Flat,
                        });
                    }
                }
            }
        } else {
            // Normal play: determine what the side has in reserves.
            let (stones, caps) = match color {
                Color::White => (reserves[0], reserves[1]),
                Color::Black => (reserves[2], reserves[3]),
            };

            // Enumerate empty squares in row-major order.
            // For each square, enumerate piece types in order: Flat(0), Wall(1), Cap(2).
            for r in 0..size {
                for c in 0..size {
                    let sq = Square::from_rc(r, c);
                    if board.get(sq).is_empty() {
                        if stones > 0 {
                            moves.push(Move::Place {
                                square: sq,
                                piece_type: PieceType::Flat,
                            });
                            moves.push(Move::Place {
                                square: sq,
                                piece_type: PieceType::Wall,
                            });
                        }
                        if caps > 0 {
                            moves.push(Move::Place {
                                square: sq,
                                piece_type: PieceType::Cap,
                            });
                        }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Movement generation
    // -----------------------------------------------------------------------

    fn generate_movements(
        board: &Board,
        config: &GameConfig,
        color: Color,
        templates: &TemplateTable,
        moves: &mut MoveList,
    ) {
        let size = config.size;
        let carry_limit = config.carry_limit;

        // Iterate over source squares in row-major order.
        for r in 0..size {
            for c in 0..size {
                let sq = Square::from_rc(r, c);
                let stack = board.get(sq);

                // Can only move stacks the side controls (top piece is their color).
                let top_piece = match stack.top {
                    Some(p) if p.color() == color => p,
                    _ => continue,
                };

                let stack_height = stack.height;
                let top_is_cap = top_piece.is_cap();

                // For each direction (N=0, E=1, S=2, W=3):
                for &dir in &Direction::ALL {
                    // Compute distance to edge.
                    let d = Self::distance_to_edge(r, c, dir, size);
                    if d == 0 {
                        continue;
                    }

                    let (dr, dc) = dir.delta();

                    // For each pickup count k = 1..=min(carry_limit, stack_height):
                    let max_pickup = carry_limit.min(stack_height);
                    for k in 1..=max_pickup {
                        // For each travel length t = 1..=min(k, d):
                        let max_travel = k.min(d);
                        for t in 1..=max_travel {
                            // Look up the template range from the template table.
                            let range = templates.lookup_range(k, t);

                            // For each template in the range:
                            for idx in 0..range.count {
                                let template_id = DropTemplateId(range.base_id + idx);
                                let seq = templates.get_sequence(template_id);

                                // Check legality of this spread.
                                if Self::is_spread_legal(
                                    board, r, c, dr, dc, &seq.drops, top_is_cap,
                                ) {
                                    moves.push(Move::Spread {
                                        src: sq,
                                        dir,
                                        pickup: k,
                                        template: template_id,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Compute the distance from (r, c) to the board edge in the given
    /// direction. This is how many squares can be traversed before going
    /// off-board.
    #[inline]
    fn distance_to_edge(r: u8, c: u8, dir: Direction, size: u8) -> u8 {
        match dir {
            Direction::North => r,
            Direction::South => size - 1 - r,
            Direction::West => c,
            Direction::East => size - 1 - c,
        }
    }

    /// Check whether a spread with the given drop sequence is legal.
    ///
    /// `drops` is the sequence `[d_1, ..., d_t]` where `d_i >= 1` and their
    /// sum equals the pickup count. Pieces are dropped from the bottom of the
    /// carried sub-stack first. At the last step, the final piece(s) dropped
    /// include the original top piece of the source stack.
    ///
    /// Legality rules for each target square along the ray:
    ///
    /// - **Empty or flat-topped**: always legal to enter.
    /// - **Capstone-topped**: always blocks (movement is illegal).
    /// - **Wall-topped**: blocks movement, EXCEPT on the **last step** if:
    ///   1. Exactly 1 piece is dropped (`d_t == 1`), AND
    ///   2. The piece being dropped is a capstone (the original top of the
    ///      source stack, which is the last piece remaining in the carried
    ///      sub-stack).
    ///      This is the "capstone flattens wall" rule.
    fn is_spread_legal(
        board: &Board,
        src_r: u8,
        src_c: u8,
        dr: i8,
        dc: i8,
        drops: &[u8],
        top_is_cap: bool,
    ) -> bool {
        let t = drops.len();
        let mut cr = src_r as i8;
        let mut cc = src_c as i8;

        for (step, &drop_count) in drops.iter().enumerate() {
            cr += dr;
            cc += dc;
            let target_sq = Square::from_rc(cr as u8, cc as u8);
            let target_stack = board.get(target_sq);

            let is_last = step == t - 1;

            match target_stack.top {
                None => {
                    // Empty square: always fine.
                }
                Some(piece) => {
                    if piece.is_cap() {
                        // Capstone on target: always blocks.
                        return false;
                    }
                    if piece.is_wall() {
                        // Wall on target: only allowed on the last step via
                        // the capstone flatten rule.
                        if is_last && drop_count == 1 && top_is_cap {
                            // Capstone flattens wall: legal.
                        } else {
                            return false;
                        }
                    }
                    // Flat on target: always fine.
                }
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Grammar actions
// ---------------------------------------------------------------------------

/// Grammar-maximum action enumeration.
///
/// Returns all structurally valid moves for a given board size. This includes
/// ALL placements (`N*N` squares x 3 piece types = `N*N*3`) and ALL movements
/// (sum over squares and directions of templates). Does NOT depend on any game
/// position.
pub fn grammar_actions(size: u8) -> Vec<Move> {
    let templates = TemplateTable::build(size);
    let carry_limit = size;
    let mut actions = Vec::new();

    // 1. Placements: every square x Flat/Wall/Cap, row-major then piece-type order.
    for r in 0..size {
        for c in 0..size {
            let sq = Square::from_rc(r, c);
            for &pt in &[PieceType::Flat, PieceType::Wall, PieceType::Cap] {
                actions.push(Move::Place {
                    square: sq,
                    piece_type: pt,
                });
            }
        }
    }

    // 2. Movements: every square x every direction x every pickup x every
    //    template where travel stays on board.
    for r in 0..size {
        for c in 0..size {
            let sq = Square::from_rc(r, c);

            for &dir in &Direction::ALL {
                let d = MoveGen::distance_to_edge(r, c, dir, size);
                if d == 0 {
                    continue;
                }

                for k in 1..=carry_limit {
                    let max_travel = k.min(d);
                    for t in 1..=max_travel {
                        let range = templates.lookup_range(k, t);
                        for idx in 0..range.count {
                            let template_id = DropTemplateId(range.base_id + idx);
                            actions.push(Move::Spread {
                                src: sq,
                                dir,
                                pickup: k,
                                template: template_id,
                            });
                        }
                    }
                }
            }
        }
    }

    actions
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
    use crate::templates::TemplateTable;

    // -----------------------------------------------------------------------
    // grammar_actions counts
    // -----------------------------------------------------------------------

    #[test]
    fn grammar_actions_3x3() {
        assert_eq!(grammar_actions(3).len(), 135);
    }

    #[test]
    fn grammar_actions_4x4() {
        assert_eq!(grammar_actions(4).len(), 496);
    }

    #[test]
    fn grammar_actions_5x5() {
        assert_eq!(grammar_actions(5).len(), 1575);
    }

    #[test]
    fn grammar_actions_6x6() {
        assert_eq!(grammar_actions(6).len(), 4572);
    }

    #[test]
    fn grammar_actions_7x7() {
        assert_eq!(grammar_actions(7).len(), 12495);
    }

    #[test]
    fn grammar_actions_8x8() {
        assert_eq!(grammar_actions(8).len(), 32704);
    }

    // -----------------------------------------------------------------------
    // Opening move generation
    // -----------------------------------------------------------------------

    /// On a fresh game (ply=0), legal_moves returns only opponent flat
    /// placements: N*N moves for each board size.
    #[test]
    fn opening_ply0_fresh_board() {
        for size in 3..=8u8 {
            let config = GameConfig::standard(size);
            let board = Board::empty();
            let templates = TemplateTable::build(size);
            let reserves = [
                config.stones,
                config.capstones,
                config.stones,
                config.capstones,
            ];

            let moves =
                MoveGen::legal_moves(&board, &config, Color::White, 0, &reserves, &templates);

            let n2 = (size as usize) * (size as usize);
            assert_eq!(
                moves.len(),
                n2,
                "ply 0 should produce {} moves for size {}, got {}",
                n2,
                size,
                moves.len()
            );

            // All moves should be flat placements.
            for m in &moves {
                match m {
                    Move::Place { piece_type, .. } => {
                        assert_eq!(*piece_type, PieceType::Flat);
                    }
                    _ => panic!("expected only flat placements in opening"),
                }
            }
        }
    }

    /// On ply 1, Black also only gets flat placements on empty squares.
    #[test]
    fn opening_ply1_one_piece_placed() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // Simulate ply 0: White placed a Black flat at (0,0).
        board
            .get_mut(Square::from_rc(0, 0))
            .push(Piece::new(Color::Black, PieceType::Flat));

        let reserves = [
            config.stones,
            config.capstones,
            config.stones - 1,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::Black, 1, &reserves, &templates);

        // 25 - 1 = 24 empty squares.
        assert_eq!(moves.len(), 24);
        for m in &moves {
            match m {
                Move::Place { piece_type, .. } => {
                    assert_eq!(*piece_type, PieceType::Flat);
                }
                _ => panic!("expected only flat placements in opening"),
            }
        }
    }

    /// No movements are generated during the opening phase.
    #[test]
    fn opening_no_movements() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // Place a white flat at (2,2) — still ply 0.
        board
            .get_mut(Square::from_rc(2, 2))
            .push(Piece::new(Color::White, PieceType::Flat));

        let reserves = [
            config.stones - 1,
            config.capstones,
            config.stones,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 0, &reserves, &templates);

        // Even though White has a piece on the board, no movements at ply 0.
        for m in &moves {
            assert!(matches!(m, Move::Place { .. }), "no movements in opening");
        }
    }

    // -----------------------------------------------------------------------
    // Normal placement generation
    // -----------------------------------------------------------------------

    #[test]
    fn normal_placements_with_reserves() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let board = Board::empty();
        let templates = TemplateTable::build(size);

        // After opening (ply >= 2), empty board, White has stones and caps.
        let reserves = [
            config.stones,
            config.capstones,
            config.stones,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // 25 empty squares x (Flat + Wall + Cap) = 75 placements.
        // No movements since no pieces on board.
        assert_eq!(moves.len(), 75);
    }

    #[test]
    fn normal_placements_no_caps() {
        let size = 3u8;
        let config = GameConfig::standard(size);
        let board = Board::empty();
        let templates = TemplateTable::build(size);

        // 3x3 has 0 capstones.
        assert_eq!(config.capstones, 0);
        let reserves = [
            config.stones,
            config.capstones,
            config.stones,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // 9 empty squares x (Flat + Wall) = 18 placements, no caps.
        assert_eq!(moves.len(), 18);
    }

    #[test]
    fn normal_placements_no_stones_left() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let board = Board::empty();
        let templates = TemplateTable::build(size);

        // White has 0 stones, 1 cap.
        let reserves = [0, 1, config.stones, config.capstones];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // 25 empty squares x Cap only = 25 placements.
        assert_eq!(moves.len(), 25);
    }

    // -----------------------------------------------------------------------
    // Placement ordering
    // -----------------------------------------------------------------------

    #[test]
    fn placements_are_sorted_row_major_then_piece_type() {
        let size = 4u8;
        let config = GameConfig::standard(size);
        let board = Board::empty();
        let templates = TemplateTable::build(size);

        // 4x4 has 0 capstones, so only Flat and Wall.
        let reserves = [
            config.stones,
            config.capstones,
            config.stones,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // Check ordering: for each consecutive pair of placements, the ordering
        // should be (square index ASC, piece_type ASC).
        let placements: Vec<_> = moves
            .iter()
            .filter_map(|m| match m {
                Move::Place { square, piece_type } => Some((*square, *piece_type)),
                _ => None,
            })
            .collect();

        for i in 1..placements.len() {
            let (sq_a, pt_a) = placements[i - 1];
            let (sq_b, pt_b) = placements[i];
            let key_a = (sq_a.0, pt_a as u8);
            let key_b = (sq_b.0, pt_b as u8);
            assert!(
                key_a < key_b,
                "placements not sorted: ({:?},{:?}) >= ({:?},{:?})",
                sq_a,
                pt_a,
                sq_b,
                pt_b,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Movement ordering
    // -----------------------------------------------------------------------

    #[test]
    fn movements_sorted_by_src_dir_pickup_template() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // Place white flats in a column so we have multiple stacks.
        for r in 0..3u8 {
            board
                .get_mut(Square::from_rc(r, 2))
                .push(Piece::new(Color::White, PieceType::Flat));
        }

        let reserves = [
            config.stones - 3,
            config.capstones,
            config.stones,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        let movements: Vec<_> = moves
            .iter()
            .filter_map(|m| match m {
                Move::Spread {
                    src,
                    dir,
                    pickup,
                    template,
                } => Some((*src, *dir, *pickup, *template)),
                _ => None,
            })
            .collect();

        for i in 1..movements.len() {
            let (sq_a, dir_a, k_a, t_a) = movements[i - 1];
            let (sq_b, dir_b, k_b, t_b) = movements[i];
            let key_a = (sq_a.0, dir_a as u8, k_a, t_a.0);
            let key_b = (sq_b.0, dir_b as u8, k_b, t_b.0);
            assert!(
                key_a < key_b,
                "movements not sorted at index {}: {:?} >= {:?}",
                i,
                key_a,
                key_b,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Capstone flatten legality
    // -----------------------------------------------------------------------

    #[test]
    fn capstone_can_flatten_wall_on_last_step() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // Place a white capstone at (2,2).
        board
            .get_mut(Square::from_rc(2, 2))
            .push(Piece::new(Color::White, PieceType::Cap));

        // Place a black wall at (2,3) — one step East.
        board
            .get_mut(Square::from_rc(2, 3))
            .push(Piece::new(Color::Black, PieceType::Wall));

        let reserves = [
            config.stones,
            config.capstones - 1,
            config.stones - 1,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // There should be a spread: pick up 1 from (2,2), go East, drop [1].
        // This is the capstone-flattens-wall move.
        let flatten_move = moves.iter().find(|m| {
            matches!(
                m,
                Move::Spread {
                    src,
                    dir: Direction::East,
                    pickup: 1,
                    ..
                } if src.row() == 2 && src.col() == 2
            )
        });
        assert!(
            flatten_move.is_some(),
            "capstone should be able to flatten wall going East"
        );
    }

    #[test]
    fn capstone_cannot_flatten_wall_with_multiple_drops() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // Stack: white flat on bottom, white cap on top at (2,2).
        board
            .get_mut(Square::from_rc(2, 2))
            .push(Piece::new(Color::White, PieceType::Flat));
        board
            .get_mut(Square::from_rc(2, 2))
            .push(Piece::new(Color::White, PieceType::Cap));

        // Black wall at (2,3) — one step East.
        board
            .get_mut(Square::from_rc(2, 3))
            .push(Piece::new(Color::Black, PieceType::Wall));

        let reserves = [
            config.stones - 1,
            config.capstones - 1,
            config.stones - 1,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // Spread East picking up 2, travel 1 => drop [2] on (2,3) which has a
        // wall. Since drop_count = 2 (not 1), capstone flatten does NOT apply.
        let bad_move = moves.iter().find(|m| {
            matches!(
                m,
                Move::Spread {
                    src,
                    dir: Direction::East,
                    pickup: 2,
                    template,
                    ..
                } if src.row() == 2 && src.col() == 2 && {
                    let seq = templates.get_sequence(*template);
                    seq.drops.len() == 1 && seq.drops[0] == 2
                }
            )
        });
        assert!(
            bad_move.is_none(),
            "cannot flatten wall when dropping 2 pieces on it"
        );

        // However: spread East picking up 1, travel 1 => drop [1] on (2,3)
        // which has a wall. Pickup=1 means the piece being dropped is the cap
        // (top of source), and drop_count=1 on the last step. This IS legal.
        let flatten_move = moves.iter().find(|m| {
            matches!(
                m,
                Move::Spread {
                    src,
                    dir: Direction::East,
                    pickup: 1,
                    ..
                } if src.row() == 2 && src.col() == 2
            )
        });
        assert!(flatten_move.is_some(), "pickup 1 cap can flatten wall");
    }

    #[test]
    fn flat_cannot_flatten_wall() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // White flat at (2,2).
        board
            .get_mut(Square::from_rc(2, 2))
            .push(Piece::new(Color::White, PieceType::Flat));

        // Black wall at (2,3).
        board
            .get_mut(Square::from_rc(2, 3))
            .push(Piece::new(Color::Black, PieceType::Wall));

        let reserves = [
            config.stones - 1,
            config.capstones,
            config.stones - 1,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // No spread East from (2,2) should exist (flat cannot flatten wall).
        let east_spread = moves.iter().find(|m| {
            matches!(
                m,
                Move::Spread {
                    src,
                    dir: Direction::East,
                    ..
                } if src.row() == 2 && src.col() == 2
            )
        });
        assert!(
            east_spread.is_none(),
            "flat cannot enter a wall-occupied square"
        );
    }

    #[test]
    fn nothing_can_enter_capstone_square() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // White cap at (2,2).
        board
            .get_mut(Square::from_rc(2, 2))
            .push(Piece::new(Color::White, PieceType::Cap));

        // Black cap at (2,3).
        board
            .get_mut(Square::from_rc(2, 3))
            .push(Piece::new(Color::Black, PieceType::Cap));

        let reserves = [
            config.stones,
            config.capstones - 1,
            config.stones,
            config.capstones - 1,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // No spread East from (2,2): target has a capstone.
        let east_spread = moves.iter().find(|m| {
            matches!(
                m,
                Move::Spread {
                    src,
                    dir: Direction::East,
                    ..
                } if src.row() == 2 && src.col() == 2
            )
        });
        assert!(
            east_spread.is_none(),
            "cannot enter a capstone-occupied square"
        );
    }

    // -----------------------------------------------------------------------
    // Carry limit enforcement
    // -----------------------------------------------------------------------

    #[test]
    fn carry_limit_enforced() {
        let size = 3u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // Build a tall stack at (1,1): 5 white flats, then a white cap on top.
        // (3x3 has 0 capstones in standard config, but we can still construct
        // the board state manually for testing carry limit.)
        for _ in 0..5 {
            board
                .get_mut(Square::from_rc(1, 1))
                .push(Piece::new(Color::White, PieceType::Flat));
        }
        board
            .get_mut(Square::from_rc(1, 1))
            .push(Piece::new(Color::White, PieceType::Cap));

        assert_eq!(board.get(Square::from_rc(1, 1)).height, 6);

        // Carry limit for 3x3 is 3.
        assert_eq!(config.carry_limit, 3);

        let reserves = [config.stones - 5, 0, config.stones, config.capstones];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // All spread moves from (1,1) should have pickup <= 3.
        let spreads_from_center: Vec<_> = moves
            .iter()
            .filter_map(|m| match m {
                Move::Spread { src, pickup, .. } if src.row() == 1 && src.col() == 1 => {
                    Some(*pickup)
                }
                _ => None,
            })
            .collect();

        assert!(
            !spreads_from_center.is_empty(),
            "should have some spreads from center"
        );
        for &k in &spreads_from_center {
            assert!(k <= 3, "pickup {} exceeds carry limit 3", k);
        }
        // Should also have pickup=3 (max carry limit).
        assert!(
            spreads_from_center.contains(&3),
            "should have pickup=3 (carry limit)"
        );
    }

    // -----------------------------------------------------------------------
    // Placements come before movements
    // -----------------------------------------------------------------------

    #[test]
    fn placements_before_movements() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // Place a white flat so we get some movements.
        board
            .get_mut(Square::from_rc(2, 2))
            .push(Piece::new(Color::White, PieceType::Flat));

        let reserves = [
            config.stones - 1,
            config.capstones,
            config.stones,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        let mut seen_movement = false;
        for m in &moves {
            match m {
                Move::Place { .. } => {
                    assert!(!seen_movement, "placement appeared after movement");
                }
                Move::Spread { .. } => {
                    seen_movement = true;
                }
            }
        }
        assert!(seen_movement, "should have at least one movement");
    }

    // -----------------------------------------------------------------------
    // Edge distance / boundary correctness
    // -----------------------------------------------------------------------

    #[test]
    fn corner_piece_limited_directions() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // White flat at top-left corner (0,0).
        board
            .get_mut(Square::from_rc(0, 0))
            .push(Piece::new(Color::White, PieceType::Flat));

        let reserves = [
            config.stones - 1,
            config.capstones,
            config.stones,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        let spread_dirs: Vec<_> = moves
            .iter()
            .filter_map(|m| match m {
                Move::Spread { src, dir, .. } if src.0 == 0 => Some(*dir),
                _ => None,
            })
            .collect();

        // From (0,0), North and West have distance 0. Only South and East.
        assert!(
            !spread_dirs.contains(&Direction::North),
            "cannot go North from row 0"
        );
        assert!(
            !spread_dirs.contains(&Direction::West),
            "cannot go West from col 0"
        );
        assert!(
            spread_dirs.contains(&Direction::South),
            "should be able to go South"
        );
        assert!(
            spread_dirs.contains(&Direction::East),
            "should be able to go East"
        );
    }

    // -----------------------------------------------------------------------
    // Cannot move opponent's stacks
    // -----------------------------------------------------------------------

    #[test]
    fn cannot_move_opponent_stacks() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // Black flat at (2,2), White flat at (3,3).
        board
            .get_mut(Square::from_rc(2, 2))
            .push(Piece::new(Color::Black, PieceType::Flat));
        board
            .get_mut(Square::from_rc(3, 3))
            .push(Piece::new(Color::White, PieceType::Flat));

        let reserves = [
            config.stones - 1,
            config.capstones,
            config.stones - 1,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // White should only have spreads from (3,3), not from (2,2).
        for m in &moves {
            if let Move::Spread { src, .. } = m {
                assert_eq!(
                    (src.row(), src.col()),
                    (3, 3),
                    "White should only spread from (3,3), not {:?}",
                    src,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Intermediate walls block spread
    // -----------------------------------------------------------------------

    #[test]
    fn wall_blocks_intermediate_step() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // White cap at (2,0), black wall at (2,1), empty at (2,2).
        board
            .get_mut(Square::from_rc(2, 0))
            .push(Piece::new(Color::White, PieceType::Cap));
        board
            .get_mut(Square::from_rc(2, 1))
            .push(Piece::new(Color::Black, PieceType::Wall));

        let reserves = [
            config.stones,
            config.capstones - 1,
            config.stones - 1,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // Spread East from (2,0):
        // pickup=1, travel=1 => [1] on (2,1): wall, cap flatten => LEGAL
        // (pickup max is min(carry_limit=5, height=1) = 1, so travel max = 1.)
        // So only 1 legal East spread.
        let east_moves: Vec<_> = moves
            .iter()
            .filter(|m| {
                matches!(
                    m,
                    Move::Spread {
                        src,
                        dir: Direction::East,
                        ..
                    } if src.row() == 2 && src.col() == 0
                )
            })
            .collect();

        assert_eq!(
            east_moves.len(),
            1,
            "should have exactly 1 spread East (cap flattens wall): got {}",
            east_moves.len(),
        );
    }

    #[test]
    fn wall_blocks_intermediate_with_tall_stack() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // Build a 3-tall stack at (2,0): 2 white flats + white cap on top.
        board
            .get_mut(Square::from_rc(2, 0))
            .push(Piece::new(Color::White, PieceType::Flat));
        board
            .get_mut(Square::from_rc(2, 0))
            .push(Piece::new(Color::White, PieceType::Flat));
        board
            .get_mut(Square::from_rc(2, 0))
            .push(Piece::new(Color::White, PieceType::Cap));

        // Black wall at (2,1).
        board
            .get_mut(Square::from_rc(2, 1))
            .push(Piece::new(Color::Black, PieceType::Wall));

        let reserves = [
            config.stones - 2,
            config.capstones - 1,
            config.stones - 1,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // Spreads East from (2,0):
        // pickup=1 travel=1: [1] on (2,1) wall, cap flatten => LEGAL
        // pickup=2 travel=1: [2] on (2,1) wall, drop_count=2 => ILLEGAL
        // pickup=3 travel=1: [3] on (2,1) wall, drop_count=3 => ILLEGAL
        // Any travel >= 2: (2,1) is intermediate with wall => ILLEGAL
        //   (pickup=2 travel=2: [1,1] step 1 on wall is NOT last => ILLEGAL)
        //   (pickup=3 travel=2: [1,2] or [2,1] step 1 on wall NOT last => ILLEGAL)
        //   (pickup=3 travel=3: [1,1,1] step 1 on wall NOT last => ILLEGAL)
        //
        // So only 1 legal East spread.
        let east_moves: Vec<_> = moves
            .iter()
            .filter(|m| {
                matches!(
                    m,
                    Move::Spread {
                        src,
                        dir: Direction::East,
                        ..
                    } if src.row() == 2 && src.col() == 0
                )
            })
            .collect();

        assert_eq!(
            east_moves.len(),
            1,
            "only pickup=1 cap flatten should work: got {}",
            east_moves.len(),
        );
    }

    // -----------------------------------------------------------------------
    // Direction helpers
    // -----------------------------------------------------------------------

    #[test]
    fn direction_deltas() {
        assert_eq!(Direction::North.delta(), (-1, 0));
        assert_eq!(Direction::East.delta(), (0, 1));
        assert_eq!(Direction::South.delta(), (1, 0));
        assert_eq!(Direction::West.delta(), (0, -1));
    }

    #[test]
    fn direction_all_count() {
        assert_eq!(Direction::ALL.len(), 4);
    }

    // -----------------------------------------------------------------------
    // Capstone flatten multi-step: flatten only on final square
    // -----------------------------------------------------------------------

    #[test]
    fn capstone_flattens_wall_on_final_square_of_multi_step() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // 2-tall stack at (2,0): white flat + white cap on top.
        board
            .get_mut(Square::from_rc(2, 0))
            .push(Piece::new(Color::White, PieceType::Flat));
        board
            .get_mut(Square::from_rc(2, 0))
            .push(Piece::new(Color::White, PieceType::Cap));

        // (2,1) is empty.
        // Black wall at (2,2).
        board
            .get_mut(Square::from_rc(2, 2))
            .push(Piece::new(Color::Black, PieceType::Wall));

        let reserves = [
            config.stones - 1,
            config.capstones - 1,
            config.stones - 1,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // Spread East from (2,0) with pickup=2 travel=2 template [1,1]:
        // Step 1: drop 1 flat on (2,1) which is empty => OK.
        // Step 2 (last): drop 1 cap on (2,2) which has a wall, d_t=1, cap => LEGAL.
        let multi_step_flatten = moves.iter().find(|m| {
            if let Move::Spread {
                src,
                dir: Direction::East,
                pickup: 2,
                template,
            } = m
            {
                if src.row() == 2 && src.col() == 0 {
                    let seq = templates.get_sequence(*template);
                    return seq.drops.as_slice() == [1, 1];
                }
            }
            false
        });
        assert!(
            multi_step_flatten.is_some(),
            "capstone should flatten wall on the last step of a multi-step spread"
        );
    }

    // -----------------------------------------------------------------------
    // Stacking on friendly flats is legal
    // -----------------------------------------------------------------------

    #[test]
    fn spread_onto_friendly_flat() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // White flat at (2,2) and (2,3).
        board
            .get_mut(Square::from_rc(2, 2))
            .push(Piece::new(Color::White, PieceType::Flat));
        board
            .get_mut(Square::from_rc(2, 3))
            .push(Piece::new(Color::White, PieceType::Flat));

        let reserves = [
            config.stones - 2,
            config.capstones,
            config.stones,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // Spread East from (2,2) pickup=1 travel=1 [1]: drop on (2,3) which
        // has a friendly flat => legal.
        let stack_move = moves.iter().find(|m| {
            matches!(
                m,
                Move::Spread {
                    src,
                    dir: Direction::East,
                    pickup: 1,
                    ..
                } if src.row() == 2 && src.col() == 2
            )
        });
        assert!(
            stack_move.is_some(),
            "should be able to spread onto a friendly flat"
        );
    }

    // -----------------------------------------------------------------------
    // Stacking on enemy flats is also legal
    // -----------------------------------------------------------------------

    #[test]
    fn spread_onto_enemy_flat() {
        let size = 5u8;
        let config = GameConfig::standard(size);
        let mut board = Board::empty();
        let templates = TemplateTable::build(size);

        // White flat at (2,2), black flat at (2,3).
        board
            .get_mut(Square::from_rc(2, 2))
            .push(Piece::new(Color::White, PieceType::Flat));
        board
            .get_mut(Square::from_rc(2, 3))
            .push(Piece::new(Color::Black, PieceType::Flat));

        let reserves = [
            config.stones - 1,
            config.capstones,
            config.stones - 1,
            config.capstones,
        ];

        let moves = MoveGen::legal_moves(&board, &config, Color::White, 2, &reserves, &templates);

        // Spread East from (2,2) onto enemy flat => legal.
        let stack_move = moves.iter().find(|m| {
            matches!(
                m,
                Move::Spread {
                    src,
                    dir: Direction::East,
                    pickup: 1,
                    ..
                } if src.row() == 2 && src.col() == 2
            )
        });
        assert!(
            stack_move.is_some(),
            "should be able to spread onto an enemy flat"
        );
    }
}
