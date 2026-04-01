//! Move ordering for alpha-beta search.
//!
//! Order: TT move → tactical moves → killer moves → history-ordered quiet moves.

use tak_core::board::Square;
use tak_core::moves::{Move, MoveList};
use tak_core::piece::{Color, PieceType};
use tak_core::state::GameState;

use crate::eval::Score;

/// Killer move table: stores 2 killer moves per ply.
pub struct KillerTable {
    killers: Vec<[Option<Move>; 2]>,
}

impl KillerTable {
    pub fn new(max_ply: usize) -> Self {
        KillerTable {
            killers: vec![[None; 2]; max_ply],
        }
    }

    /// Record a killer move at the given ply (a quiet move that caused a beta cutoff).
    pub fn store(&mut self, ply: usize, mv: Move) {
        if ply >= self.killers.len() {
            return;
        }
        let slot = &mut self.killers[ply];
        if slot[0] == Some(mv) {
            return;
        }
        slot[1] = slot[0];
        slot[0] = Some(mv);
    }

    /// Check if a move is a killer at this ply.
    pub fn is_killer(&self, ply: usize, mv: &Move) -> bool {
        if ply >= self.killers.len() {
            return false;
        }
        self.killers[ply][0].as_ref() == Some(mv) || self.killers[ply][1].as_ref() == Some(mv)
    }

    pub fn clear(&mut self) {
        for slot in &mut self.killers {
            *slot = [None; 2];
        }
    }
}

/// History heuristic table: indexed by (color, from_square, to_square_or_type).
/// For simplicity, we index by move hash.
pub struct HistoryTable {
    table: Vec<i32>,
    mask: usize,
}

impl Default for HistoryTable {
    fn default() -> Self {
        Self::new()
    }
}

impl HistoryTable {
    pub fn new() -> Self {
        let size = 1 << 14; // 16K entries
        HistoryTable {
            table: vec![0; size],
            mask: size - 1,
        }
    }

    fn index(&self, mv: &Move, side: Color) -> usize {
        let h = move_hash(mv, side);
        (h as usize) & self.mask
    }

    /// Record a history bonus for a move that caused a cutoff.
    pub fn record_cutoff(&mut self, mv: &Move, side: Color, depth: u8) {
        let idx = self.index(mv, side);
        let bonus = (depth as i32) * (depth as i32);
        self.table[idx] = (self.table[idx] + bonus).min(10_000);
    }

    /// Record a history penalty for a move that didn't cause a cutoff.
    pub fn record_fail(&mut self, mv: &Move, side: Color, depth: u8) {
        let idx = self.index(mv, side);
        let penalty = (depth as i32) * (depth as i32);
        self.table[idx] = (self.table[idx] - penalty).max(-10_000);
    }

    /// Get the history score for a move.
    pub fn score(&self, mv: &Move, side: Color) -> i32 {
        let idx = self.index(mv, side);
        self.table[idx]
    }

    pub fn clear(&mut self) {
        self.table.fill(0);
    }

    /// Age the table (halve all values) between searches.
    pub fn age(&mut self) {
        for v in &mut self.table {
            *v /= 2;
        }
    }
}

/// Simple hash of a move for indexing.
fn move_hash(mv: &Move, side: Color) -> u32 {
    let mut h = side as u32;
    match mv {
        Move::Place { square, piece_type } => {
            h = h.wrapping_mul(31).wrapping_add(square.0 as u32);
            h = h.wrapping_mul(31).wrapping_add(*piece_type as u32);
        }
        Move::Spread {
            src,
            dir,
            pickup,
            template,
        } => {
            h = h.wrapping_mul(31).wrapping_add(src.0 as u32);
            h = h.wrapping_mul(31).wrapping_add(*dir as u32);
            h = h.wrapping_mul(31).wrapping_add(*pickup as u32);
            h = h.wrapping_mul(31).wrapping_add(template.0 as u32);
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Move scoring and sorting
// ---------------------------------------------------------------------------

/// Score all legal moves for ordering. Higher = search first.
pub fn score_moves(
    moves: &MoveList,
    state: &GameState,
    tt_move: Option<Move>,
    killers: &KillerTable,
    history: &HistoryTable,
    ply: usize,
) -> Vec<(usize, Score)> {
    let side = state.side_to_move;
    let size = state.config.size;
    let board = &state.board;

    let mut scored: Vec<(usize, Score)> = moves
        .iter()
        .enumerate()
        .map(|(i, mv)| {
            let mut s: Score = 0;

            // TT move gets highest priority.
            if tt_move.as_ref() == Some(mv) {
                return (i, 1_000_000);
            }

            match mv {
                Move::Place { square, piece_type } => {
                    // Capstone placements score higher.
                    s += match piece_type {
                        PieceType::Cap => 500,
                        PieceType::Wall => 200,
                        PieceType::Flat => 100,
                    };
                    // Center placement bonus.
                    let r = square.row();
                    let c = square.col();
                    s += center_bonus(r, c, size);
                }
                Move::Spread {
                    src,
                    dir,
                    pickup,
                    template,
                } => {
                    // Stack moves: bonus for capturing opponent's top pieces.
                    let dst = spread_destination(*src, *dir, *pickup, *template, state);
                    if let Some(dst_sq) = dst {
                        let dst_stack = board.get(dst_sq);
                        if let Some(top) = dst_stack.top {
                            if top.color() != side {
                                s += 600; // Capturing an opponent stack.
                                if top.is_wall() {
                                    s += 400; // Flattening a wall with cap.
                                }
                            }
                        }
                    }
                    s += (*pickup as Score) * 10; // Slight bias toward bigger pickups.
                }
            }

            // Killer bonus.
            if killers.is_killer(ply, mv) {
                s += 5000;
            }

            // History score.
            s += history.score(mv, side);

            (i, s)
        })
        .collect();

    // Sort descending by score.
    scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored
}

/// Compute the final destination square of a spread move.
fn spread_destination(
    src: Square,
    dir: tak_core::moves::Direction,
    _pickup: u8,
    template: tak_core::templates::DropTemplateId,
    state: &GameState,
) -> Option<Square> {
    let seq = state.templates.get_sequence(template);
    let travel = seq.drops.len() as u8;
    let (dr, dc) = dir.delta();
    let r = src.row() as i8 + dr * travel as i8;
    let c = src.col() as i8 + dc * travel as i8;
    if r >= 0 && r < state.config.size as i8 && c >= 0 && c < state.config.size as i8 {
        Some(Square::from_rc(r as u8, c as u8))
    } else {
        None
    }
}

fn center_bonus(r: u8, c: u8, size: u8) -> Score {
    let half = size as i32 / 2;
    let dr = (r as i32 - half).abs().min((r as i32 - (half - 1)).abs());
    let dc = (c as i32 - half).abs().min((c as i32 - (half - 1)).abs());
    ((half * 2 - dr - dc).max(0)) * 5
}
