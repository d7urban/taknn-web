//! Heuristic position evaluation for Tak.
//!
//! Returns a score in centipawns from the side-to-move perspective.
//! Positive = good for the side to move.

use tak_core::board::{Board, Square};
use tak_core::piece::{Color, PieceType};
use tak_core::state::GameState;

/// Score in centipawns. Positive = good for side to move.
pub type Score = i32;

pub const SCORE_INF: Score = 30_000;
pub const SCORE_MATE: Score = 29_000;
pub const SCORE_FLAT_WIN: Score = 28_000;

/// Trait for position evaluation.
pub trait Evaluator {
    /// Evaluate a leaf position. Returns score from side-to-move perspective.
    fn evaluate(&self, state: &GameState) -> Score;
}

// ---------------------------------------------------------------------------
// Heuristic evaluation weights
// ---------------------------------------------------------------------------

const W_FLAT_COUNT: Score = 200;
const W_TOP_FLAT: Score = 100;
const W_CENTER: Score = 30;
const W_ROAD_GROUP: Score = 40;
const W_WALL: Score = 50;
const W_WALL_CENTER: Score = 20;
const W_CAP: Score = 150;
const W_CAP_CENTER: Score = 30;
const W_STACK_CONTROL: Score = 15;
const W_RESERVE: Score = 3;
const W_HARD_CAP_FLAT: Score = 60;

/// Heuristic evaluator for Checkpoint 2.
pub struct HeuristicEval;

impl Evaluator for HeuristicEval {
    fn evaluate(&self, state: &GameState) -> Score {
        let score = eval_for_white(state);
        if state.side_to_move == Color::White {
            score
        } else {
            -score
        }
    }
}

/// Compute the absolute evaluation from White's perspective.
fn eval_for_white(state: &GameState) -> Score {
    let size = state.config.size;
    let board = &state.board;
    let mut score: Score = 0;

    // Per-square evaluation.
    let mut white_flats = 0i32;
    let mut black_flats = 0i32;

    for r in 0..size {
        for c in 0..size {
            let sq = Square::from_rc(r, c);
            let stack = board.get(sq);
            let top = match stack.top {
                Some(p) => p,
                None => continue,
            };

            let sign: Score = if top.color() == Color::White { 1 } else { -1 };
            let center_bonus = center_weight(r, c, size);

            match top.piece_type() {
                PieceType::Flat => {
                    // Top flat: counts for road and flat win.
                    if top.color() == Color::White {
                        white_flats += 1;
                    } else {
                        black_flats += 1;
                    }
                    score += sign * W_TOP_FLAT;
                    score += sign * center_bonus * W_CENTER / 4;
                }
                PieceType::Wall => {
                    // Walls block roads and stacks. Good for defense.
                    score += sign * W_WALL;
                    score += sign * center_bonus * W_WALL_CENTER / 4;
                }
                PieceType::Cap => {
                    // Capstones are powerful: block, can flatten walls.
                    score += sign * W_CAP;
                    score += sign * center_bonus * W_CAP_CENTER / 4;
                }
            }

            // Stack control: bonus for friendly pieces beneath the top.
            if stack.height > 1 {
                let controlled = controlled_depth(stack, top.color());
                score += sign * controlled * W_STACK_CONTROL;
            }
        }
    }

    // Flat count difference (the key material metric in Tak).
    score += (white_flats - black_flats) * W_FLAT_COUNT;

    // Hard cap on flat advantage to avoid overvaluing flat leads
    // when the position is tactically dangerous.
    let flat_diff = white_flats - black_flats;
    if flat_diff.abs() > 3 {
        score += if flat_diff > 0 {
            W_HARD_CAP_FLAT
        } else {
            -W_HARD_CAP_FLAT
        };
    }

    // Road connectivity: bonus for longest connected group touching edges.
    let (w_road, b_road) = road_connectivity(board, size);
    score += (w_road as Score - b_road as Score) * W_ROAD_GROUP;

    // Reserve pressure.
    let w_reserves = state.reserves[0] as Score + state.reserves[1] as Score;
    let b_reserves = state.reserves[2] as Score + state.reserves[3] as Score;
    score += (w_reserves - b_reserves) * W_RESERVE;

    score
}

/// Returns a center weight [0..4] — higher for more central squares.
fn center_weight(r: u8, c: u8, size: u8) -> Score {
    let half = size as Score / 2;
    let dr = (r as Score - half).abs().min((r as Score - (half - 1)).abs());
    let dc = (c as Score - half).abs().min((c as Score - (half - 1)).abs());
    let max_dist = half;
    let dist = dr + dc;
    (max_dist * 2 - dist).max(0)
}

/// Count how many pieces below the top belong to the same color.
fn controlled_depth(stack: &tak_core::board::Stack, top_color: Color) -> Score {
    let mut count: Score = 0;
    for &color in stack.below.iter() {
        if color == top_color {
            count += 1;
        } else {
            break;
        }
    }
    count
}

/// Road connectivity heuristic: for each color, compute the longest
/// connected group size of road-eligible pieces (flats + caps).
/// Returns (white_best, black_best).
fn road_connectivity(board: &Board, size: u8) -> (u8, u8) {
    // Union-Find over the NxN grid.
    let n = size as usize;
    let mut parent = [0u8; 64];
    let mut rank = [0u8; 64];
    let mut group_size = [1u8; 64];
    for i in 0..64 {
        parent[i] = i as u8;
    }

    let idx = |r: usize, c: usize| -> usize { r * 8 + c };

    fn find(parent: &mut [u8; 64], x: usize) -> usize {
        let mut x = x;
        while parent[x] as usize != x {
            parent[x] = parent[parent[x] as usize];
            x = parent[x] as usize;
        }
        x
    }

    fn union(parent: &mut [u8; 64], rank: &mut [u8; 64], group_size: &mut [u8; 64], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return;
        }
        if rank[ra] < rank[rb] {
            parent[ra] = rb as u8;
            group_size[rb] = group_size[rb].saturating_add(group_size[ra]);
        } else if rank[ra] > rank[rb] {
            parent[rb] = ra as u8;
            group_size[ra] = group_size[ra].saturating_add(group_size[rb]);
        } else {
            parent[rb] = ra as u8;
            group_size[ra] = group_size[ra].saturating_add(group_size[rb]);
            rank[ra] += 1;
        }
    }

    // Compute per-color.
    let mut white_best = 0u8;
    let mut black_best = 0u8;

    for color in [Color::White, Color::Black] {
        // Reset UF.
        for i in 0..64 {
            parent[i] = i as u8;
            rank[i] = 0;
            group_size[i] = 1;
        }

        for r in 0..n {
            for c in 0..n {
                let sq = Square::from_rc(r as u8, c as u8);
                let top = match board.get(sq).top {
                    Some(p) if p.color() == color && !p.is_wall() => p,
                    _ => continue,
                };
                let _ = top;
                let i = idx(r, c);

                // Check right and down neighbors.
                if c + 1 < n {
                    let nsq = Square::from_rc(r as u8, (c + 1) as u8);
                    if let Some(np) = board.get(nsq).top {
                        if np.color() == color && !np.is_wall() {
                            union(&mut parent, &mut rank, &mut group_size, i, idx(r, c + 1));
                        }
                    }
                }
                if r + 1 < n {
                    let nsq = Square::from_rc((r + 1) as u8, c as u8);
                    if let Some(np) = board.get(nsq).top {
                        if np.color() == color && !np.is_wall() {
                            union(&mut parent, &mut rank, &mut group_size, i, idx(r + 1, c));
                        }
                    }
                }
            }
        }

        let best = &mut if color == Color::White {
            &mut white_best
        } else {
            &mut black_best
        };

        // Find max group size among road-eligible squares.
        for r in 0..n {
            for c in 0..n {
                let sq = Square::from_rc(r as u8, c as u8);
                if let Some(p) = board.get(sq).top {
                    if p.color() == color && !p.is_wall() {
                        let root = find(&mut parent, idx(r, c));
                        let s = group_size[root];
                        if s > **best {
                            **best = s;
                        }
                    }
                }
            }
        }
    }

    (white_best, black_best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tak_core::piece::Piece;
    use tak_core::rules::GameConfig;

    #[test]
    fn eval_symmetric_at_start() {
        let state = GameState::new(GameConfig::standard(5));
        let eval = HeuristicEval;
        let score = eval.evaluate(&state);
        // Empty board should be roughly symmetric.
        assert_eq!(score, 0, "empty board should evaluate to 0");
    }

    #[test]
    fn eval_prefers_more_friendly_flats() {
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);
        // Skip opening.
        state.ply = 2;
        state.side_to_move = Color::White;

        // Place some white flats.
        state.board.get_mut(Square::from_rc(0, 0)).push(Piece::new(Color::White, PieceType::Flat));
        state.board.get_mut(Square::from_rc(0, 1)).push(Piece::new(Color::White, PieceType::Flat));
        state.board.get_mut(Square::from_rc(0, 2)).push(Piece::new(Color::White, PieceType::Flat));
        state.reserves[0] -= 3;

        // Place one black flat.
        state.board.get_mut(Square::from_rc(4, 4)).push(Piece::new(Color::Black, PieceType::Flat));
        state.reserves[2] -= 1;

        let eval = HeuristicEval;
        let score = eval.evaluate(&state);
        assert!(score > 0, "white should have advantage with more flats, got {}", score);
    }

    #[test]
    fn eval_road_connectivity_bonus() {
        let config = GameConfig::standard(5);
        let mut state = GameState::new(config);
        state.ply = 2;
        state.side_to_move = Color::White;

        // Connected line of 4 white flats.
        for c in 0..4 {
            state.board.get_mut(Square::from_rc(2, c)).push(Piece::new(Color::White, PieceType::Flat));
            state.reserves[0] -= 1;
        }
        // 4 scattered black flats (not connected).
        state.board.get_mut(Square::from_rc(0, 0)).push(Piece::new(Color::Black, PieceType::Flat));
        state.board.get_mut(Square::from_rc(1, 2)).push(Piece::new(Color::Black, PieceType::Flat));
        state.board.get_mut(Square::from_rc(3, 1)).push(Piece::new(Color::Black, PieceType::Flat));
        state.board.get_mut(Square::from_rc(4, 4)).push(Piece::new(Color::Black, PieceType::Flat));
        state.reserves[2] -= 4;

        let eval = HeuristicEval;
        let score = eval.evaluate(&state);
        // White has 4 connected vs black has 4 scattered — white should have road bonus.
        assert!(score > 0, "white should have road connectivity advantage, got {}", score);
    }
}
