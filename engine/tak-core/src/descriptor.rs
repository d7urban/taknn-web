//! Move descriptors for the policy head neural network.

use arrayvec::ArrayVec;

use crate::board::Square;
use crate::moves::Move;
use crate::state::GameState;

/// Compact descriptor for one legal move, used as input to the policy MLP.
#[derive(Clone, Debug)]
pub struct MoveDescriptor {
    pub src: u8,
    pub dst: u8,
    pub path: ArrayVec<u8, 7>,
    pub move_type: u8,         // 0=placement, 1=spread
    pub piece_type: u8,        // 0=flat, 1=wall, 2=cap, 3=N/A (spread)
    pub direction: u8,         // 0=N, 1=E, 2=S, 3=W, 4=N/A (placement)
    pub pickup_count: u8,      // 0 (placement) or 1..=carry_limit
    pub drop_template_id: u16, // 0 = N/A (placement)
    pub travel_length: u8,     // 0 (placement) or 1..=size-1
    pub capstone_flatten: bool,
    pub enters_occupied: bool,
    pub opening_phase: bool,
}

/// Build descriptors for all legal moves in a position.
pub fn build_descriptors(state: &GameState, moves: &[Move]) -> Vec<MoveDescriptor> {
    moves.iter().map(|mv| build_one(state, *mv)).collect()
}

fn build_one(state: &GameState, mv: Move) -> MoveDescriptor {
    match mv {
        Move::Place { square, piece_type } => MoveDescriptor {
            src: square.0,
            dst: square.0,
            path: ArrayVec::new(),
            move_type: 0,
            piece_type: piece_type as u8,
            direction: 4,
            pickup_count: 0,
            drop_template_id: 0,
            travel_length: 0,
            capstone_flatten: false,
            enters_occupied: !state.board.get(square).is_empty(),
            opening_phase: state.is_opening_phase(),
        },
        Move::Spread { src, dir, pickup, template } => {
            let drops = &state.templates.get_sequence(template).drops;
            let travel_len = drops.len() as u8;

            let (dr, dc) = dir.delta();
            let mut r = src.row() as i8;
            let mut c = src.col() as i8;
            let mut path = ArrayVec::new();
            for _ in 0..travel_len {
                r += dr;
                c += dc;
                path.push(Square::from_rc(r as u8, c as u8).0);
            }
            let dst = *path.last().unwrap_or(&src.0);

            let top_is_cap = state.board.get(src).top.is_some_and(|p| p.is_cap());
            let dst_sq = Square(dst);
            let dst_has_wall = state.board.get(dst_sq).top.is_some_and(|p| p.is_wall());
            let last_drop_one = drops.last().copied() == Some(1);
            let capstone_flatten = top_is_cap && dst_has_wall && last_drop_one;
            let enters_occupied = !state.board.get(dst_sq).is_empty();

            MoveDescriptor {
                src: src.0,
                dst,
                path,
                move_type: 1,
                piece_type: 3,
                direction: dir as u8,
                pickup_count: pickup,
                drop_template_id: template.0,
                travel_length: travel_len,
                capstone_flatten,
                enters_occupied,
                opening_phase: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::PieceType;
    use crate::rules::GameConfig;

    #[test]
    fn placement_descriptor_fields() {
        let state = GameState::new(GameConfig::standard(5));
        let moves = state.legal_moves();
        let descs = build_descriptors(&state, &moves);
        assert_eq!(descs.len(), moves.len());
        for (desc, mv) in descs.iter().zip(moves.iter()) {
            assert_eq!(desc.move_type, 0);
            assert_eq!(desc.direction, 4);
            assert_eq!(desc.pickup_count, 0);
            assert_eq!(desc.travel_length, 0);
            assert!(desc.path.is_empty());
            assert!(desc.opening_phase);
            if let Move::Place { square, .. } = mv {
                assert_eq!(desc.src, square.0);
                assert_eq!(desc.dst, square.0);
            }
        }
    }

    #[test]
    fn spread_descriptor_path_and_dst() {
        let mut state = GameState::new(GameConfig::standard(5));
        state.apply_move(Move::Place { square: Square::from_rc(0, 0), piece_type: PieceType::Flat });
        state.apply_move(Move::Place { square: Square::from_rc(1, 0), piece_type: PieceType::Flat });
        state.apply_move(Move::Place { square: Square::from_rc(2, 0), piece_type: PieceType::Flat });
        state.apply_move(Move::Place { square: Square::from_rc(3, 0), piece_type: PieceType::Flat });

        let moves = state.legal_moves();
        let descs = build_descriptors(&state, &moves);
        for (desc, mv) in descs.iter().zip(moves.iter()) {
            if let Move::Spread { src, dir, .. } = mv {
                assert_eq!(desc.move_type, 1);
                assert_eq!(desc.src, src.0);
                assert_eq!(desc.direction, *dir as u8);
                assert!(desc.travel_length > 0);
                assert_eq!(desc.path.len(), desc.travel_length as usize);
                assert_eq!(desc.dst, desc.path[desc.path.len() - 1]);
            }
        }
    }

    #[test]
    fn descriptor_count_matches_moves() {
        let state = GameState::new(GameConfig::standard(6));
        let moves = state.legal_moves();
        let descs = build_descriptors(&state, &moves);
        assert_eq!(descs.len(), moves.len());
    }
}
