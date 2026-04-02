use tak_data::shard::{TrainingRecord, ShardWriter, ShardReader};
use tak_core::state::GameState;
use tak_core::rules::GameConfig;
use tak_core::piece::{Color, Piece, PieceType};
use tak_core::board::Square;
use half::f16;

#[test]
fn shard_roundtrip() {
    let config = GameConfig::standard(5);
    let mut state = GameState::new(config);
    state.ply = 2;
    state.side_to_move = Color::White;
    state.reserves = [19, 1, 20, 1];
    
    state.board.get_mut(Square::from_rc(0, 0)).push(Piece::new(Color::White, PieceType::Flat));
    state.board.get_mut(Square::from_rc(1, 1)).push(Piece::new(Color::Black, PieceType::Wall));
    state.board.get_mut(Square::from_rc(2, 2)).push(Piece::new(Color::White, PieceType::Cap));
    
    // Add a stack (not too deep, so buried piece order doesn't matter for zobrist)
    state.board.get_mut(Square::from_rc(3, 3)).push(Piece::new(Color::White, PieceType::Flat));
    state.board.get_mut(Square::from_rc(3, 3)).push(Piece::new(Color::Black, PieceType::Flat));
    
    // Refresh zobrist
    state.zobrist = tak_core::zobrist::compute_full(&state.board, state.config.size, state.side_to_move, &state.reserves);

    let record = TrainingRecord {
        board_size: 5,
        side_to_move: Color::White,
        ply: 2,
        reserves: [19, 1, 20, 1],
        komi: 0,
        half_komi: false,
        game_result: tak_core::state::GameResult::Ongoing,
        flat_margin: 1,
        search_depth: 4,
        search_nodes: 100,
        game_id: 42,
        source_model_id: 0,
        tactical_phase: tak_core::tactical::TacticalPhase::Quiet,
        teacher_wdl: [f16::from_f32(0.33); 3],
        teacher_margin: 0.0,
        policy_target: vec![(0, f16::from_f32(0.5)), (1, f16::from_f32(0.5))],
        board_data: TrainingRecord::pack_board(&state),
    };

    let mut buf = Vec::new();
    {
        let mut writer = ShardWriter::new(&mut buf).unwrap();
        writer.write_record(&record).unwrap();
        writer.finish().unwrap();
    }

    let mut reader = ShardReader::new(buf.as_slice()).unwrap();
    let recovered = reader.next_record().unwrap().unwrap();

    assert_eq!(recovered.board_size, record.board_size);
    assert_eq!(recovered.side_to_move, record.side_to_move);
    assert_eq!(recovered.ply, record.ply);
    assert_eq!(recovered.reserves, record.reserves);
    assert_eq!(recovered.policy_target, record.policy_target);
    assert_eq!(recovered.board_data, record.board_data);

    let recovered_state = recovered.unpack_board().unwrap();
    
    if recovered_state.zobrist != state.zobrist {
        eprintln!("Original Zobrist:  {}", state.zobrist);
        eprintln!("Recovered Zobrist: {}", recovered_state.zobrist);
        
        // Detailed comparison
        for r in 0..5 {
            for c in 0..5 {
                let sq = Square::from_rc(r, c);
                let s1 = state.board.get(sq);
                let s2 = recovered_state.board.get(sq);
                if s1.height != s2.height || s1.top != s2.top || s1.below != s2.below {
                    eprintln!("Mismatch at ({},{}):", r, c);
                    eprintln!("  Orig:      h={}, top={:?}, below={:?}", s1.height, s1.top, s1.below);
                    eprintln!("  Recovered: h={}, top={:?}, below={:?}", s2.height, s2.top, s2.below);
                }
            }
        }
    }
    
    assert_eq!(recovered_state.zobrist, state.zobrist);
}
