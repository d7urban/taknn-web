use rand::Rng;
use half::f16;
use tak_core::state::{GameResult, GameState};
use tak_core::rules::GameConfig;
use tak_core::tactical::TacticalFlags;
use tak_search::pvs::{PvsSearch, SearchConfig};
use tak_search::eval::HeuristicEval;
use crate::shard::TrainingRecord;

#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    pub board_size: u8,
    pub search_config: SearchConfig,
    pub temp_schedule: TemperatureSchedule,
}

#[derive(Clone, Debug)]
pub struct TemperatureSchedule {
    pub warm_plies: u16,
    pub warm_temp: f32,
    pub cool_temp: f32,
}

impl Default for TemperatureSchedule {
    fn default() -> Self {
        TemperatureSchedule {
            warm_plies: 8,
            warm_temp: 1.0,
            cool_temp: 0.1,
        }
    }
}

pub struct SelfPlayEngine {
    pub config: SelfPlayConfig,
}

impl SelfPlayEngine {
    pub fn new(config: SelfPlayConfig) -> Self {
        SelfPlayEngine { config }
    }

    pub fn play_game<R: Rng>(&self, rng: &mut R, game_id: u32) -> Vec<TrainingRecord> {
        let mut state = GameState::new(GameConfig::standard(self.config.board_size));
        let mut search = PvsSearch::new(self.config.search_config, HeuristicEval);
        let mut history = Vec::new();

        while state.result == GameResult::Ongoing && state.ply < 200 {
            let temp = if state.ply < self.config.temp_schedule.warm_plies {
                self.config.temp_schedule.warm_temp
            } else {
                self.config.temp_schedule.cool_temp
            };

            let moves = state.legal_moves();
            if moves.is_empty() { break; }

            let mut move_scores = Vec::with_capacity(moves.len());
            for &mv in &moves {
                let undo = state.apply_move(mv);
                let result = search.search(&mut state);
                move_scores.push(-result.score);
                state.undo_move(mv, &undo);
            }

            let policy = softmax(&move_scores, temp);
            let move_idx = sample_index(rng, &policy);
            let selected_move = moves[move_idx];

            let tactical = TacticalFlags::compute(&state);
            let record_state = state.clone();

            history.push((record_state, tactical, policy, search.config.max_depth, 0));

            state.apply_move(selected_move);
        }

        let final_result = state.result;
        let final_margin = state.flat_margin();

        history.into_iter().map(|(s, t, policy, depth, nodes)| {
            let mut sparse_policy = Vec::new();
            for (i, &p) in policy.iter().enumerate() {
                if p > 0.001 {
                    sparse_policy.push((i as u16, f16::from_f32(p)));
                }
            }

            TrainingRecord {
                board_size: s.config.size,
                side_to_move: s.side_to_move,
                ply: s.ply,
                reserves: s.reserves,
                komi: s.config.komi,
                half_komi: s.config.half_komi,
                game_result: final_result,
                flat_margin: final_margin,
                search_depth: depth,
                search_nodes: nodes,
                game_id,
                source_model_id: 0,
                tactical_phase: t.phase(),
                teacher_wdl: [f16::from_f32(0.33); 3],
                teacher_margin: 0.0,
                policy_target: sparse_policy,
                board_data: TrainingRecord::pack_board(&s),
            }
        }).collect()
    }
}

fn softmax(scores: &[i32], temp: f32) -> Vec<f32> {
    let mut probs: Vec<f32> = scores.iter().map(|&s| (s as f32 / (100.0 * temp)).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    probs
}

fn sample_index<R: Rng>(rng: &mut R, probs: &[f32]) -> usize {
    let r: f32 = rng.random();
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn play_partial_game_3x3() {
        let config = SelfPlayConfig {
            board_size: 3,
            search_config: SearchConfig {
                max_depth: 1,
                max_time_ms: 10,
                tt_size_mb: 1,
            },
            temp_schedule: TemperatureSchedule::default(),
        };
        let engine = SelfPlayEngine::new(config);
        let _rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        // Just play a few plies to verify record generation
        let mut state = GameState::new(GameConfig::standard(3));
        let mut search = PvsSearch::new(engine.config.search_config, HeuristicEval);
        
        let moves = state.legal_moves();
        let mut move_scores = Vec::with_capacity(moves.len());
        for &mv in &moves {
            let undo = state.apply_move(mv);
            let result = search.search(&mut state);
            move_scores.push(-result.score);
            state.undo_move(mv, &undo);
        }
        let policy = softmax(&move_scores, 1.0);
        
        let mut sparse_policy = Vec::new();
        for (i, &p) in policy.iter().enumerate() {
            if p > 0.001 {
                sparse_policy.push((i as u16, f16::from_f32(p)));
            }
        }
        
        let record = TrainingRecord {
            board_size: 3,
            side_to_move: state.side_to_move,
            ply: state.ply,
            reserves: state.reserves,
            komi: state.config.komi,
            half_komi: state.config.half_komi,
            game_result: GameResult::Ongoing,
            flat_margin: 0,
            search_depth: 1,
            search_nodes: 0,
            game_id: 1,
            source_model_id: 0,
            tactical_phase: tak_core::tactical::TacticalPhase::Quiet,
            teacher_wdl: [f16::from_f32(0.33); 3],
            teacher_margin: 0.0,
            policy_target: sparse_policy,
            board_data: TrainingRecord::pack_board(&state),
        };
        
        assert_eq!(record.board_size, 3);
        assert!(!record.board_data.is_empty());
    }

    #[test]
    #[ignore] // Slow test, run explicitly with --ignored
    fn play_full_game_3x3() {
        let config = SelfPlayConfig {
            board_size: 3,
            search_config: SearchConfig {
                max_depth: 2,
                max_time_ms: 100,
                tt_size_mb: 1,
            },
            temp_schedule: TemperatureSchedule::default(),
        };
        let engine = SelfPlayEngine::new(config);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let records = engine.play_game(&mut rng, 1);
        assert!(!records.is_empty());
        assert!(records.last().unwrap().game_result.is_terminal());
    }
}
