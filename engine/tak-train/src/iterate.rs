use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use tch::{nn, Device, Tensor};

use tak_core::moves::Move;
use tak_core::piece::Color;
use tak_core::rules::GameConfig;
use tak_core::state::{GameResult, GameState};
use tak_core::tactical::TacticalFlags;
use tak_core::tensor::{BoardTensor, C_IN};
use tak_data::shard::{ShardWriter, TrainingRecord};
use tak_search::eval::{Evaluator, Score, SCORE_FLAT_WIN, SCORE_MATE};
use tak_search::pvs::{PvsSearch, SearchConfig};

use crate::checkpoint::{self, TrainingState};
use crate::config::{NetConfig, TrainConfig};
use crate::data::{load_shards, train_val_split, ShardLoader};
use crate::net::TakNet;
use crate::trainer::Trainer;

use rand_distr::{Distribution, Gamma};

const MAX_GAME_PLIES: u16 = 200;

#[derive(Clone, Debug)]
pub struct TemperatureSchedule {
    pub noise_plies: u16,
    pub warm_plies: u16,
    pub warm_temp: f32,
    pub cool_temp: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_weight: f32,
}

#[derive(Clone, Debug)]
pub struct BatchedInferenceConfig {
    pub max_batch_size: usize,
    pub max_wait_micros: u64,
}

impl Default for BatchedInferenceConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_wait_micros: 500,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TeacherIterationConfig {
    pub primary_size: u8,
    pub iterations: usize,
    pub games_per_iteration: u32,
    pub replay_window_shards: usize,
    pub eval_games: usize,
    pub promotion_win_rate: f64,
    pub seed: u64,
    pub bootstrap_shards_dir: PathBuf,
    pub selfplay_dir: PathBuf,
    pub initial_checkpoint: Option<PathBuf>,
    pub temperature: TemperatureSchedule,
    pub inference: BatchedInferenceConfig,
    pub search_config: SearchConfig,
    pub train_config: TrainConfig,
}

#[derive(Clone, Debug)]
pub struct SelfPlayStats {
    pub games: u32,
    pub records: usize,
    pub workers: usize,
}

#[derive(Clone, Debug)]
pub struct MatchSummary {
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    pub score: f64,
}

struct SelfPlayShardConfig<'a> {
    checkpoint_path: &'a Path,
    net_config: &'a NetConfig,
    device: Device,
    board_size: u8,
    inference: &'a BatchedInferenceConfig,
    search_config: SearchConfig,
    temperature: &'a TemperatureSchedule,
    model_id: u16,
    seed: u64,
}

#[derive(Copy, Clone)]
struct SelfPlayGameConfig<'a> {
    board_size: u8,
    search_config: SearchConfig,
    temperature: &'a TemperatureSchedule,
    model_id: u16,
}

struct MatchEvalConfig<'a> {
    candidate_checkpoint: &'a Path,
    incumbent_checkpoint: &'a Path,
    net_config: &'a NetConfig,
    device: Device,
    board_size: u8,
    inference: &'a BatchedInferenceConfig,
    search_config: SearchConfig,
    games: usize,
}

struct EvalRequest {
    board_data: [f32; C_IN * 64],
    size_id: i64,
    reply: mpsc::SyncSender<Score>,
}

#[derive(Clone)]
struct BatchedNeuralEvaluator {
    tx: mpsc::Sender<EvalRequest>,
}

struct BatchedEvaluatorService {
    evaluator: Option<BatchedNeuralEvaluator>,
    worker: Option<JoinHandle<anyhow::Result<()>>>,
}

impl MatchSummary {
    pub fn win_rate(&self, games: usize) -> f64 {
        if games == 0 {
            1.0
        } else {
            self.score / games as f64
        }
    }
}

impl Evaluator for BatchedNeuralEvaluator {
    fn evaluate(&self, state: &GameState) -> Score {
        let bt = BoardTensor::encode(state);
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.tx
            .send(EvalRequest {
                board_data: bt.data,
                size_id: bt.size_id as i64,
                reply: reply_tx,
            })
            .expect("batched evaluator worker stopped while sending request");
        reply_rx
            .recv()
            .expect("batched evaluator worker stopped while waiting for reply")
    }
}

impl BatchedEvaluatorService {
    fn start(
        checkpoint_path: &Path,
        net_config: &NetConfig,
        device: Device,
        config: &BatchedInferenceConfig,
    ) -> anyhow::Result<Self> {
        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let checkpoint_path = checkpoint_path.to_path_buf();
        let net_config = net_config.clone();
        let config = config.clone();

        let worker = thread::Builder::new()
            .name("tak-batched-eval".into())
            .spawn(move || {
                batched_eval_worker(rx, ready_tx, checkpoint_path, net_config, device, config)
            })
            .context("spawn batched evaluator worker")?;

        match ready_rx
            .recv()
            .context("wait for batched evaluator startup")?
        {
            Ok(()) => Ok(Self {
                evaluator: Some(BatchedNeuralEvaluator { tx }),
                worker: Some(worker),
            }),
            Err(err) => {
                let _ = worker.join();
                Err(err)
            }
        }
    }

    fn evaluator(&self) -> BatchedNeuralEvaluator {
        self.evaluator
            .as_ref()
            .expect("batched evaluator accessed after shutdown")
            .clone()
    }
}

impl Drop for BatchedEvaluatorService {
    fn drop(&mut self) {
        let _ = self.evaluator.take();
        if let Some(worker) = self.worker.take() {
            match worker.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => eprintln!("Warning: batched evaluator worker failed: {err}"),
                Err(_) => eprintln!("Warning: batched evaluator worker panicked"),
            }
        }
    }
}

pub fn run_teacher_iteration_loop(
    config: &TeacherIterationConfig,
    device: Device,
) -> anyhow::Result<()> {
    fs::create_dir_all(&config.train_config.checkpoint_dir)?;
    fs::create_dir_all(&config.selfplay_dir)?;

    let bootstrap_shards = list_tknn_files(&config.bootstrap_shards_dir)?;

    if let Some(initial) = &config.initial_checkpoint {
        install_initial_checkpoint(
            initial,
            &config.train_config.checkpoint_dir,
            &config.train_config.net,
        )?;
        println!(
            "Installed initial teacher checkpoint from {}",
            initial.display()
        );
    }

    if checkpoint::best_checkpoint_path(&config.train_config.checkpoint_dir).is_none() {
        if bootstrap_shards.is_empty() {
            bail!(
                "no promoted checkpoint found and no bootstrap shards available in {}",
                config.bootstrap_shards_dir.display()
            );
        }

        let bootstrap_dir = config
            .train_config
            .checkpoint_dir
            .join("iterations")
            .join("bootstrap");
        println!(
            "Training bootstrap teacher from {} shard(s)",
            bootstrap_shards.len()
        );
        train_teacher_candidate(
            &config.train_config,
            device,
            &bootstrap_shards,
            None,
            &bootstrap_dir,
        )?;
        promote_candidate(&bootstrap_dir, &config.train_config.checkpoint_dir)?;
        let promoted_weights =
            checkpoint::best_checkpoint_path(&config.train_config.checkpoint_dir)
                .unwrap_or(config.train_config.checkpoint_dir.join("best.safetensors"));
        println!(
            "Promoted bootstrap teacher to {}",
            promoted_weights.display()
        );
    }

    for iteration in 1..=config.iterations {
        println!(
            "\n== Teacher Iteration {iteration}/{} ==",
            config.iterations
        );

        let current_weights = checkpoint::best_checkpoint_path(&config.train_config.checkpoint_dir)
            .unwrap_or(config.train_config.checkpoint_dir.join("best.safetensors"));

        let selfplay_t0 = Instant::now();
        let mut iteration_shards = Vec::new();

        let quotas = [
            (3, 0.02),
            (4, 0.10),
            (5, 0.25),
            (6, 0.55),
            (7, 0.05),
            (8, 0.03),
        ];

        for &(size, pct) in &quotas {
            let target_games = (config.games_per_iteration as f64 * pct).round() as u32;
            if target_games == 0 {
                continue;
            }

            let shard_path = config
                .selfplay_dir
                .join(format!("iter_{iteration:04}_{}x{}.tknn", size, size));

            let selfplay_config = SelfPlayShardConfig {
                checkpoint_path: &current_weights,
                net_config: &config.train_config.net,
                device,
                board_size: size,
                inference: &config.inference,
                search_config: config.search_config,
                temperature: &config.temperature,
                model_id: u16::try_from(iteration).unwrap_or(u16::MAX),
                seed: config
                    .seed
                    .wrapping_add(iteration as u64)
                    .wrapping_add(size as u64 * 100),
            };

            let stats = generate_selfplay_shard(&selfplay_config, &shard_path, target_games)?;
            println!(
                "  {}x{}: generated {} self-play game(s), {} record(s) with {} worker(s) -> {}",
                size,
                size,
                stats.games,
                stats.records,
                stats.workers,
                shard_path.file_name().unwrap().to_string_lossy()
            );
            iteration_shards.push(shard_path);
        }

        println!(
            "Iteration {iteration} self-play complete in {:.1}s",
            selfplay_t0.elapsed().as_secs_f64()
        );

        let replay_shards = collect_replay_shards(
            &bootstrap_shards,
            &config.selfplay_dir,
            config.replay_window_shards,
        )?;
        println!("Replay window: {} shard(s)", replay_shards.len());

        let candidate_dir = config
            .train_config
            .checkpoint_dir
            .join("iterations")
            .join(format!("iter_{iteration:04}"));
        let train_t0 = Instant::now();
        let candidate_best = train_teacher_candidate(
            &config.train_config,
            device,
            &replay_shards,
            Some(&current_weights),
            &candidate_dir,
        )?;
        println!(
            "Candidate training took {:.1}s",
            train_t0.elapsed().as_secs_f64()
        );

        if config.eval_games == 0 {
            promote_candidate(&candidate_dir, &config.train_config.checkpoint_dir)?;
            println!("Promoted candidate without match evaluation (eval_games=0)");
            continue;
        }

        let eval_t0 = Instant::now();
        let eval_config = MatchEvalConfig {
            candidate_checkpoint: &candidate_best,
            incumbent_checkpoint: &current_weights,
            net_config: &config.train_config.net,
            device,
            board_size: config.primary_size,
            inference: &config.inference,
            search_config: config.search_config,
            games: config.eval_games,
        };
        let summary = evaluate_candidate(&eval_config)?;
        let win_rate = summary.win_rate(config.eval_games);
        println!(
            "Candidate vs incumbent: +{} ={} -{} | score {:.1}/{} ({:.1}%) in {:.1}s",
            summary.wins,
            summary.draws,
            summary.losses,
            summary.score,
            config.eval_games,
            100.0 * win_rate,
            eval_t0.elapsed().as_secs_f64()
        );

        if win_rate >= config.promotion_win_rate {
            promote_candidate(&candidate_dir, &config.train_config.checkpoint_dir)?;
            println!(
                "Promoted candidate (threshold {:.1}%)",
                100.0 * config.promotion_win_rate
            );
        } else {
            println!(
                "Kept incumbent (threshold {:.1}%)",
                100.0 * config.promotion_win_rate
            );
        }
    }

    Ok(())
}

fn train_teacher_candidate(
    base_config: &TrainConfig,
    device: Device,
    shard_paths: &[PathBuf],
    init_weights: Option<&Path>,
    output_dir: &Path,
) -> anyhow::Result<PathBuf> {
    if output_dir.exists() {
        fs::remove_dir_all(output_dir)
            .with_context(|| format!("remove stale candidate dir {}", output_dir.display()))?;
    }

    let mut train_config = base_config.clone();
    train_config.checkpoint_dir = output_dir.to_path_buf();

    let records = load_shards(shard_paths);
    if records.is_empty() {
        bail!("no training records loaded from replay shards");
    }

    let (train_records, val_records) = train_val_split(records, train_config.val_split);
    if train_records.is_empty() {
        bail!("training split is empty");
    }

    println!(
        "Training candidate on {} shard(s): train {} record(s), val {} record(s)",
        shard_paths.len(),
        train_records.len(),
        val_records.len()
    );

    let mut train_loader = ShardLoader::new(train_records, train_config.batch_size);
    let mut val_loader = ShardLoader::new(val_records, train_config.batch_size);
    let mut trainer = Trainer::new(train_config.clone(), device)?;

    if let Some(path) = init_weights {
        let missing = checkpoint::load_weights(&mut trainer.vs, path)?;
        if !missing.is_empty() {
            eprintln!(
                "Warning: {} missing key(s) when loading seed weights from {}",
                missing.len(),
                path.display()
            );
        }
        println!("Seeded candidate from {}", path.display());
    }

    trainer.train(&mut train_loader, &mut val_loader);

    let best =
        checkpoint::best_checkpoint_path(output_dir).unwrap_or(output_dir.join("best.safetensors"));
    if !best.exists() {
        bail!("candidate training did not produce {}", best.display());
    }

    Ok(best)
}

fn generate_selfplay_shard(
    config: &SelfPlayShardConfig<'_>,
    output_path: &Path,
    num_games: u32,
) -> anyhow::Result<SelfPlayStats> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let worker_count = usize::max(
        1,
        usize::min(rayon::current_num_threads(), num_games as usize),
    );
    let games_per_worker = num_games.div_ceil(worker_count as u32);
    let evaluator_service = BatchedEvaluatorService::start(
        config.checkpoint_path,
        config.net_config,
        config.device,
        config.inference,
    )?;
    let evaluator = evaluator_service.evaluator();
    let game_config = SelfPlayGameConfig {
        board_size: config.board_size,
        search_config: config.search_config,
        temperature: config.temperature,
        model_id: config.model_id,
    };

    let family_counts =
        std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
    let max_per_family = std::cmp::max(1, num_games / 10); // Don't let any one family take up >10% of games

    let chunked_results: Vec<anyhow::Result<Vec<TrainingRecord>>> = (0..worker_count)
        .into_par_iter()
        .map(|worker_idx| {
            let first_game = worker_idx as u32 * games_per_worker;
            if first_game >= num_games {
                return Ok(Vec::new());
            }
            let last_game = ((worker_idx as u32 + 1) * games_per_worker).min(num_games);
            let evaluator = evaluator.clone();

            let mut records = Vec::new();
            for game_id in first_game..last_game {
                let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(game_id as u64));
                records.extend(play_selfplay_game(
                    evaluator.clone(),
                    game_config,
                    &mut rng,
                    game_id,
                    std::sync::Arc::clone(&family_counts),
                    max_per_family,
                ));
            }
            Ok(records)
        })
        .collect();

    let mut all_records = Vec::new();
    for chunk in chunked_results {
        all_records.extend(chunk?);
    }

    let tmp_path = output_path.with_extension("tknn.tmp");
    let file =
        fs::File::create(&tmp_path).with_context(|| format!("create {}", tmp_path.display()))?;
    let mut writer = ShardWriter::new(file)
        .with_context(|| format!("open shard writer for {}", tmp_path.display()))?;

    let total_records = all_records.len();
    for record in &all_records {
        writer.write_record(record)?;
    }

    writer.finish()?;
    fs::rename(&tmp_path, output_path)
        .with_context(|| format!("rename {} -> {}", tmp_path.display(), output_path.display()))?;

    Ok(SelfPlayStats {
        games: num_games,
        records: total_records,
        workers: worker_count,
    })
}

fn canonicalize_opening(moves: &[Move], size: u8) -> Vec<Move> {
    use tak_core::symmetry::D4;
    let mut best_family = None;
    for &sym in &D4::ALL {
        let mut family = Vec::with_capacity(moves.len());
        for &mv in moves {
            family.push(sym.transform_move(mv, size));
        }
        if best_family.is_none() || Some(&family) < best_family.as_ref() {
            best_family = Some(family);
        }
    }
    best_family.unwrap()
}

fn play_selfplay_game<R: Rng>(
    evaluator: BatchedNeuralEvaluator,
    config: SelfPlayGameConfig<'_>,
    rng: &mut R,
    game_id: u32,
    family_counts: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<Vec<Move>, u32>>>,
    max_per_family: u32,
) -> Vec<TrainingRecord> {
    loop {
        let mut state = GameState::new(GameConfig::standard(config.board_size));
        let mut search = PvsSearch::new(config.search_config, evaluator.clone());
        let mut history = Vec::new();
        let mut plies_so_far = Vec::new();
        let mut aborted = false;

        while state.result == GameResult::Ongoing && state.ply < MAX_GAME_PLIES {
            let temp = if state.ply < config.temperature.warm_plies {
                config.temperature.warm_temp
            } else {
                config.temperature.cool_temp
            };

            let result = search.search(&mut state);
            if result.root_scores.is_empty() {
                break;
            }

            let moves = state.legal_moves();
            let mut move_scores = vec![0i32; moves.len()];
            for rs in &result.root_scores {
                if let Some(idx) = moves.iter().position(|&mv| mv == rs.mv) {
                    move_scores[idx] = rs.score;
                }
            }

            let mut policy = softmax(&move_scores, temp);

            if state.ply < config.temperature.noise_plies && policy.len() > 1 {
                let alpha = config.temperature.dirichlet_alpha;
                let weight = config.temperature.dirichlet_weight;
                if let Ok(gamma) = Gamma::new(alpha, 1.0) {
                    let noise: Vec<f32> = (0..policy.len()).map(|_| gamma.sample(rng)).collect();
                    let sum: f32 = noise.iter().sum();
                    if sum > 0.0 {
                        for (p, n) in policy.iter_mut().zip(noise.iter()) {
                            *p = (1.0 - weight) * *p + weight * (*n / sum);
                        }
                    }
                }
            }

            let move_idx = sample_index(rng, &policy);
            let selected_move = moves[move_idx];

            let tactical_phase = TacticalFlags::phase_fast(&state);
            history.push((
                state.clone(),
                tactical_phase,
                policy,
                result.depth,
                result.nodes as u32,
                result.score,
            ));

            state.apply_move(selected_move);
            plies_so_far.push(selected_move);

            if state.ply == 4 {
                let family = canonicalize_opening(&plies_so_far, config.board_size);
                let mut counts = family_counts.lock().unwrap();
                let count = counts.entry(family).or_insert(0);
                if *count >= max_per_family {
                    aborted = true;
                    break;
                }
                *count += 1;
            }
        }

        if aborted {
            continue; // Re-roll game
        }

        let final_result = state.result;
        let final_margin = state.flat_margin();

        return history
            .into_iter()
            .map(|(position, phase, policy, depth, nodes, search_score)| {
                let sparse_policy = policy
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &prob)| {
                        if prob > 0.001 {
                            Some((idx as u16, f16::from_f32(prob)))
                        } else {
                            None
                        }
                    })
                    .collect();

                TrainingRecord {
                    board_size: position.config.size,
                    side_to_move: position.side_to_move,
                    ply: position.ply,
                    reserves: position.reserves,
                    komi: position.config.komi,
                    half_komi: position.config.half_komi,
                    game_result: final_result,
                    flat_margin: final_margin,
                    search_depth: depth,
                    search_nodes: nodes,
                    game_id,
                    source_model_id: config.model_id,
                    tactical_phase: phase,
                    teacher_wdl: score_to_wdl(search_score),
                    teacher_margin: (search_score as f32 / 500.0).clamp(-1.0, 1.0),
                    policy_target: sparse_policy,
                    board_data: TrainingRecord::pack_board(&position),
                }
            })
            .collect();
    }
}

fn evaluate_candidate(config: &MatchEvalConfig<'_>) -> anyhow::Result<MatchSummary> {
    let worker_count = usize::max(1, usize::min(rayon::current_num_threads(), config.games));
    let games_per_worker = config.games.div_ceil(worker_count);
    let candidate_service = BatchedEvaluatorService::start(
        config.candidate_checkpoint,
        config.net_config,
        config.device,
        config.inference,
    )?;
    let incumbent_service = BatchedEvaluatorService::start(
        config.incumbent_checkpoint,
        config.net_config,
        config.device,
        config.inference,
    )?;
    let candidate_eval = candidate_service.evaluator();
    let incumbent_eval = incumbent_service.evaluator();
    let partials: Vec<anyhow::Result<MatchSummary>> = (0..worker_count)
        .into_par_iter()
        .map(|worker_idx| {
            let first_game = worker_idx * games_per_worker;
            if first_game >= config.games {
                return Ok(MatchSummary {
                    wins: 0,
                    draws: 0,
                    losses: 0,
                    score: 0.0,
                });
            }
            let last_game = ((worker_idx + 1) * games_per_worker).min(config.games);
            let candidate_eval = candidate_eval.clone();
            let incumbent_eval = incumbent_eval.clone();

            let mut summary = MatchSummary {
                wins: 0,
                draws: 0,
                losses: 0,
                score: 0.0,
            };

            for game_idx in first_game..last_game {
                let candidate_is_white = game_idx % 2 == 0;
                let result = play_match(
                    candidate_eval.clone(),
                    incumbent_eval.clone(),
                    config.board_size,
                    config.search_config,
                    candidate_is_white,
                );

                summary.score += result;
                if result >= 0.99 {
                    summary.wins += 1;
                } else if result <= 0.01 {
                    summary.losses += 1;
                } else {
                    summary.draws += 1;
                }
            }

            Ok(summary)
        })
        .collect();

    let mut summary = MatchSummary {
        wins: 0,
        draws: 0,
        losses: 0,
        score: 0.0,
    };

    for partial in partials {
        let partial = partial?;
        summary.wins += partial.wins;
        summary.draws += partial.draws;
        summary.losses += partial.losses;
        summary.score += partial.score;
    }

    Ok(summary)
}

fn play_match(
    candidate_eval: BatchedNeuralEvaluator,
    incumbent_eval: BatchedNeuralEvaluator,
    board_size: u8,
    search_config: SearchConfig,
    candidate_is_white: bool,
) -> f64 {
    let mut state = GameState::new(GameConfig::standard(board_size));
    let mut candidate_search = PvsSearch::new(search_config, candidate_eval);
    let mut incumbent_search = PvsSearch::new(search_config, incumbent_eval);
    let candidate_color = if candidate_is_white {
        Color::White
    } else {
        Color::Black
    };

    while state.result == GameResult::Ongoing && state.ply < MAX_GAME_PLIES {
        let net = if state.side_to_move == candidate_color {
            &mut candidate_search
        } else {
            &mut incumbent_search
        };

        let mv = best_move_for_state(&mut state, net);
        state.apply_move(mv);
    }

    match state.result {
        GameResult::RoadWin(color) | GameResult::FlatWin(color) => {
            if color == candidate_color {
                1.0
            } else {
                0.0
            }
        }
        GameResult::Draw | GameResult::Ongoing => 0.5,
    }
}

fn best_move_for_state<E: Evaluator>(state: &mut GameState, search: &mut PvsSearch<E>) -> Move {
    let fallback = state
        .legal_moves()
        .into_iter()
        .next()
        .expect("search asked for a move in a terminal or illegal state");
    search.search(state).best_move.unwrap_or(fallback)
}

fn batched_eval_worker(
    rx: mpsc::Receiver<EvalRequest>,
    ready_tx: mpsc::SyncSender<anyhow::Result<()>>,
    checkpoint_path: PathBuf,
    net_config: NetConfig,
    device: Device,
    config: BatchedInferenceConfig,
) -> anyhow::Result<()> {
    let (_vs, net) = match load_teacher_net(&checkpoint_path, &net_config, device) {
        Ok(loaded) => {
            let _ = ready_tx.send(Ok(()));
            loaded
        }
        Err(err) => {
            let context = err.context(format!(
                "start batched evaluator from {}",
                checkpoint_path.display()
            ));
            let message = format!("{context:#}");
            let _ = ready_tx.send(Err(anyhow::anyhow!(message)));
            return Err(context);
        }
    };

    let max_batch_size = config.max_batch_size.max(1);
    let wait = Duration::from_micros(config.max_wait_micros);
    let mut pending = Vec::with_capacity(max_batch_size);

    while let Ok(first) = rx.recv() {
        pending.push(first);
        let deadline = Instant::now() + wait;

        while pending.len() < max_batch_size {
            let timeout = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(timeout) {
                Ok(req) => pending.push(req),
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        let batch_len = pending.len();
        let mut board_flat = vec![0.0f32; batch_len * C_IN * 64];
        let mut size_ids = vec![0i64; batch_len];
        for (idx, req) in pending.iter().enumerate() {
            let start = idx * C_IN * 64;
            let end = start + C_IN * 64;
            board_flat[start..end].copy_from_slice(&req.board_data);
            size_ids[idx] = req.size_id;
        }

        let board = Tensor::from_slice(&board_flat)
            .reshape([batch_len as i64, C_IN as i64, 8, 8])
            .to(device);
        let size_id = Tensor::from_slice(&size_ids).to(device);
        let out = net.infer(&board, &size_id);
        let wdl = out.wdl.to(Device::Cpu);
        let margin = out.margin.to(Device::Cpu);

        for (idx, req) in pending.drain(..).enumerate() {
            let scalar = wdl.double_value(&[idx as i64, 0]) - wdl.double_value(&[idx as i64, 2])
                + 0.5 * margin.double_value(&[idx as i64, 0]);
            let _ = req.reply.send(scalar_to_score(scalar));
        }
    }

    Ok(())
}

fn load_teacher_net(
    checkpoint_path: &Path,
    net_config: &NetConfig,
    device: Device,
) -> anyhow::Result<(nn::VarStore, TakNet)> {
    let mut vs = nn::VarStore::new(device);
    let net = TakNet::new(&vs, net_config);
    let missing = checkpoint::load_weights(&mut vs, checkpoint_path)?;
    if !missing.is_empty() {
        eprintln!(
            "Warning: {} missing key(s) when loading {}",
            missing.len(),
            checkpoint_path.display()
        );
    }
    Ok((vs, net))
}

fn install_initial_checkpoint(
    checkpoint_path: &Path,
    promoted_dir: &Path,
    net_config: &NetConfig,
) -> anyhow::Result<()> {
    let target = match checkpoint_path.extension().and_then(|ext| ext.to_str()) {
        Some("pt") => promoted_dir.join("best.pt"),
        _ => promoted_dir.join("best.safetensors"),
    };
    copy_atomic(checkpoint_path, &target)?;
    let state = TrainingState {
        epoch: 0,
        global_step: 0,
        best_val_loss: 0.0,
        net_config: Some(net_config.clone()),
        model_type: "teacher".into(),
    };
    checkpoint::save_training_state(promoted_dir, &state)?;
    Ok(())
}

fn promote_candidate(candidate_dir: &Path, promoted_dir: &Path) -> anyhow::Result<()> {
    let candidate_weights = checkpoint::best_checkpoint_path(candidate_dir)
        .unwrap_or(candidate_dir.join("best.safetensors"));
    if !candidate_weights.exists() {
        bail!(
            "candidate checkpoint missing: {}",
            candidate_weights.display()
        );
    }

    let promoted_weights = match candidate_weights.extension().and_then(|ext| ext.to_str()) {
        Some("pt") => promoted_dir.join("best.pt"),
        _ => promoted_dir.join("best.safetensors"),
    };
    copy_atomic(&candidate_weights, &promoted_weights)?;
    let state = checkpoint::load_training_state(candidate_dir);
    checkpoint::save_training_state(promoted_dir, &state)?;
    Ok(())
}

fn copy_atomic(src: &Path, dst: &Path) -> anyhow::Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = dst.with_extension("tmp");
    fs::copy(src, &tmp).with_context(|| format!("copy {} -> {}", src.display(), tmp.display()))?;
    fs::rename(&tmp, dst)
        .with_context(|| format!("rename {} -> {}", tmp.display(), dst.display()))?;
    Ok(())
}

fn collect_replay_shards(
    bootstrap_shards: &[PathBuf],
    selfplay_dir: &Path,
    replay_window_shards: usize,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut replay = bootstrap_shards.to_vec();
    let mut selfplay = list_tknn_files(selfplay_dir)?;
    if selfplay.len() > replay_window_shards {
        selfplay = selfplay[selfplay.len() - replay_window_shards..].to_vec();
    }
    replay.extend(selfplay);
    Ok(replay)
}

fn list_tknn_files(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().map(|ext| ext == "tknn").unwrap_or(false))
        .collect();
    files.sort();
    Ok(files)
}

fn scalar_to_score(scalar: f64) -> Score {
    let limit = SCORE_FLAT_WIN as f64 - 500.0;
    (scalar * 1000.0).round().clamp(-limit, limit) as Score
}

fn score_to_wdl(score: i32) -> [f16; 3] {
    if score.abs() >= SCORE_MATE - 100 {
        if score > 0 {
            [f16::from_f32(1.0), f16::from_f32(0.0), f16::from_f32(0.0)]
        } else {
            [f16::from_f32(0.0), f16::from_f32(0.0), f16::from_f32(1.0)]
        }
    } else if score.abs() >= SCORE_FLAT_WIN - 100 {
        if score > 0 {
            [f16::from_f32(0.9), f16::from_f32(0.1), f16::from_f32(0.0)]
        } else {
            [f16::from_f32(0.0), f16::from_f32(0.1), f16::from_f32(0.9)]
        }
    } else {
        let s = score as f64 / 150.0;
        let win_raw = 1.0 / (1.0 + (-s).exp());
        let loss_raw = 1.0 - win_raw;
        let draw = (0.3 * (-s * s / 2.0).exp()).max(0.01);
        let non_draw = 1.0 - draw;
        let win = (non_draw * win_raw) as f32;
        let loss = (non_draw * loss_raw) as f32;
        [
            f16::from_f32(win),
            f16::from_f32(draw as f32),
            f16::from_f32(loss),
        ]
    }
}

fn softmax(scores: &[i32], temp: f32) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let denom = (100.0f32 * temp.max(1e-3)).max(1e-3);
    let max_score = scores.iter().copied().max().unwrap_or(0) as f32;
    let mut probs: Vec<f32> = scores
        .iter()
        .map(|&score| ((score as f32 - max_score) / denom).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        let uniform = 1.0 / probs.len().max(1) as f32;
        probs.fill(uniform);
        return probs;
    }
    for prob in &mut probs {
        *prob /= sum;
    }
    probs
}

fn sample_index<R: Rng>(rng: &mut R, probs: &[f32]) -> usize {
    let r: f32 = rng.random();
    let mut cumulative = 0.0;
    for (idx, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if r < cumulative {
            return idx;
        }
    }
    probs.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_score_stays_below_terminal_band() {
        let score = scalar_to_score(1000.0);
        assert!(score < SCORE_FLAT_WIN);
        assert!(score > -SCORE_FLAT_WIN);
    }

    #[test]
    fn softmax_normalizes() {
        let probs = softmax(&[100, 0, -100], 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
