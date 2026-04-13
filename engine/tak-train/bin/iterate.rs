use std::path::PathBuf;

use clap::Parser;
use tch::Device;

use tak_search::pvs::SearchConfig;
use tak_train::config::{NetConfig, TrainConfig};
use tak_train::iterate::{
    run_teacher_iteration_loop, BatchedInferenceConfig, TeacherIterationConfig, TemperatureSchedule,
};

#[derive(Parser)]
#[command(
    name = "tak-iterate",
    about = "Iteratively improve the Tak teacher network"
)]
struct Args {
    /// Directory containing bootstrap .tknn shards.
    #[arg(long, default_value = "shards")]
    bootstrap_shards: PathBuf,

    /// Directory where teacher self-play shards will be written.
    #[arg(long, default_value = "selfplay")]
    selfplay_dir: PathBuf,

    /// Optional checkpoint to start from before iterating.
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Checkpoint directory used for promoted and per-iteration models.
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: PathBuf,

    /// Primary board size used for candidate-vs-incumbent evaluation matches.
    #[arg(long, default_value_t = 6)]
    primary_board_size: u8,

    /// Number of self-play / retrain / promote iterations to run.
    #[arg(long, default_value_t = 5)]
    iterations: usize,

    /// Self-play games generated per iteration.
    #[arg(long, default_value_t = 200)]
    games_per_iteration: u32,

    /// Number of newest self-play shards to keep in the replay window.
    #[arg(long, default_value_t = 8)]
    replay_window_shards: usize,

    /// Number of Rayon worker threads to use for self-play and evaluation.
    #[arg(short = 'j', long, default_value_t = 0)]
    threads: usize,

    /// Candidate-vs-incumbent evaluation games after each iteration.
    #[arg(long, default_value_t = 20)]
    eval_games: usize,

    /// Minimum score fraction required to promote the candidate.
    #[arg(long, default_value_t = 0.55)]
    promotion_win_rate: f64,

    /// Search depth for self-play and evaluation.
    #[arg(long, default_value_t = 4)]
    search_depth: u8,

    /// Search time limit per move in milliseconds.
    #[arg(long, visible_alias = "search-time", default_value_t = 500)]
    search_time_ms: u64,

    /// Transposition table size in MB.
    #[arg(long, default_value_t = 16)]
    tt_size_mb: usize,

    /// Number of warm plies sampled with the warm temperature.
    #[arg(long, default_value_t = 8)]
    warm_plies: u16,

    /// Exploration temperature during warm plies.
    #[arg(long, default_value_t = 1.0)]
    warm_temp: f32,

    /// Exploration temperature after warm plies.
    #[arg(long, default_value_t = 0.1)]
    cool_temp: f32,

    /// Maximum neural inferences to batch together.
    #[arg(long, default_value_t = 64)]
    nn_batch_size: usize,

    /// Maximum time to wait for additional batched eval requests, in microseconds.
    #[arg(long, default_value_t = 500)]
    nn_batch_wait_us: u64,

    /// Teacher channels.
    #[arg(long)]
    channels: Option<i64>,

    /// Teacher residual blocks.
    #[arg(long)]
    blocks: Option<i64>,

    /// Teacher FiLM embedding dimension.
    #[arg(long)]
    film_dim: Option<i64>,

    /// Training epochs per iteration.
    #[arg(long, default_value_t = 50)]
    epochs: usize,

    /// Training batch size.
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,

    /// Learning rate.
    #[arg(long, default_value_t = 2e-3)]
    lr: f64,

    /// Validation split fraction.
    #[arg(long, default_value_t = 0.1)]
    val_split: f64,

    /// Base RNG seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Force CPU even if CUDA is available.
    #[arg(long)]
    cpu: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("failed to configure rayon thread pool");
    }

    let device = if !args.cpu && tch::Cuda::is_available() {
        println!("Using CUDA");
        Device::Cuda(0)
    } else {
        println!("Using CPU");
        Device::Cpu
    };
    println!("Using {} Rayon worker(s)", rayon::current_num_threads());

    let mut net = NetConfig::teacher();
    if let Some(channels) = args.channels {
        net.channels = channels;
    }
    if let Some(blocks) = args.blocks {
        net.num_blocks = blocks;
    }
    if let Some(film_dim) = args.film_dim {
        net.film_embed_dim = film_dim;
    }

    let train_config = TrainConfig {
        net,
        learning_rate: args.lr,
        epochs: args.epochs,
        batch_size: args.batch_size,
        val_split: args.val_split,
        checkpoint_dir: args.checkpoint_dir,
        ..Default::default()
    };

    let config = TeacherIterationConfig {
        primary_size: args.primary_board_size,
        iterations: args.iterations,
        games_per_iteration: args.games_per_iteration,
        replay_window_shards: args.replay_window_shards,
        eval_games: args.eval_games,
        promotion_win_rate: args.promotion_win_rate,
        seed: args.seed,
        bootstrap_shards_dir: args.bootstrap_shards,
        selfplay_dir: args.selfplay_dir,
        initial_checkpoint: args.resume,
        temperature: TemperatureSchedule {
            noise_plies: 6,
            warm_plies: args.warm_plies,
            warm_temp: args.warm_temp,
            cool_temp: args.cool_temp,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
        },
        inference: BatchedInferenceConfig {
            max_batch_size: args.nn_batch_size,
            max_wait_micros: args.nn_batch_wait_us,
        },
        search_config: SearchConfig {
            max_depth: args.search_depth,
            max_time_ms: args.search_time_ms,
            tt_size_mb: args.tt_size_mb,
        },
        train_config,
    };

    run_teacher_iteration_loop(&config, device)
}

#[cfg(test)]
mod tests {
    use clap::error::ErrorKind;
    use clap::Parser;

    use super::Args;

    #[test]
    fn parses_primary_board_size_flag() {
        let args = Args::try_parse_from(["tak-iterate", "--primary-board-size", "7"])
            .expect("primary board size flag should parse");

        assert_eq!(args.primary_board_size, 7);
    }

    #[test]
    fn rejects_legacy_board_size_flag() {
        let err = match Args::try_parse_from(["tak-iterate", "--board-size", "7"]) {
            Ok(_) => panic!("legacy board size flag should be rejected"),
            Err(err) => err,
        };

        assert_eq!(err.kind(), ErrorKind::UnknownArgument);
    }
}
