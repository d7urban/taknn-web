//! Standalone self-play data generator.
//!
//! Generates zstd-compressed shard files of training records using
//! PVS search with heuristic evaluation.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use clap::Parser;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use tak_data::selfplay::{SelfPlayConfig, SelfPlayEngine, TemperatureSchedule};
use tak_data::shard::ShardWriter;
use tak_search::pvs::SearchConfig;

#[derive(Parser)]
#[command(name = "selfplay", about = "Generate Tak self-play training data")]
struct Args {
    /// Output directory for shard files.
    #[arg(short, long, default_value = "shards")]
    output_dir: PathBuf,

    /// Board size (3-8).
    #[arg(short = 's', long, default_value_t = 4)]
    board_size: u8,

    /// Number of games to generate.
    #[arg(short = 'n', long, default_value_t = 100)]
    num_games: u32,

    /// Number of parallel worker threads.
    #[arg(short = 'j', long, default_value_t = 0)]
    threads: usize,

    /// Max search depth per position.
    #[arg(short = 'd', long, default_value_t = 4)]
    max_depth: u8,

    /// Max search time per position in milliseconds.
    #[arg(short = 't', long, default_value_t = 500)]
    max_time_ms: u64,

    /// Transposition table size in MB per thread.
    #[arg(long, default_value_t = 16)]
    tt_size_mb: usize,

    /// Games per shard file.
    #[arg(long, default_value_t = 50)]
    games_per_shard: u32,

    /// Base random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Temperature for first N plies.
    #[arg(long, default_value_t = 8)]
    warm_plies: u16,
}

fn main() {
    let args = Args::parse();

    if args.board_size < 3 || args.board_size > 8 {
        eprintln!("Error: board_size must be 3-8");
        std::process::exit(1);
    }

    // Set up thread pool.
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("Failed to configure thread pool");
    }

    std::fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");

    let num_shards = args.num_games.div_ceil(args.games_per_shard);
    let total_records = AtomicU32::new(0);
    let total_games = AtomicU32::new(0);

    let start = std::time::Instant::now();

    (0..num_shards).into_par_iter().for_each(|shard_idx| {
        let first_game = shard_idx * args.games_per_shard;
        let last_game = ((shard_idx + 1) * args.games_per_shard).min(args.num_games);
        let games_in_shard = last_game - first_game;

        let config = SelfPlayConfig {
            board_size: args.board_size,
            search_config: SearchConfig {
                max_depth: args.max_depth,
                max_time_ms: args.max_time_ms,
                tt_size_mb: args.tt_size_mb,
            },
            temp_schedule: TemperatureSchedule {
                warm_plies: args.warm_plies,
                warm_temp: 1.0,
                cool_temp: 0.1,
            },
        };
        let engine = SelfPlayEngine::new(config);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed.wrapping_add(shard_idx as u64));

        let shard_path = args.output_dir.join(format!(
            "shard_{:04}_{size}x{size}.tknn",
            shard_idx,
            size = args.board_size
        ));
        let tmp_path = shard_path.with_extension("tknn.tmp");
        let file = std::fs::File::create(&tmp_path)
            .unwrap_or_else(|e| panic!("Failed to create {}: {e}", tmp_path.display()));
        let mut writer = ShardWriter::new(file)
            .unwrap_or_else(|e| panic!("Failed to init shard writer: {e}"));

        let mut shard_records = 0u32;
        for game_offset in 0..games_in_shard {
            let game_id = first_game + game_offset;
            let records = engine.play_game(&mut rng, game_id);
            for record in &records {
                writer.write_record(record).expect("Failed to write record");
            }
            shard_records += records.len() as u32;
        }

        writer.finish().expect("Failed to finish shard");
        std::fs::rename(&tmp_path, &shard_path)
            .unwrap_or_else(|e| panic!("Failed to rename {}: {e}", tmp_path.display()));

        total_records.fetch_add(shard_records, Ordering::Relaxed);
        let done = total_games.fetch_add(games_in_shard, Ordering::Relaxed) + games_in_shard;
        eprintln!(
            "[{done}/{total}] shard {shard_idx}: {games_in_shard} games, {shard_records} records → {}",
            shard_path.display(),
            total = args.num_games,
        );
    });

    let elapsed = start.elapsed();
    let recs = total_records.load(Ordering::Relaxed);
    eprintln!(
        "\nDone: {} games → {} records in {:.1}s ({:.1} games/s, {:.1} records/s)",
        args.num_games,
        recs,
        elapsed.as_secs_f64(),
        args.num_games as f64 / elapsed.as_secs_f64(),
        recs as f64 / elapsed.as_secs_f64(),
    );
}
