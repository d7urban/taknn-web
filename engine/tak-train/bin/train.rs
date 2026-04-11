use std::path::PathBuf;

use clap::Parser;
use tch::Device;

use tak_train::config::{NetConfig, TrainConfig};
use tak_train::data::{load_shards, train_val_split, ShardLoader};
use tak_train::trainer::Trainer;

#[derive(Parser)]
#[command(name = "tak-train", about = "Train TakNN teacher/student model")]
struct Args {
    /// Directory containing .tknn shard files
    #[arg(long, default_value = "shards")]
    shards: PathBuf,

    /// Model type: "teacher" or "student"
    #[arg(long, default_value = "teacher")]
    model: String,

    /// Number of channels
    #[arg(long)]
    channels: Option<i64>,

    /// Number of residual blocks
    #[arg(long)]
    blocks: Option<i64>,

    /// FiLM embedding dimension
    #[arg(long)]
    film_dim: Option<i64>,

    /// Number of epochs
    #[arg(long, default_value_t = 50)]
    epochs: usize,

    /// Batch size
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value_t = 2e-3)]
    lr: f64,

    /// Validation split fraction
    #[arg(long, default_value_t = 0.1)]
    val_split: f64,

    /// Checkpoint output directory
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: PathBuf,

    /// Use CPU even if CUDA is available
    #[arg(long)]
    cpu: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let device = if !args.cpu && tch::Cuda::is_available() {
        println!("Using CUDA");
        Device::Cuda(0)
    } else {
        println!("Using CPU");
        Device::Cpu
    };

    // Network config
    let mut net_cfg = if args.model == "student" {
        NetConfig::student()
    } else {
        NetConfig::teacher()
    };
    if let Some(c) = args.channels {
        net_cfg.channels = c;
    }
    if let Some(b) = args.blocks {
        net_cfg.num_blocks = b;
    }
    if let Some(f) = args.film_dim {
        net_cfg.film_embed_dim = f;
    }

    let config = TrainConfig {
        net: net_cfg,
        learning_rate: args.lr,
        epochs: args.epochs,
        batch_size: args.batch_size,
        checkpoint_dir: args.checkpoint_dir,
        val_split: args.val_split,
        ..Default::default()
    };

    // Load data
    let shard_files: Vec<PathBuf> = std::fs::read_dir(&args.shards)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "tknn").unwrap_or(false))
        .collect();

    if shard_files.is_empty() {
        anyhow::bail!("No .tknn files found in {}", args.shards.display());
    }
    println!("Found {} shard files", shard_files.len());

    let records = load_shards(&shard_files);
    println!("Loaded {} records", records.len());

    let (train_records, val_records) = train_val_split(records, config.val_split);
    println!(
        "Train: {}, Val: {}",
        train_records.len(),
        val_records.len()
    );

    let mut train_loader = ShardLoader::new(train_records, config.batch_size);
    let mut val_loader = ShardLoader::new(val_records, config.batch_size);

    // Train
    let mut trainer = Trainer::new(config, device)?;
    trainer.train(&mut train_loader, &mut val_loader);

    Ok(())
}
