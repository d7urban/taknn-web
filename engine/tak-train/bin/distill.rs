use std::path::PathBuf;

use clap::Parser;
use tch::Device;

use tak_train::config::{NetConfig, TrainConfig};
use tak_train::data::{load_shards, train_val_split, ShardLoader};
use tak_train::distill::DistillTrainer;

#[derive(Parser)]
#[command(name = "tak-distill", about = "Distill teacher → student")]
struct Args {
    /// Path to teacher checkpoint (.pt)
    #[arg(long)]
    teacher: PathBuf,

    /// Directory containing .tknn shard files
    #[arg(long, default_value = "shards")]
    shards: PathBuf,

    /// Student channels
    #[arg(long, default_value_t = 64)]
    student_channels: i64,

    /// Student blocks
    #[arg(long, default_value_t = 6)]
    student_blocks: i64,

    /// Student FiLM dimension
    #[arg(long, default_value_t = 16)]
    student_film_dim: i64,

    /// Number of epochs
    #[arg(long, default_value_t = 30)]
    epochs: usize,

    /// Batch size
    #[arg(long, default_value_t = 256)]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    /// Distillation temperature
    #[arg(long, default_value_t = 2.0)]
    temperature: f64,

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

    let teacher_cfg = NetConfig::teacher();
    let student_cfg = NetConfig {
        channels: args.student_channels,
        num_blocks: args.student_blocks,
        film_embed_dim: args.student_film_dim,
        num_sizes: 6,
    };

    let config = TrainConfig {
        net: student_cfg,
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

    // Build distiller and load teacher
    let mut distiller = DistillTrainer::new(&teacher_cfg, config, device, args.temperature)?;
    distiller.load_teacher(&args.teacher)?;
    println!("Loaded teacher from {}", args.teacher.display());

    distiller.distill(&mut train_loader, &mut val_loader);

    Ok(())
}
