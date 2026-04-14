use std::path::PathBuf;

use clap::Parser;
use tch::{nn, Device};

use tak_train::checkpoint;
use tak_train::config::NetConfig;
use tak_train::export;
use tak_train::net::TakNet;
use tak_train::policy::PolicyScorer;

#[derive(Parser)]
#[command(
    name = "tak-export",
    about = "Export TakNN model for browser deployment"
)]
struct Args {
    /// Checkpoint directory (must contain .pt + training_state.json)
    #[arg(long)]
    checkpoint: PathBuf,

    /// Checkpoint name (default: "best")
    #[arg(long, default_value = "best")]
    name: String,

    /// Output directory
    #[arg(long, default_value = "exports")]
    out: PathBuf,

    /// Model type override: "teacher" or "student"
    /// (auto-detected from training_state.json if not set)
    #[arg(long)]
    model: Option<String>,

    /// Skip ONNX conversion script generation
    #[arg(long)]
    skip_onnx: bool,

    /// Skip TPOL policy weight export
    #[arg(long)]
    skip_policy: bool,

    /// Generate an INT8 quantized ONNX model
    #[arg(long)]
    quantize: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Always export on CPU for reproducibility
    let device = Device::Cpu;

    // Load training state to determine model config
    let state = checkpoint::load_training_state(&args.checkpoint);
    let net_cfg = if let Some(ref model_type) = args.model {
        match model_type.as_str() {
            "teacher" => NetConfig::teacher(),
            "student" => NetConfig::student(),
            other => anyhow::bail!("Unknown model type: {other}"),
        }
    } else if let Some(cfg) = state.net_config {
        cfg
    } else {
        match state.model_type.as_str() {
            "teacher" => NetConfig::teacher(),
            _ => NetConfig::student(),
        }
    };

    let model_type = args.model.as_deref().unwrap_or(if net_cfg.channels >= 128 {
        "teacher"
    } else {
        "student"
    });

    println!(
        "Model: {model_type} ({}ch × {} blocks, FiLM dim {})",
        net_cfg.channels, net_cfg.num_blocks, net_cfg.film_embed_dim
    );

    // Build model and load weights
    let mut vs = nn::VarStore::new(device);
    let _net = TakNet::new(&vs, &net_cfg);
    let _policy = PolicyScorer::new(&vs, &net_cfg);

    let loaded = checkpoint::load_checkpoint(&mut vs, &args.checkpoint, &args.name)?;
    println!(
        "Loaded checkpoint: epoch {}, step {}, val_loss {:.4}",
        loaded.epoch, loaded.global_step, loaded.best_val_loss
    );

    let param_count: i64 = vs
        .trainable_variables()
        .iter()
        .map(|t| t.numel() as i64)
        .sum();
    println!("Parameters: {param_count}");

    std::fs::create_dir_all(&args.out)?;

    // ONNX conversion script
    if !args.skip_onnx {
        let checkpoint_path = args.out.join(format!("{model_type}.safetensors"));
        let onnx_path = args.out.join(format!("{model_type}_trunk.onnx"));
        let script_path = args.out.join(format!("convert_{model_type}_onnx.py"));
        export::generate_onnx_script(
            &net_cfg,
            &checkpoint_path,
            &onnx_path,
            &script_path,
            args.quantize,
        )?;
    }

    // TPOL policy weight export
    if !args.skip_policy {
        let tpol_path = args.out.join(format!("{model_type}_policy.bin"));
        export::export_policy_weights(&vs, &tpol_path)?;
    }

    // Also export a standalone checkpoint + config for portability
    export::export_checkpoint(&vs, &net_cfg, &args.out, model_type)?;

    println!("\nExport complete.");
    Ok(())
}
