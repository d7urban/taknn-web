/*!
Export pipeline: tch checkpoints, TPOL policy weights, and ONNX conversion.

Exports:
  1. All model weights as a tch checkpoint + JSON config sidecar
  2. Policy MLP + embedding weights as TPOL binary (for WASM policy scorer)
  3. A Python script for ONNX conversion (trunk + value heads)

The policy head can't be exported to ONNX because it takes variable-length
descriptor inputs. Instead, we export the trunk (which produces spatial +
global features) and the policy weights separately. The WASM side handles
descriptor construction and MLP eval.
*/

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use tch::{nn, Device, Kind, Tensor};

use crate::config::NetConfig;

// ---------------------------------------------------------------------------
// TPOL policy weight export
// ---------------------------------------------------------------------------

/// Policy-related parameter name substrings to include in the TPOL binary.
const POLICY_KEYS: &[&str] = &[
    "policy_mlp",
    "move_type_emb",
    "piece_type_emb",
    "direction_emb",
    "pickup_count_emb",
    "drop_template_emb",
    "travel_length_emb",
];

/// Export policy MLP and embedding weights as a flat TPOL binary for WASM.
///
/// ## Format
/// ```text
/// Header: magic "TPOL" (4B) + version u32 LE (4B) + num_blobs u32 LE (4B)
/// Each blob:
///   name_len  u16 LE (2B)
///   name      UTF-8 bytes
///   ndims     u8 (1B)
///   dims      u32 LE × ndims
///   data      f32 LE × numel
/// ```
pub fn export_policy_weights(vs: &nn::VarStore, out_path: &Path) -> anyhow::Result<()> {
    let variables = vs.variables_.lock().unwrap();
    let named_vars = &variables.named_variables;

    // Collect policy-related variables, sorted by name for determinism
    let mut policy_vars: BTreeMap<&str, &Tensor> = BTreeMap::new();

    for (name, tensor) in named_vars {
        if POLICY_KEYS.iter().any(|key| name.contains(key)) {
            policy_vars.insert(name.as_str(), tensor);
        }
    }

    if policy_vars.is_empty() {
        anyhow::bail!("No policy parameters found in VarStore");
    }

    let mut f = File::create(out_path)?;

    // Header
    f.write_all(b"TPOL")?;
    f.write_all(&1u32.to_le_bytes())?; // version
    f.write_all(&(policy_vars.len() as u32).to_le_bytes())?;

    for (&name, tensor) in &policy_vars {
        let tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float);
        let shape = tensor.size();

        // Name
        let name_bytes = name.as_bytes();
        f.write_all(&(name_bytes.len() as u16).to_le_bytes())?;
        f.write_all(name_bytes)?;

        // Dims
        f.write_all(&[shape.len() as u8])?;
        for &dim in &shape {
            f.write_all(&(dim as u32).to_le_bytes())?;
        }

        // Float32 data (little-endian)
        let flat: Vec<f32> = Vec::<f32>::try_from(tensor.reshape([-1]))?;
        for &val in &flat {
            f.write_all(&val.to_le_bytes())?;
        }
    }

    f.flush()?;

    let size_kb = fs::metadata(out_path)?.len() as f64 / 1024.0;
    println!(
        "Exported policy weights: {} ({:.0} KB, {} blobs)",
        out_path.display(),
        size_kb,
        policy_vars.len()
    );
    for (&name, tensor) in &policy_vars {
        println!("  {name}: {:?}", tensor.size());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Full model checkpoint export
// ---------------------------------------------------------------------------

/// Export all model weights in safetensors format.
///
/// Also writes a `net_config.json` sidecar with the architecture config,
/// so the weights can be loaded without guessing the architecture.
pub fn export_checkpoint(
    vs: &nn::VarStore,
    cfg: &NetConfig,
    out_dir: &Path,
    name: &str,
) -> anyhow::Result<()> {
    fs::create_dir_all(out_dir)?;

    let weights_path = out_dir.join(format!("{name}.safetensors"));
    vs.save(&weights_path)?;
    let size_kb = fs::metadata(&weights_path)?.len() as f64 / 1024.0;
    println!(
        "Exported checkpoint: {} ({:.0} KB)",
        weights_path.display(),
        size_kb
    );

    let config_path = out_dir.join(format!("{name}_config.json"));
    fs::write(&config_path, serde_json::to_string_pretty(cfg)?)?;
    println!("Exported config: {}", config_path.display());

    Ok(())
}

// ---------------------------------------------------------------------------
// ONNX conversion script generation
// ---------------------------------------------------------------------------

/// Generate a Python script that loads a tch checkpoint and exports to ONNX.
///
/// This avoids tch-rs TorchScript tracing limitations (single return value)
/// by using Python's native `torch.onnx.export` which handles multi-output
/// models correctly.
pub fn generate_onnx_script(
    cfg: &NetConfig,
    checkpoint_path: &Path,
    onnx_path: &Path,
    script_path: &Path,
) -> anyhow::Result<()> {
    let script = format!(
        r#"#!/usr/bin/env python3
"""Convert a tch checkpoint to ONNX for browser deployment.

Generated by tak-export. Requires: torch, safetensors

Usage:
    python {script_name}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

# Architecture config
CHANNELS = {channels}
NUM_BLOCKS = {num_blocks}
FILM_EMBED_DIM = {film_embed_dim}
NUM_SIZES = {num_sizes}
C_IN = 31


class FiLMResBlock(nn.Module):
    def __init__(self, channels, film_embed_dim, prefix=""):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gamma1 = nn.Linear(film_embed_dim, channels)
        self.beta1 = nn.Linear(film_embed_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gamma2 = nn.Linear(film_embed_dim, channels)
        self.beta2 = nn.Linear(film_embed_dim, channels)

    def forward(self, x, e):
        residual = x
        h = self.conv1(x)
        h = self.bn1(h)
        g1 = self.gamma1(e).unsqueeze(-1).unsqueeze(-1)
        b1 = self.beta1(e).unsqueeze(-1).unsqueeze(-1)
        h = F.relu(h * g1 + b1)
        h = self.conv2(h)
        h = self.bn2(h)
        g2 = self.gamma2(e).unsqueeze(-1).unsqueeze(-1)
        b2 = self.beta2(e).unsqueeze(-1).unsqueeze(-1)
        h = h * g2 + b2
        return F.relu(h + residual)


class TakNet(nn.Module):
    def __init__(self):
        super().__init__()
        c, e = CHANNELS, FILM_EMBED_DIM
        vh = 128 if c >= 128 else 64
        self.stem = nn.Sequential(
            nn.Conv2d(C_IN, c, 3, padding=1, bias=True),
            nn.BatchNorm2d(c),
        )
        self.size_embed = nn.Embedding(NUM_SIZES, e)
        self.blocks = nn.ModuleList(
            [FiLMResBlock(c, e) for _ in range(NUM_BLOCKS)]
        )
        self.v_hidden = nn.Linear(c + e, vh)
        self.wdl_head = nn.Linear(vh, 3)
        self.margin_head = nn.Linear(vh, 1)
        self.road_threat = nn.Conv2d(c, 2, 1)
        self.block_threat = nn.Conv2d(c, 2, 1)
        self.cap_flatten = nn.Conv2d(c, 1, 1)
        self.endgame_head = nn.Linear(c + e, 1)

    def forward(self, board, size_id):
        e = self.size_embed(size_id)
        x = F.relu(self.stem(board))
        for block in self.blocks:
            x = block(x, e)
        g = x.mean(dim=[-2, -1])
        v_in = torch.cat([g, e], dim=1)
        v_hid = F.relu(self.v_hidden(v_in))
        wdl = F.softmax(self.wdl_head(v_hid), dim=1)
        margin = torch.tanh(self.margin_head(v_hid))
        return wdl, margin, x, g


class TrunkWrapper(nn.Module):
    """Wraps TakNet to return a tuple (required for ONNX export)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, board, size_id):
        return self.model(board, size_id)


def load_and_export():
    weights_path = "{checkpoint_path}"
    onnx_path = "{onnx_path}"

    # Build model and load weights
    model = TakNet()
    state = load_file(weights_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: {{len(missing)}} missing keys (policy params expected)")
    if unexpected:
        print(f"Warning: {{len(unexpected)}} unexpected keys")

    model.eval()
    wrapper = TrunkWrapper(model)
    wrapper.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {{CHANNELS}}ch x {{NUM_BLOCKS}} blocks, {{param_count:,}} params")

    # Export
    board = torch.randn(1, C_IN, 8, 8)
    size_id = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (board, size_id),
        onnx_path,
        input_names=["board_tensor", "size_id"],
        output_names=["wdl", "margin", "spatial", "global_pool"],
        external_data=False,
        dynamic_axes={{
            k: {{0: "batch"}}
            for k in ["board_tensor", "size_id", "wdl", "margin", "spatial", "global_pool"]
        }},
        opset_version=17,
    )

    import os
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"Exported ONNX: {{onnx_path}} ({{size_kb:.0f}} KB)")

    # Optional: verify
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(onnx_path)
        with torch.no_grad():
            pt_wdl, pt_margin, _, _ = model(board, size_id)

        ort_out = sess.run(None, {{
            "board_tensor": board.numpy(),
            "size_id": size_id.numpy(),
        }})
        wdl_diff = np.abs(pt_wdl.numpy() - ort_out[0]).max()
        margin_diff = np.abs(pt_margin.numpy() - ort_out[1]).max()
        ok = wdl_diff < 1e-4 and margin_diff < 1e-4
        print(f"ONNX verification: {{'PASS' if ok else 'FAIL'}} (wdl={{wdl_diff:.6f}}, margin={{margin_diff:.6f}})")
    except ImportError:
        print("onnxruntime not installed, skipping verification")


if __name__ == "__main__":
    load_and_export()
"#,
        script_name = script_path.file_name().unwrap().to_string_lossy(),
        channels = cfg.channels,
        num_blocks = cfg.num_blocks,
        film_embed_dim = cfg.film_embed_dim,
        num_sizes = cfg.num_sizes,
        checkpoint_path = checkpoint_path.display(),
        onnx_path = onnx_path.display(),
    );

    fs::write(script_path, script)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = fs::Permissions::from_mode(0o755);
        fs::set_permissions(script_path, perms)?;
    }

    println!(
        "Generated ONNX conversion script: {}",
        script_path.display()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NetConfig;
    use crate::net::TakNet;
    use crate::policy::PolicyScorer;
    use tch::nn;

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("tak_export_test_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn tpol_header_and_blob_count() {
        let cfg = NetConfig::student();
        let vs = nn::VarStore::new(Device::Cpu);
        let _net = TakNet::new(&vs, &cfg);
        let _policy = PolicyScorer::new(&vs, &cfg);

        let dir = tmp_dir("tpol_header");
        let path = dir.join("policy.bin");

        export_policy_weights(&vs, &path).unwrap();

        let data = std::fs::read(&path).unwrap();
        assert_eq!(&data[0..4], b"TPOL");
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        assert_eq!(version, 1);
        let num_blobs = u32::from_le_bytes(data[8..12].try_into().unwrap());
        // 6 embedding weights + 4 MLP params (2 linear layers × {weight, bias})
        assert_eq!(num_blobs, 10, "expected 10 policy blobs, got {num_blobs}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tpol_round_trip_read() {
        let cfg = NetConfig::student();
        let vs = nn::VarStore::new(Device::Cpu);
        let _net = TakNet::new(&vs, &cfg);
        let _policy = PolicyScorer::new(&vs, &cfg);

        let dir = tmp_dir("tpol_roundtrip");
        let path = dir.join("policy.bin");

        export_policy_weights(&vs, &path).unwrap();

        // Read back and verify blob format
        let data = std::fs::read(&path).unwrap();
        let num_blobs = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let mut offset = 12;
        for _ in 0..num_blobs {
            let name_len =
                u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as usize;
            offset += 2;
            let name = std::str::from_utf8(&data[offset..offset + name_len]).unwrap();
            offset += name_len;
            let ndims = data[offset] as usize;
            offset += 1;
            let mut numel: usize = 1;
            for _ in 0..ndims {
                let dim = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                numel *= dim;
                offset += 4;
            }
            offset += numel * 4; // skip float32 data
            assert!(!name.is_empty(), "blob name should not be empty");
        }
        assert_eq!(offset, data.len(), "all data should be consumed");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn checkpoint_export_creates_files() {
        let cfg = NetConfig::student();
        let vs = nn::VarStore::new(Device::Cpu);
        let _net = TakNet::new(&vs, &cfg);

        let dir = tmp_dir("checkpoint");
        export_checkpoint(&vs, &cfg, &dir, "student").unwrap();

        assert!(dir.join("student.safetensors").exists());
        assert!(dir.join("student_config.json").exists());

        // Verify config round-trips
        let json = std::fs::read_to_string(dir.join("student_config.json")).unwrap();
        let loaded: NetConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.channels, 64);
        assert_eq!(loaded.num_blocks, 6);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn onnx_script_generated() {
        let cfg = NetConfig::student();
        let dir = tmp_dir("onnx_script");

        let checkpoint = dir.join("model.safetensors");
        let onnx = dir.join("model.onnx");
        let script = dir.join("convert_onnx.py");

        generate_onnx_script(&cfg, &checkpoint, &onnx, &script).unwrap();

        assert!(script.exists());
        let content = std::fs::read_to_string(&script).unwrap();
        assert!(content.contains("CHANNELS = 64"));
        assert!(content.contains("NUM_BLOCKS = 6"));
        assert!(content.contains("torch.onnx.export"));
        assert!(content.contains("model.safetensors"));
        assert!(content.contains("load_file("));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
