/*!
Checkpoint I/O — tch `VarStore` weights + JSON training state sidecar.

Follows the same pattern as HexZero's `checkpoint.rs`: atomic writes via
tmp+rename, iteration pruning, best-model promotion.
*/

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tch::nn;

use crate::config::NetConfig;

// ---------------------------------------------------------------------------
// Weight I/O
// ---------------------------------------------------------------------------

/// Save model weights (VarStore) to `path` using tch's format.
pub fn save_weights(vs: &nn::VarStore, path: &Path) -> anyhow::Result<()> {
    vs.save(path)
        .with_context(|| format!("save_weights: {}", path.display()))
}

/// Load weights from file into `vs`, ignoring missing keys.
pub fn load_weights(vs: &mut nn::VarStore, path: &Path) -> anyhow::Result<Vec<String>> {
    let missing = vs
        .load_partial(path)
        .with_context(|| format!("load_weights: {}", path.display()))?;
    Ok(missing)
}

// ---------------------------------------------------------------------------
// Training state (JSON sidecar)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrainingState {
    pub epoch: usize,
    pub global_step: usize,
    pub best_val_loss: f64,
    pub net_config: Option<NetConfig>,
    pub model_type: String, // "teacher" or "student"
}

pub fn save_training_state(dir: &Path, state: &TrainingState) -> anyhow::Result<()> {
    fs::create_dir_all(dir)?;
    let tmp = dir.join("training_state.json.tmp");
    let dest = dir.join("training_state.json");
    fs::write(&tmp, serde_json::to_string_pretty(state)?)?;
    fs::rename(tmp, dest)?;
    Ok(())
}

pub fn load_training_state(dir: &Path) -> TrainingState {
    let path = dir.join("training_state.json");
    fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Checkpoint save/load
// ---------------------------------------------------------------------------

/// Save a full checkpoint: weights + training state.
///
/// Writes atomically: `{dir}/{name}.pt` + `training_state.json`.
pub fn save_checkpoint(
    vs: &nn::VarStore,
    state: &TrainingState,
    dir: &Path,
    name: &str,
) -> anyhow::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let weights_name = format!("{name}.pt");
    let tmp_path = dir.join(format!("tmp_{weights_name}"));
    let final_path = dir.join(&weights_name);

    save_weights(vs, &tmp_path)?;
    fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("rename tmp → {weights_name}"))?;

    save_training_state(dir, state)?;
    Ok(final_path)
}

/// Load a checkpoint: weights + training state.
pub fn load_checkpoint(
    vs: &mut nn::VarStore,
    dir: &Path,
    name: &str,
) -> anyhow::Result<TrainingState> {
    let weights_path = dir.join(format!("{name}.pt"));
    let missing = load_weights(vs, &weights_path)?;
    if !missing.is_empty() {
        eprintln!(
            "Warning: {} missing keys when loading {}",
            missing.len(),
            weights_path.display()
        );
    }
    Ok(load_training_state(dir))
}

/// Return path of best checkpoint if it exists.
pub fn best_checkpoint_path(dir: &Path) -> Option<PathBuf> {
    let p = dir.join("best.pt");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tak_train_ckpt_test_{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn training_state_round_trip() {
        let dir = tmp_dir("round_trip");
        let state = TrainingState {
            epoch: 42,
            global_step: 1000,
            best_val_loss: 0.123,
            net_config: Some(NetConfig::teacher()),
            model_type: "teacher".into(),
        };
        save_training_state(&dir, &state).unwrap();
        let loaded = load_training_state(&dir);
        assert_eq!(loaded.epoch, 42);
        assert_eq!(loaded.global_step, 1000);
        assert!((loaded.best_val_loss - 0.123).abs() < 1e-6);
        assert_eq!(loaded.model_type, "teacher");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_state_returns_default() {
        let dir = tmp_dir("missing");
        let state = load_training_state(&dir);
        assert_eq!(state.epoch, 0);
        assert_eq!(state.global_step, 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_and_load_checkpoint() {
        use crate::config::NetConfig;
        use crate::net::TakNet;

        let dir = tmp_dir("save_load");
        let cfg = NetConfig::student();

        // Save
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let _net = TakNet::new(&vs, &cfg);

        let state = TrainingState {
            epoch: 5,
            global_step: 500,
            best_val_loss: 0.5,
            net_config: Some(cfg.clone()),
            model_type: "student".into(),
        };
        let path = save_checkpoint(&vs, &state, &dir, "best").unwrap();
        assert!(path.exists());

        // Load into fresh VarStore
        let mut vs2 = nn::VarStore::new(tch::Device::Cpu);
        let _net2 = TakNet::new(&vs2, &cfg);
        let loaded = load_checkpoint(&mut vs2, &dir, "best").unwrap();
        assert_eq!(loaded.epoch, 5);
        assert_eq!(loaded.model_type, "student");

        let _ = fs::remove_dir_all(&dir);
    }
}
