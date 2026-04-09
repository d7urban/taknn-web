import os
import random
import torch
from torch.utils.data import Dataset
from .shard import ShardDataset
from .labels import descriptors_to_tensors
import tak_python


class Manifest:
    def __init__(self, manifest_path):
        self.manifest_path = manifest_path
        self.shards = []
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                self.shards = [line.strip() for line in f if line.strip()]

    def add_shard(self, shard_path):
        self.shards.append(shard_path)
        with open(self.manifest_path, 'a') as f:
            f.write(shard_path + "\n")


class ReplayBuffer(Dataset):
    """Replay buffer with tactical thinning and size quotas."""

    # Tactical phase keep probabilities (BUILD_SPEC 5.6):
    # Quiet=1/4, SemiTactical=1/2, Tactical=1/1
    THIN_RATES = {0: 0.25, 1: 0.5, 2: 1.0}

    def __init__(self, manifest, window_size=5_000_000, size_quotas=None):
        self.manifest = manifest
        self.window_size = window_size
        self.size_quotas = size_quotas or {
            3: 0.02, 4: 0.10, 5: 0.25, 6: 0.55, 7: 0.05, 8: 0.03
        }
        self.all_records = []
        self._load_shards()

    def _load_shards(self):
        if not self.manifest.shards:
            return

        buckets = {size: [] for size in range(3, 9)}
        total = 0

        for shard_path in reversed(self.manifest.shards):
            ds = ShardDataset([shard_path])
            for i in range(len(ds)):
                record = ds[i]
                phase = record.get('tactical_phase', 0)
                if random.random() > self.THIN_RATES.get(phase, 1.0):
                    continue

                board_size = record['size_id'] + 3
                quota = int(self.window_size * self.size_quotas.get(board_size, 0))
                if len(buckets[board_size]) < quota:
                    buckets[board_size].append(record)
                    total += 1

            if total >= self.window_size:
                break

        self.all_records = []
        for size in range(3, 9):
            self.all_records.extend(buckets[size])
        random.shuffle(self.all_records)

    def __len__(self):
        return len(self.all_records)

    def __getitem__(self, idx):
        return self.all_records[idx]


def collate_with_descriptors(batch):
    """Custom collate_fn that pads variable-length descriptors and policy targets.

    Each sample in batch should have:
        board_tensor, size_id, wdl, margin, policy_indices, policy_probs,
        descriptors (from descriptors_to_tensors), and optionally aux labels.

    Returns a dict with all tensors padded to max_moves in the batch.
    """
    max_moves = max(sample["descriptors"]["src"].shape[0] for sample in batch)
    B = len(batch)

    # Standard fixed-shape tensors
    board_tensors = torch.stack([s["board_tensor"] for s in batch])
    size_ids = torch.stack([s["size_id"] for s in batch])
    wdl = torch.stack([s["wdl"] for s in batch])
    margin = torch.stack([s["margin"] for s in batch])
    num_moves = torch.tensor([s["descriptors"]["src"].shape[0] for s in batch], dtype=torch.long)

    # Pad descriptors
    desc_keys_long = ["src", "dst", "move_type", "piece_type", "direction",
                      "pickup_count", "drop_template_id", "travel_length"]
    desc_keys_float = ["capstone_flatten", "enters_occupied", "opening_phase"]
    padded_descs = {}

    for key in desc_keys_long:
        t = torch.zeros(B, max_moves, dtype=torch.long)
        for i, s in enumerate(batch):
            n = s["descriptors"][key].shape[0]
            t[i, :n] = s["descriptors"][key]
        padded_descs[key] = t

    for key in desc_keys_float:
        t = torch.zeros(B, max_moves, dtype=torch.float32)
        for i, s in enumerate(batch):
            n = s["descriptors"][key].shape[0]
            t[i, :n] = s["descriptors"][key]
        padded_descs[key] = t

    # Path: [B, M, 7]
    path_t = torch.full((B, max_moves, 7), 255, dtype=torch.long)
    for i, s in enumerate(batch):
        n = s["descriptors"]["path"].shape[0]
        path_t[i, :n, :] = s["descriptors"]["path"]
    padded_descs["path"] = path_t

    # Pad policy targets to max_moves
    policy_target = torch.zeros(B, max_moves, dtype=torch.float32)
    for i, s in enumerate(batch):
        indices = s["policy_indices"].long()
        probs = s["policy_probs"].float()
        # Indices are into legal move list, so they map directly
        for j in range(len(indices)):
            idx = indices[j].item()
            if idx < max_moves:
                policy_target[i, idx] = probs[j].item()

    result = {
        "board_tensor": board_tensors,
        "size_id": size_ids,
        "wdl": wdl,
        "margin": margin,
        "descriptors": padded_descs,
        "num_moves": num_moves,
        "policy_target": policy_target,
    }

    # Optional aux labels
    if "road_threat_label" in batch[0]:
        result["road_threat_label"] = torch.stack([s["road_threat_label"] for s in batch])
        result["block_threat_label"] = torch.stack([s["block_threat_label"] for s in batch])
        result["cap_flatten_label"] = torch.stack([s["cap_flatten_label"] for s in batch])
        result["endgame_label"] = torch.stack([s["endgame_label"] for s in batch])

    return result


def apply_d4_augmentation(board_tensor, policy_indices, policy_probs, tps_str, board_size, transform=None):
    """Apply a random D4 symmetry transform to board tensor and remap policy indices.

    Uses the Rust engine (via PyO3) for exact policy index remapping.

    Args:
        board_tensor: [31, 8, 8] tensor
        policy_indices: tensor of move indices
        policy_probs: tensor of move probabilities
        tps_str: TPS string of the position (needed to reconstruct game state in Rust)
        board_size: board size (3..8)
        transform: D4 transform index (0..7), or None for random
    Returns:
        (augmented_tensor, remapped_indices, policy_probs)
    """
    if transform is None:
        transform = random.randint(0, 7)

    if transform == 0:
        return board_tensor, policy_indices, policy_probs

    # Transform spatial channels via torch ops
    spatial = board_tensor[0:23, :, :].clone()

    if transform == 1:  # Rot90
        spatial = torch.rot90(spatial, k=1, dims=[1, 2])
    elif transform == 2:  # Rot180
        spatial = torch.rot90(spatial, k=2, dims=[1, 2])
    elif transform == 3:  # Rot270
        spatial = torch.rot90(spatial, k=3, dims=[1, 2])
    elif transform == 4:  # ReflectH
        spatial = torch.flip(spatial, dims=[1])
    elif transform == 5:  # ReflectV
        spatial = torch.flip(spatial, dims=[2])
    elif transform == 6:  # ReflectMain
        spatial = spatial.transpose(1, 2)
    elif transform == 7:  # ReflectAnti
        spatial = torch.rot90(spatial, k=1, dims=[1, 2]).flip(dims=[1])

    aug_tensor = board_tensor.clone()
    aug_tensor[0:23, :, :] = spatial

    # Remap policy indices via Rust D4 transform
    game = tak_python.PyGameState(board_size, tps_str)
    tmap = game.get_transformation_map(transform)

    remapped = torch.tensor(
        [tmap[idx.item()] for idx in policy_indices],
        dtype=policy_indices.dtype,
    )
    return aug_tensor, remapped, policy_probs
