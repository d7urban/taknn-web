import os
import random
import torch
from torch.utils.data import Dataset
from .shard import ShardDataset
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
