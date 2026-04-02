"""Shard dataset backed by the Rust PyO3 reader (tak_python.PyShardReader)."""

import torch
from torch.utils.data import Dataset
import tak_python


class ShardDataset(Dataset):
    """Reads zstd-compressed .tknn shard files via the Rust reader."""

    def __init__(self, shard_paths):
        self.records = []
        for path in shard_paths:
            reader = tak_python.PyShardReader(path)
            while True:
                rec = reader.next()
                if rec is None:
                    break
                self.records.append(rec)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return {
            "board_tensor": torch.from_numpy(rec["board_tensor"].copy()),
            "size_id": rec["board_size"] - 3,
            "side_to_move": rec["side_to_move"],
            "game_result": rec["game_result"],
            "flat_margin": rec["flat_margin"],
            "teacher_wdl": torch.tensor(rec["teacher_wdl"], dtype=torch.float32),
            "teacher_margin": rec["teacher_margin"],
            "policy_indices": torch.tensor(rec["policy_indices"], dtype=torch.long),
            "policy_probs": torch.tensor(rec["policy_probs"], dtype=torch.float32),
        }
