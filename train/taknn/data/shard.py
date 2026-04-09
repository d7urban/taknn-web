"""Shard dataset backed by the Rust PyO3 reader (tak_python.PyShardReader)."""

import torch
from torch.utils.data import Dataset
import tak_python

from .labels import compute_aux_labels, descriptors_to_tensors


class ShardDataset(Dataset):
    """Reads zstd-compressed .tknn shard files via the Rust reader."""

    def __init__(self, shard_paths, load_descriptors=False, load_aux_labels=False):
        """
        Args:
            shard_paths: list of .tknn file paths
            load_descriptors: if True, compute move descriptors for each position
            load_aux_labels: if True, compute auxiliary spatial labels
        """
        self.records = []
        self.load_descriptors = load_descriptors
        self.load_aux_labels = load_aux_labels
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
        board_size = rec["board_size"]
        stm = rec["side_to_move"]

        # WDL target: blend search-derived teacher_wdl with game outcome
        result = rec["game_result"]
        search_wdl = list(rec["teacher_wdl"])
        is_placeholder = all(abs(w - 0.33) < 0.01 for w in search_wdl)

        # Game outcome WDL (from STM perspective)
        white_wins = result in (0, 2)
        black_wins = result in (1, 3)
        draw = result == 4
        if draw:
            game_wdl = [0.0, 1.0, 0.0]
        elif (white_wins and stm == 0) or (black_wins and stm == 1):
            game_wdl = [1.0, 0.0, 0.0]
        else:
            game_wdl = [0.0, 0.0, 1.0]

        if is_placeholder:
            wdl = game_wdl
        else:
            wdl = [0.7 * s + 0.3 * g for s, g in zip(search_wdl, game_wdl)]

        # Margin target
        margin_norm = rec["teacher_margin"]
        if is_placeholder:
            margin = rec["flat_margin"]
            if stm == 1:
                margin = -margin
            margin_norm = max(-1.0, min(1.0, margin / 16.0))

        sample = {
            "board_tensor": torch.from_numpy(rec["board_tensor"].copy()),
            "size_id": torch.tensor(board_size - 3, dtype=torch.long),
            "wdl": torch.tensor(wdl, dtype=torch.float32),
            "margin": torch.tensor([margin_norm], dtype=torch.float32),
            "policy_indices": torch.tensor(rec["policy_indices"], dtype=torch.long),
            "policy_probs": torch.tensor(rec["policy_probs"], dtype=torch.float32),
            "game_result": result,
            "side_to_move": stm,
            "tactical_phase": rec.get("tactical_phase", 0),
        }

        # Compute descriptors on-the-fly from TPS
        if self.load_descriptors or self.load_aux_labels:
            tps = rec["tps"]
            sample["tps"] = tps

        if self.load_descriptors:
            game = tak_python.PyGameState(board_size, tps)
            descs = game.get_move_descriptors()
            sample["descriptors"] = descriptors_to_tensors(descs)

        if self.load_aux_labels:
            aux = compute_aux_labels(tps, board_size)
            sample["road_threat_label"] = aux["road_threat"]
            sample["block_threat_label"] = aux["block_threat"]
            sample["cap_flatten_label"] = aux["cap_flatten"]
            sample["endgame_label"] = aux["endgame"]

        return sample
