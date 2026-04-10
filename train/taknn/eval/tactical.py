"""Tactical test suite runner using policy head scoring."""

import json
import os

import torch
import tak_python

from ..data.labels import descriptors_to_tensors


class TacticalSuiteRunner:
    """Evaluates model policy accuracy on positions with known tactical solutions.

    Each suite position has:
        tps: TPS string of the position
        size: board size (3-8)
        category: one of "road_win_in_1", "forced_road_block", "capstone_flatten"
        expected_move_indices: list of acceptable move indices (any match = correct)

    The runner scores all legal moves via the policy head and checks whether
    the model's top-1 (or top-k) choice is among the expected moves.
    """

    def __init__(self, model, suite_path, device="cpu"):
        self.model = model
        self.device = device
        self.positions = []

        if os.path.isdir(suite_path):
            for fname in sorted(os.listdir(suite_path)):
                if fname.endswith(".jsonl"):
                    self._load_suite(os.path.join(suite_path, fname))
        else:
            self._load_suite(suite_path)

    def _load_suite(self, path):
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.positions.append(json.loads(line))
        except FileNotFoundError:
            pass

    @torch.no_grad()
    def run(self, top_k=1):
        """Run the suite, checking if any top-k policy move is correct.

        Returns dict with overall and per-category results.
        """
        self.model.eval()
        results_by_cat = {}
        overall_correct = 0
        overall_total = 0

        for pos in self.positions:
            tps = pos["tps"]
            size = pos["size"]
            expected = set(pos["expected_move_indices"])
            category = pos.get("category", "unknown")

            if category not in results_by_cat:
                results_by_cat[category] = {"correct": 0, "total": 0}

            game = tak_python.PyGameState(size, tps)
            if game.is_terminal() or game.legal_move_count() == 0:
                continue

            # Get model outputs
            tensor = torch.from_numpy(game.encode_tensor().copy())
            tensor = tensor.unsqueeze(0).float().to(self.device)
            size_id = torch.tensor([size - 3], dtype=torch.long, device=self.device)

            outputs = self.model(tensor, size_id)

            # Get move descriptors and score via policy head
            descs = game.get_move_descriptors()
            desc_tensors = descriptors_to_tensors(descs)

            # Batch dimension: unsqueeze all descriptor tensors
            batched_descs = {}
            for k, v in desc_tensors.items():
                batched_descs[k] = v.unsqueeze(0).to(self.device)

            num_moves = torch.tensor([len(descs)], dtype=torch.long, device=self.device)

            logits = self.model.score_moves(outputs, batched_descs, num_moves)  # [1, M]
            logits = logits.squeeze(0)  # [M]

            # Get top-k predictions
            k = min(top_k, logits.shape[0])
            top_indices = logits.topk(k).indices.tolist()

            hit = any(idx in expected for idx in top_indices)
            if hit:
                overall_correct += 1
                results_by_cat[category]["correct"] += 1

            overall_total += 1
            results_by_cat[category]["total"] += 1

        per_cat = {}
        for cat, counts in results_by_cat.items():
            t = counts["total"]
            c = counts["correct"]
            per_cat[cat] = {
                "accuracy": c / t if t > 0 else 0,
                "correct": c,
                "total": t,
            }

        return {
            "accuracy": overall_correct / overall_total if overall_total > 0 else 0,
            "correct": overall_correct,
            "total": overall_total,
            "per_category": per_cat,
        }
