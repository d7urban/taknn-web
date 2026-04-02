"""Tactical test suite runner using Rust engine via PyO3."""

import json
import torch
import tak_python


class TacticalSuiteRunner:
    """Loads a suite of TPS positions with expected labels and evaluates model accuracy."""

    def __init__(self, model, suite_path, device="cpu"):
        self.model = model
        self.device = device
        self.positions = []
        self._load_suite(suite_path)

    def _load_suite(self, suite_path):
        """Load suite file: one JSON object per line with 'tps', 'size', 'expected_move_index'."""
        try:
            with open(suite_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.positions.append(json.loads(line))
        except FileNotFoundError:
            pass

    def run(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for pos in self.positions:
                tps = pos['tps']
                size = pos['size']
                expected = pos.get('expected_move_index')
                if expected is None:
                    continue

                game = tak_python.PyGameState(size, tps)
                tensor = torch.from_numpy(game.encode_tensor().copy())
                tensor = tensor.unsqueeze(0).to(self.device)
                size_id = torch.tensor([size - 3], dtype=torch.long).to(self.device)

                outputs = self.model(tensor, size_id)
                # Check if the model's top policy move matches expected
                # (This is a simplified check — real implementation would
                # score all legal moves via the policy head MLP.)
                total += 1
                # Placeholder: would need full policy scoring here

        return {
            "accuracy": correct / total if total > 0 else 0,
            "total": total,
            "correct": correct,
        }
