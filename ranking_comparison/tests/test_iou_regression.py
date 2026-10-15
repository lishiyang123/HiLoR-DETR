import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from nearest_neighbor_baselines.iou_regression import (
    IoURegressorHead,
    IoURegressorReplay,
    build_iou_targets,
)


class IoURegressionTests(unittest.TestCase):
    def test_targets_use_max_same_class_iou(self):
        record = {
            "pred_boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]),
            "orig_size": torch.tensor([100, 100]),
            "gt_boxes": torch.tensor([[25.0, 25.0, 75.0, 75.0]]),
            "gt_labels": torch.tensor([2]),
            "gt_iscrowd": torch.tensor([0]),
        }
        targets = build_iou_targets(record, num_classes=5)
        self.assertAlmostEqual(float(targets[0, 2]), 1.0, places=6)
        self.assertEqual(float(targets[0, 1]), 0.0)
        self.assertEqual(float(targets[1, 2]), 0.0)

    def test_replay_multiplies_native_score_by_predicted_iou(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = IoURegressorHead(hidden_dim=4, num_classes=2)
            for parameter in model.parameters():
                torch.nn.init.zeros_(parameter)
            weight = root / "head.pt"
            torch.save(model.state_dict(), weight)
            import hashlib, json
            digest = hashlib.sha256(weight.read_bytes()).hexdigest()
            metadata = root / "head.json"
            metadata.write_text(json.dumps({
                "hidden_dim": 4,
                "num_classes": 2,
                "source_checkpoint_sha256": "detector",
                "weight_sha256": digest,
            }), encoding="utf-8")
            replay = IoURegressorReplay(
                checkpoint=weight,
                metadata=metadata,
                capture_checkpoint_sha256="detector",
            )
            scores = replay.scores({
                "pred_logits": torch.zeros(1, 2),
                "hL": torch.zeros(1, 4),
            })
            np.testing.assert_allclose(scores, np.asarray([0.25, 0.25]), atol=1e-7)


if __name__ == "__main__":
    unittest.main()

