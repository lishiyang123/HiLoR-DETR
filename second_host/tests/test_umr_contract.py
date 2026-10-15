import tempfile
import unittest
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
import torch

from src.zoo.rtdetr.hilor_umr import UMR, UMRHead


class UMRContractTest(unittest.TestCase):
    def test_optimized_donor_checkpoint_names_are_accepted(self):
        torch.manual_seed(19)
        head = UMRHead(num_classes=2)
        reverse_names = {
            "hidden_projection.": "h_proj.",
            "class_embedding.": "cls_emb.",
            "input_projection.": "in_proj.",
            "norm1.": "ln1.",
            "norm2.": "ln2.",
            "output.": "out.",
        }
        donor_state = {}
        for key, value in head.state_dict().items():
            donor_key = key
            for host_prefix, donor_prefix in reverse_names.items():
                if key.startswith(host_prefix):
                    donor_key = donor_prefix + key[len(host_prefix):]
                    break
            donor_state[donor_key] = value

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "umr.pt"
            torch.save(donor_state, checkpoint)
            umr = UMR(
                checkpoint=str(checkpoint),
                num_classes=2,
                num_queries=4,
                topk=4,
                duplicate_penalty=0.2,
            )
            logits = torch.randn(1, 4, 2)
            boxes = torch.tensor(
                [[[0.3, 0.3, 0.2, 0.2], [0.35, 0.35, 0.2, 0.2],
                  [0.7, 0.7, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]]]
            )
            scale = torch.tensor([640.0, 640.0, 640.0, 640.0])
            xyxy = torch.stack(
                (
                    boxes[..., 0] - boxes[..., 2] / 2,
                    boxes[..., 1] - boxes[..., 3] / 2,
                    boxes[..., 0] + boxes[..., 2] / 2,
                    boxes[..., 1] + boxes[..., 3] / 2,
                ),
                dim=-1,
            ) * scale
            hidden = torch.randn(1, 4, 256)
            output = umr(logits, boxes, xyxy, hidden)
            self.assertEqual(tuple(output.shape), (1, 4, 2))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(all(not parameter.requires_grad for parameter in umr.head.parameters()))


if __name__ == "__main__":
    unittest.main()
