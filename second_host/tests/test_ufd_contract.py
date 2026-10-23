import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
import torch
from torch import nn

from src.zoo.rtdetr.hilor_gsa import HiLoRSetCriterion


class EmptyMatcher(nn.Module):
    def forward(self, outputs, targets):
        return [
            (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
            for _ in targets
        ]


class UFDContractTest(unittest.TestCase):
    def make_criterion(self, cdf_path):
        return HiLoRSetCriterion(
            matcher=EmptyMatcher(),
            weight_dict={"loss_vfl": 1.0},
            losses=["vfl"],
            num_classes=2,
            lcct_enabled=False,
            ufd_enabled=True,
            ufd_cdf_path=str(cdf_path),
            ufd_total_epochs=10,
            ufd_k=2,
            ufd_alpha_u_max=0.0,
            ufd_lambda_max=0.75,
            ufd_reg_max=32,
        )

    def test_deterministic_conflict_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            cdf_path = Path(directory) / "cdf.json"
            cdf_path.write_text(
                json.dumps(
                    {
                        "area_640_sorted": [1.0, 2.0],
                        "aspect_difficulty_sorted": [0.0, 1.0],
                        "max_ioa_sorted": [0.0, 1.0],
                    }
                ),
                encoding="utf-8",
            )
            criterion = self.make_criterion(cdf_path)
            queues = [
                [(3, 0.9, 0.9), (4, 0.8, 0.8)],
                [(3, 0.7, 1.1), (5, 0.6, 0.6)],
            ]
            assigned = criterion._resolve_conflicts(queues)
            # GT1 wins token 3 by priority; GT0 deterministically backfills.
            self.assertEqual(assigned[1][0][0], 3)
            self.assertEqual(assigned[0][0][0], 4)
            flat = [token for image in assigned for token, _ in image]
            self.assertEqual(len(flat), len(set(flat)))

    def test_zero_lambda_returns_graph_connected_zeros(self):
        with tempfile.TemporaryDirectory() as directory:
            cdf_path = Path(directory) / "cdf.json"
            cdf_path.write_text(
                json.dumps(
                    {
                        "area_640_sorted": [1.0],
                        "aspect_difficulty_sorted": [0.0],
                        "max_ioa_sorted": [0.0],
                    }
                ),
                encoding="utf-8",
            )
            criterion = self.make_criterion(cdf_path)
            logits = torch.randn(1, 3, 132, requires_grad=True)
            losses = criterion._ufd_loss(
                {"ufd_logits": logits},
                [{"boxes": torch.empty(0, 4)}],
                epoch=0,
                step=0,
                epoch_step=5,
            )
            total = sum(losses.values())
            total.backward()
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_active_loss_is_finite_and_backpropagates(self):
        with tempfile.TemporaryDirectory() as directory:
            cdf_path = Path(directory) / "cdf.json"
            cdf_path.write_text(
                json.dumps(
                    {
                        "area_640_sorted": [1000.0, 20000.0],
                        "aspect_difficulty_sorted": [0.0, 1.0],
                        "max_ioa_sorted": [0.0, 1.0],
                    }
                ),
                encoding="utf-8",
            )
            criterion = self.make_criterion(cdf_path)
            logits = torch.randn(1, 1, 132, requires_grad=True)
            outputs = {
                "ufd_logits": logits,
                "ufd_ref_points": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "ufd_valid_mask": torch.tensor([True]),
                "ufd_spatial_shapes": torch.tensor([[1, 1], [0, 0], [0, 0]]),
                "ufd_up": torch.tensor([0.5]),
                "ufd_reg_scale": torch.tensor([4.0]),
            }
            targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
            losses = criterion._ufd_loss(
                outputs, targets, epoch=1, step=0, epoch_step=5
            )
            self.assertEqual(
                set(losses), {"loss_ufd_fgl", "loss_ufd_bbox", "loss_ufd_giou"}
            )
            total = sum(losses.values())
            self.assertTrue(torch.isfinite(total))
            total.backward()
            self.assertIsNotNone(logits.grad)
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_extreme_signed_offsets_keep_giou_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            cdf_path = Path(directory) / "cdf.json"
            cdf_path.write_text(
                json.dumps(
                    {
                        "area_640_sorted": [1000.0, 20000.0],
                        "aspect_difficulty_sorted": [0.0, 1.0],
                        "max_ioa_sorted": [0.0, 1.0],
                    }
                ),
                encoding="utf-8",
            )
            criterion = self.make_criterion(cdf_path)
            # Bin zero is the most negative signed offset. Selecting it for
            # all four sides used to decode an inverted box and trip the
            # generalized_box_iou assertion.
            logits = torch.full((1, 1, 132), -100.0, requires_grad=True)
            with torch.no_grad():
                logits.reshape(1, 1, 4, 33)[..., 0] = 100.0
            outputs = {
                "ufd_logits": logits,
                "ufd_ref_points": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "ufd_valid_mask": torch.tensor([True]),
                "ufd_spatial_shapes": torch.tensor([[1, 1], [0, 0], [0, 0]]),
                "ufd_up": torch.tensor([0.5]),
                "ufd_reg_scale": torch.tensor([4.0]),
            }
            targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
            losses = criterion._ufd_loss(
                outputs, targets, epoch=5, step=0, epoch_step=5
            )
            total = sum(losses.values())
            self.assertTrue(torch.isfinite(total))
            total.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_dense_aligned_giou_avoids_pairwise_matrix(self):
        boxes = torch.tensor([[0.1, 0.2, 0.7, 0.9]]).repeat(4097, 1)
        giou = HiLoRSetCriterion._aligned_giou(boxes, boxes)
        self.assertEqual(tuple(giou.shape), (4097,))
        self.assertTrue(torch.equal(giou, torch.ones_like(giou)))


if __name__ == "__main__":
    unittest.main()
