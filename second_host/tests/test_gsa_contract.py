import unittest

from _bootstrap import ROOT  # noqa: F401
import torch
from torch import nn

from src.zoo.rtdetr.hilor_gsa import HiLoRSetCriterion
from src.zoo.rtdetr.rtdetr_criterion import SetCriterion


class FixedMatcher(nn.Module):
    def forward(self, outputs, targets):
        return [
            (
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
            )
            for _ in targets
        ]


def build_outputs(requires_grad=False):
    logits = torch.tensor(
        [[[0.4, -0.2], [0.1, -0.4], [-0.5, 0.2], [-0.8, -0.7]]],
        requires_grad=requires_grad,
    )
    # q0 has zero paired IoU, q1 is an exact residual localization.
    boxes = torch.tensor(
        [[[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2],
          [0.8, 0.8, 0.1, 0.1], [0.2, 0.8, 0.1, 0.1]]]
    )
    return {"pred_logits": logits, "pred_boxes": boxes}


TARGETS = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]


class GSAContractTest(unittest.TestCase):
    def make_hilor(self, delta=0.02):
        return HiLoRSetCriterion(
            matcher=FixedMatcher(),
            weight_dict={"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
            losses=["vfl", "boxes"],
            alpha=0.75,
            gamma=2.0,
            num_classes=2,
            lcct_enabled=True,
            lcct_warmup_epoch=0,
            lcct_delta=delta,
            lcct_continuous_handover=True,
            lcct_transition_width=0.02,
            lcct_ramp_epochs=1,
            ufd_enabled=False,
            ufd_alpha_u_max=0.0,
        )

    def test_residual_query_is_native_unmatched_and_gain_is_absolute(self):
        criterion = self.make_hilor()
        criterion.train()
        outputs = build_outputs()
        indices = criterion.matcher(outputs, TARGETS)
        criterion._build_handover_pairs(outputs, TARGETS, indices, epoch=1)
        self.assertEqual(len(criterion._hilor_handover_pairs), 1)
        pair = criterion._hilor_handover_pairs[0]
        self.assertEqual(int(pair["native_queries"][0]), 0)
        self.assertEqual(int(pair["residual_queries"][0]), 1)
        gain = float((pair["residual_iou"] - pair["native_iou"])[0])
        self.assertAlmostEqual(gain, 1.0, places=6)

    def test_gain_gate_is_strict(self):
        criterion = self.make_hilor(delta=1.0)
        criterion.train()
        outputs = build_outputs()
        criterion._build_handover_pairs(
            outputs, TARGETS, criterion.matcher(outputs, TARGETS), epoch=1
        )
        self.assertEqual(criterion._hilor_handover_pairs, [])
        self.assertEqual(criterion._hilor_last_diag["active_count"], 0)

    def test_first_ramp_epoch_is_exact_native_loss(self):
        native = SetCriterion(
            matcher=FixedMatcher(),
            weight_dict={"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
            losses=["vfl", "boxes"],
            alpha=0.75,
            gamma=2.0,
            num_classes=2,
        )
        hilor = self.make_hilor()
        native.train()
        hilor.train()
        outputs = build_outputs()
        native_losses = native(outputs, TARGETS)
        hilor_losses = hilor(outputs, TARGETS, epoch=0, step=0, epoch_step=5)
        self.assertEqual(native_losses.keys(), hilor_losses.keys())
        for key in native_losses:
            self.assertTrue(torch.equal(native_losses[key], hilor_losses[key]), key)

    def test_nonzero_boundary_entropy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "alpha_u_max must be 0"):
            HiLoRSetCriterion(
                matcher=FixedMatcher(),
                weight_dict={"loss_vfl": 1.0},
                losses=["vfl"],
                num_classes=2,
                ufd_enabled=False,
                ufd_alpha_u_max=0.5,
            )


if __name__ == "__main__":
    unittest.main()
