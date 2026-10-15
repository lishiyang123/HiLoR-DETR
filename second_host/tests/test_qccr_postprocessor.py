import unittest

from _bootstrap import ROOT  # noqa: F401
import torch

from src.zoo.rtdetr.hilor_postprocessor import HiLoRRTDETRPostProcessor
from src.zoo.rtdetr.hilor_qccr import (
    QueryCompetitionConditionalRecalibration,
    compute_competition_risk,
)
from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor


class QCCRPostprocessorTest(unittest.TestCase):
    def test_disabled_adapter_is_exact_native_bypass(self):
        torch.manual_seed(7)
        outputs = {
            "pred_logits": torch.randn(2, 4, 3),
            "pred_boxes": torch.rand(2, 4, 4),
        }
        sizes = torch.tensor([[640, 640], [480, 720]])
        native = RTDETRPostProcessor(
            num_classes=3, use_focal_loss=True, num_top_queries=5
        )
        adapter = HiLoRRTDETRPostProcessor(
            num_classes=3,
            use_focal_loss=True,
            num_top_queries=5,
            qccr_enabled=False,
            umr_enabled=False,
        )
        native_results = native(outputs, sizes)
        adapter_results = adapter(outputs, sizes)
        for native_image, adapter_image in zip(native_results, adapter_results):
            self.assertEqual(native_image.keys(), adapter_image.keys())
            for key in native_image:
                self.assertTrue(torch.equal(native_image[key], adapter_image[key]), key)

    def test_equal_logit_tie_uses_smaller_query_index(self):
        logits = torch.tensor([[[2.0], [2.0], [1.0]]])
        boxes = torch.tensor(
            [[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4],
              [0.1, 0.1, 0.1, 0.1]]]
        )
        risk = compute_competition_risk(logits, boxes)
        self.assertEqual(float(risk[0, 0, 0]), 0.0)
        self.assertEqual(float(risk[0, 1, 0]), 1.0)
        self.assertEqual(float(risk[0, 2, 0]), 0.0)

    def test_active_postprocessor_rejects_unbound_artifacts(self):
        with self.assertRaisesRegex(ValueError, "bound inference manifest"):
            HiLoRRTDETRPostProcessor(
                num_classes=3,
                num_top_queries=5,
                qccr_enabled=True,
            )

    def test_qccr_active_path_is_fp32(self):
        module = QueryCompetitionConditionalRecalibration()
        logits = torch.tensor([[[1.0], [0.0]]], dtype=torch.float16)
        boxes = torch.tensor(
            [[[0.5, 0.5, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]]],
            dtype=torch.float16,
        )
        self.assertEqual(module(logits, boxes).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
