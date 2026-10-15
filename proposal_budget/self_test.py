#!/usr/bin/env python3
"""Pure structural and matching tests for the server package."""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path

from common import BUDGETS, IOU_THRESHOLDS, MAIN_BUDGET, MAIN_IOU
from evaluate import action_label, add_rates, bbox_outside_excursion, evaluate_image, iou_xywh, stable_rank


ROOT = Path(__file__).resolve().parent


def gt(annotation_id, category_id, bbox, crowd=False):
    return {"id": annotation_id, "category_id": category_id, "bbox": tuple(bbox), "iscrowd": crowd}


def detection(index, category_id, bbox, score):
    return {
        "image_id": 1,
        "category_id": category_id,
        "bbox": tuple(bbox),
        "score": score,
        "original_index": index,
    }


class SortingProposalBudgetTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            (ROOT / "protocol/ACTION_BUDGET_PROTOCOL.json").read_text(encoding="utf-8")
        )

    def test_python_sources_parse(self):
        for name in ("common.py", "evaluate.py", "manager.py", "self_test.py"):
            ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)

    def test_protocol_is_locked_to_requested_settings(self):
        self.assertEqual(BUDGETS, tuple(range(1, 11)))
        self.assertEqual(IOU_THRESHOLDS, (0.75, 0.80))
        self.assertEqual((MAIN_BUDGET, MAIN_IOU), (5, 0.75))
        self.assertEqual(self.protocol["score_threshold"], None)
        self.assertFalse(self.protocol["compute"]["training"])
        self.assertFalse(self.protocol["compute"]["inference"])
        self.assertFalse(self.protocol["compute"]["gpu_required"])
        self.assertFalse(self.protocol["release_write"])

    def test_iou(self):
        self.assertAlmostEqual(iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(iou_xywh((0, 0, 10, 10), (5, 0, 10, 10)), 1.0 / 3.0)
        self.assertEqual(iou_xywh((0, 0, 10, 10), (20, 20, 2, 2)), 0.0)

    def test_out_of_bounds_boxes_are_measured_without_clipping(self):
        self.assertEqual(bbox_outside_excursion((0, 0, 10, 10), 100, 100), 0.0)
        self.assertAlmostEqual(bbox_outside_excursion((-2.5, 0, 10, 10), 100, 100), 2.5)
        self.assertAlmostEqual(bbox_outside_excursion((95, 95, 8, 7), 100, 100), 3.0)
        self.assertIn("preserve the exported coordinates unchanged", self.protocol["input_audit"]["out_of_bounds_bbox_policy"])

    def test_degenerate_boxes_remain_actions_with_zero_iou(self):
        self.assertEqual(iou_xywh((0, 0, 0, 10), (0, 0, 10, 10)), 0.0)
        self.assertEqual(iou_xywh((0, 0, -1, 10), (0, 0, 10, 10)), 0.0)
        rows = [gt(1, 1, (0, 0, 10, 10))]
        snapshots = evaluate_image([detection(0, 1, (0, 0, 0, 10), 0.99)], rows, 0.75, (1,))
        self.assertEqual(snapshots[1]["background_actions"], 1)
        self.assertIn("assign IoU zero", self.protocol["input_audit"]["degenerate_bbox_policy"])

    def test_success_uses_highest_iou_then_smallest_annotation_id(self):
        rows = [
            gt(4, 1, (0, 0, 10, 10)),
            gt(3, 1, (0, 0, 10, 10)),
        ]
        label, matched_id = action_label(detection(0, 1, (0, 0, 10, 10), 0.9), rows, set(), 0.75)
        self.assertEqual(label, "success")
        self.assertEqual(matched_id, 3)

    def test_action_partition_and_precedence(self):
        rows = [
            gt(1, 1, (0, 0, 10, 10)),
            gt(2, 1, (20, 0, 10, 10)),
            gt(3, 2, (40, 0, 10, 10)),
        ]
        detections = [
            detection(0, 1, (0, 0, 10, 10), 0.99),
            detection(1, 1, (0, 0, 10, 10), 0.98),
            detection(2, 1, (20, 0, 7, 10), 0.97),
            detection(3, 1, (40, 0, 10, 10), 0.96),
            detection(4, 1, (60, 0, 10, 10), 0.95),
        ]
        snapshots = evaluate_image(detections, rows, 0.75, (1, 2, 3, 4, 5))
        final = snapshots[5]
        self.assertEqual(final["successful_actions"], 1)
        self.assertEqual(final["duplicate_actions"], 1)
        self.assertEqual(final["localization_insufficient_actions"], 1)
        self.assertEqual(final["category_error_actions"], 1)
        self.assertEqual(final["background_actions"], 1)
        rated = add_rates(final)
        self.assertAlmostEqual(rated["valid_action_rate"], 0.2)
        self.assertAlmostEqual(rated["wasted_action_rate"], 0.8)

    def test_crowd_is_excluded_and_empty_noncrowd_image_is_retained(self):
        rows = [gt(9, 1, (0, 0, 10, 10), crowd=True)]
        detections = [detection(index, 1, (0, 0, 10, 10), 1.0 - index * 0.01) for index in range(10)]
        snapshot = evaluate_image(detections, rows, 0.75, (1,))[1]
        self.assertEqual(snapshot["noncrowd_gt_count"], 0)
        self.assertEqual(snapshot["background_actions"], 1)

    def test_exact_score_ties_preserve_original_order(self):
        ranked = stable_rank([
            detection(7, 1, (20, 20, 2, 2), 0.5),
            detection(8, 1, (0, 0, 10, 10), 0.5),
        ])
        self.assertEqual([row["original_index"] for row in ranked], [7, 8])

    def test_no_release_install_action(self):
        manager = (ROOT / "manager.py").read_text(encoding="utf-8")
        self.assertNotIn('"install"', manager)
        self.assertIn('"release_write": False', manager)

    def test_payload_manifest(self):
        manifest = ROOT / "PAYLOAD_MANIFEST.sha256"
        self.assertTrue(manifest.is_file())
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, relative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
