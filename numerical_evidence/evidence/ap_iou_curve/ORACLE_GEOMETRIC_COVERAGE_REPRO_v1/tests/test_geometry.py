import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle_geometry import best_iou, cxcywh_to_pixel_xyxy, summarize


class OracleGeometryTest(unittest.TestCase):
    def test_identity_and_disjoint(self):
        values, ids = best_iou([[0, 0, 1, 1], [2, 2, 3, 3]], [[0, 0, 1, 1], [4, 4, 5, 5]])
        np.testing.assert_array_equal(values, [1, 0])
        np.testing.assert_array_equal(ids, [0, 0])

    def test_threshold_is_inclusive(self):
        values, _ = best_iou([[0, 0, 0.75, 1]], [[0, 0, 1, 1]])
        self.assertEqual(values[0], 0.75)
        self.assertEqual(summarize(values)["oracle_coverage_iou_ge_075"], 1)

    def test_one_query_can_cover_two_gt(self):
        values, ids = best_iou([[0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1]])
        np.testing.assert_array_equal(ids, [0, 0])
        np.testing.assert_array_equal(values, [1, 1])

    def test_complete_pool_not_only_first_query(self):
        values, ids = best_iou([[9, 9, 10, 10], [0, 0, 1, 1]], [[0, 0, 1, 1]])
        self.assertEqual(values[0], 1)
        self.assertEqual(ids[0], 1)

    def test_gt_weighted_not_image_weighted(self):
        self.assertEqual(summarize([1, 0, 0])["mean_best_iou"], 1 / 3)
        self.assertEqual(summarize([1, 0, 0])["median_best_iou"], 0)

    def test_coordinate_conversion_no_clipping(self):
        np.testing.assert_allclose(cxcywh_to_pixel_xyxy([[0.0, 0.5, 0.2, 1.0]], 640, 320),
                                   [[-64, 0, 64, 320]])

    def test_invalid_and_empty_pools_rejected(self):
        with self.assertRaises(ValueError):
            best_iou([[0, 0, 1, 1]], [[1, 0, 0, 1]])
        with self.assertRaises(ValueError):
            best_iou(np.empty((0, 4)), [[0, 0, 1, 1]])
        with self.assertRaises(ValueError):
            best_iou([[0, 0, float("nan"), 1]], [[0, 0, 1, 1]])

    def test_reversed_prediction_is_not_flipped_or_removed(self):
        values, ids = best_iou([[1, 1, 0, 0], [0, 0, 0.75, 1]], [[0, 0, 1, 1]])
        self.assertEqual(values[0], 0.75)
        self.assertEqual(ids[0], 1)
        values, ids = best_iou([[1, 1, 0, 0]], [[0, 0, 1, 1]])
        self.assertEqual(values[0], 0)
        self.assertEqual(ids[0], 0)


if __name__ == "__main__":
    unittest.main()
