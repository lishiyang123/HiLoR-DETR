import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from nearest_neighbor_baselines.candidate_adapter import (  # noqa: E402
    candidates_from_record,
    pairwise_iou,
)


class CandidateAdapterTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "image_id": 7,
            "pred_logits": np.asarray([[0.0, 1.0], [2.0, -1.0]]),
            "pred_boxes": np.asarray(
                [[0.25, 0.25, 0.2, 0.4], [0.75, 0.75, 0.4, 0.2]]
            ),
            "orig_size": np.asarray([100, 200]),
        }

    def test_expands_every_query_class_pair(self):
        candidates = candidates_from_record(self.record)
        self.assertEqual(len(candidates.scores), 4)
        np.testing.assert_array_equal(candidates.query_ids, [0, 0, 1, 1])
        np.testing.assert_array_equal(candidates.labels, [0, 1, 0, 1])
        np.testing.assert_array_equal(candidates.candidate_ids, [0, 1, 2, 3])
        np.testing.assert_allclose(
            candidates.boxes_xyxy,
            [[30, 5, 70, 45], [30, 5, 70, 45], [110, 65, 190, 85], [110, 65, 190, 85]],
        )

    def test_global_topk_is_deterministic_on_ties(self):
        record = dict(self.record)
        record["pred_logits"] = np.zeros((2, 2))
        candidates = candidates_from_record(record)
        np.testing.assert_array_equal(candidates.topk(k=3), [0, 1, 2])

    def test_pairwise_iou_handles_invalid_area(self):
        boxes = np.asarray([[0, 0, 2, 2], [1, 1, 3, 3], [1, 1, 1, 2]])
        actual = pairwise_iou(boxes, boxes)
        self.assertAlmostEqual(actual[0, 1], 1.0 / 7.0)
        self.assertEqual(actual[2, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
