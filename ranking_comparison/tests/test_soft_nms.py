import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from nearest_neighbor_baselines.candidate_adapter import CandidateSet  # noqa: E402
from nearest_neighbor_baselines.soft_nms import soft_nms_scores  # noqa: E402


def make_candidates():
    return CandidateSet(
        image_id=1,
        boxes_xyxy=np.asarray(
            [
                [0, 0, 10, 10],
                [1, 1, 11, 11],
                [0, 0, 10, 10],
            ],
            dtype=float,
        ),
        raw_logits=np.asarray([3.0, 2.0, 1.0]),
        scores=np.asarray([0.9, 0.8, 0.7]),
        labels=np.asarray([0, 0, 1]),
        query_ids=np.asarray([0, 1, 2]),
        candidate_ids=np.asarray([0, 1, 2]),
        image_height=20,
        image_width=20,
    )


class SoftNMSTests(unittest.TestCase):
    def test_linear_decays_only_overlapping_same_class_candidate(self):
        candidates = make_candidates()
        result = soft_nms_scores(
            candidates, method="linear", iou_threshold=0.5, sigma=0.5
        )
        overlap = 81.0 / 119.0
        self.assertAlmostEqual(result[0], 0.9)
        self.assertAlmostEqual(result[1], 0.8 * (1.0 - overlap))
        self.assertAlmostEqual(result[2], 0.7)

    def test_gaussian_matches_published_decay(self):
        candidates = make_candidates()
        result = soft_nms_scores(
            candidates, method="gaussian", iou_threshold=0.5, sigma=0.5
        )
        overlap = 81.0 / 119.0
        self.assertAlmostEqual(result[1], 0.8 * np.exp(-(overlap**2) / 0.5))
        self.assertAlmostEqual(result[2], 0.7)


if __name__ == "__main__":
    unittest.main()
