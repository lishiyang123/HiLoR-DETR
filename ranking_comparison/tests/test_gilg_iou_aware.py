import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from nearest_neighbor_baselines.candidate_adapter import CandidateSet  # noqa: E402
from nearest_neighbor_baselines.gilg_iou_aware import (  # noqa: E402
    ConditionalBetaCalibrator,
    gilg_scores,
    greedy_tp_labels,
    minimum_jaccard_to_higher,
    official_near_duplicate_keep,
)


def candidates(boxes, scores, labels=None):
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if labels is None:
        labels = np.zeros(len(scores), dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    return CandidateSet(
        image_id=1,
        boxes_xyxy=boxes,
        raw_logits=np.log(scores / (1.0 - scores)),
        scores=scores,
        labels=labels,
        query_ids=np.arange(len(scores)),
        candidate_ids=np.arange(len(scores)),
        image_height=20,
        image_width=20,
    )


class GilgFeatureTests(unittest.TestCase):
    def test_minimum_jaccard_uses_only_higher_score_same_class(self):
        sample = candidates(
            [[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]],
            [0.9, 0.8, 0.7],
            [0, 0, 1],
        )
        result = minimum_jaccard_to_higher(sample)
        self.assertEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 1.0 - 81.0 / 119.0)
        self.assertEqual(result[2], 1.0)

    def test_reference_near_duplicate_prefilter_is_explicit(self):
        sample = candidates(
            [[0, 0, 10, 10], [0, 0, 10, 10], [15, 15, 20, 20]],
            [0.9, 0.8, 0.7],
        )
        np.testing.assert_array_equal(
            official_near_duplicate_keep(sample, 0.99), [True, False, True]
        )
        np.testing.assert_array_equal(
            official_near_duplicate_keep(sample, None), [True, True, True]
        )

    def test_tp_labels_are_score_ordered_one_to_one(self):
        sample = candidates(
            [[0, 0, 10, 10], [0, 0, 10, 10], [10, 10, 20, 20]],
            [0.9, 0.8, 0.7],
        )
        result = greedy_tp_labels(
            sample,
            np.asarray([[0, 0, 10, 10], [10, 10, 20, 20]]),
            np.asarray([0, 0]),
            iou_threshold=0.5,
        )
        np.testing.assert_array_equal(result, [1.0, 0.0, 1.0])


class ConditionalBetaTests(unittest.TestCase):
    def test_fit_is_deterministic_and_serializable(self):
        rng = np.random.default_rng(4)
        score = rng.uniform(0.02, 0.98, size=120)
        jaccard = rng.uniform(0.02, 0.98, size=120)
        target = ((score > 0.55) & (jaccard > 0.35)).astype(float)
        inputs = np.column_stack((score, jaccard))

        first = ConditionalBetaCalibrator().fit(inputs, target, seed=9, maxiter=80)
        second = ConditionalBetaCalibrator().fit(inputs, target, seed=9, maxiter=80)
        np.testing.assert_allclose(first.weights, second.weights, rtol=0, atol=1e-10)
        self.assertAlmostEqual(first.bias, second.bias, places=10)
        self.assertLess(first.fit_summary["final_nll"], 0.69)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            first.save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model"], "conditionally_dependent_beta")
            loaded = ConditionalBetaCalibrator.load(path)
            np.testing.assert_allclose(first.transform(inputs), loaded.transform(inputs))

    def test_gilg_scores_do_not_change_candidate_identity(self):
        sample = candidates(
            [[0, 0, 10, 10], [1, 1, 11, 11], [15, 15, 20, 20]],
            [0.9, 0.8, 0.7],
        )
        model = ConditionalBetaCalibrator(
            weights=np.asarray([1.5] * 12),
            bias=0.0,
        )
        scores, active, feature = gilg_scores(
            sample, model, near_duplicate_iou=None
        )
        self.assertEqual(scores.shape, sample.scores.shape)
        self.assertTrue(np.all(active))
        self.assertEqual(feature.shape, sample.scores.shape)
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))


if __name__ == "__main__":
    unittest.main()
