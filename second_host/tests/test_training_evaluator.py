"""Small CPU checks for the RT training/Val bbox evaluator."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

RELEASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE))
import reproduce_pipeline as pipeline
from coco_test_evaluation import fast_ground_truth


class TrainingEvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = RELEASE / 'benchmarks/02_RT_DETR_R18/source/RT-DETR_official/rtdetr_pytorch'
        sys.path.insert(0, str(source))
        from src.data import CocoEvaluator
        cls.native = CocoEvaluator
        cls.faster = pipeline.install_rt_evaluator(source)

    def gt(self):
        return fast_ground_truth(dict(info={}, images=[dict(id=1, width=100, height=100),
                                              dict(id=2, width=100, height=100)],
            categories=[dict(id=c, name=str(c)) for c in range(5)],
            annotations=[dict(id=1, image_id=1, category_id=0, bbox=[10, 10, 20, 20], area=400, iscrowd=0)]))

    def predictions(self):
        return {1: dict(boxes=torch.tensor([[10., 10., 30., 30.]]),
                        labels=torch.tensor([0]), scores=torch.tensor([.9])),
                2: dict(boxes=torch.empty((0, 4)), labels=torch.empty(0, dtype=torch.int64), scores=torch.empty(0))}

    def test_val_and_capture_use_same_class(self):
        from src import data
        from src.solver import det_engine
        self.assertIs(data.CocoEvaluator, self.faster)
        self.assertIs(det_engine.CocoEvaluator, self.faster)

    def test_fixed_predictions_equal_coco_reference(self):
        arrays = []
        with contextlib.redirect_stdout(io.StringIO()):
            for evaluator_type in (self.native, self.faster):
                evaluator = evaluator_type(self.gt(), ['bbox'])
                evaluator.update(self.predictions())
                evaluator.synchronize_between_processes()
                evaluator.accumulate()
                evaluator.summarize()
                arrays.append(evaluator.coco_eval['bbox'])
        for key in ('precision', 'recall', 'scores'):
            np.testing.assert_allclose(arrays[0].eval[key], arrays[1].eval[key], rtol=0, atol=1e-12)
        np.testing.assert_allclose(arrays[0].stats, arrays[1].stats, rtol=0, atol=1e-12)
        self.assertEqual(list(arrays[1].params.maxDets), [1, 10, 100])
        self.assertEqual(list(arrays[1].params.catIds), [0, 1, 2, 3, 4])
        self.assertEqual(list(arrays[1].params.imgIds), [1, 2])

    def test_empty_predictions_do_not_drop_images(self):
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator = self.faster(self.gt(), ['bbox'])
            empty = self.predictions()[2]
            evaluator.update({1: empty, 2: empty})
            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            evaluator.summarize()
        self.assertEqual(evaluator.img_ids, [1, 2])
        self.assertEqual(float(evaluator.coco_eval['bbox'].stats[0]), 0.)


if __name__ == '__main__':
    unittest.main()
