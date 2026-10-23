"""CPU contracts for the actual main-host geometry and ranking implementation.

Fixtures are synthetic tensors and temporary tiny heads/CDFs. These tests do
not load released detector weights, access a dataset, or run CUDA graphs.
"""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import torch
from torch import nn

from engine.deim.box_ops import generalized_box_iou, generalized_box_iou_aligned
from engine.deim.candidate_value import UMR, UMRHead, build_image_features, box_cxcywh_to_xyxy
from engine.deim.competition_mapping import apply_qccr, compute_competition_risk, compute_pairwise_iou
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.postprocessor import PostProcessor


class MainMethodContracts(unittest.TestCase):
    def make_criterion(self, directory):
        path = Path(directory) / 'difficulty.json'
        path.write_text(json.dumps(dict(area_640_sorted=[1000., 20000.],
                                        aspect_difficulty_sorted=[0., 1.],
                                        max_ioa_sorted=[0., 1.])))
        return DEIMCriterion(matcher=nn.Identity(), weight_dict={}, losses=[],
                             num_classes=5, ufd_enabled=True, ufd_cdf_path=str(path),
                             ufd_total_epochs=10, ufd_k=2, ufd_lambda_max=.75,
                             ufd_alpha_u_max=0.)

    def inputs(self, classes):
        raw = torch.arange(4 * classes, dtype=torch.float32).reshape(1, 4, classes) / 10 - 1.
        boxes = torch.tensor([[[.5, .5, .4, .4], [.5, .5, .3, .3],
                               [.25, .3, .2, .2], [.7, .8, .2, .1]]])
        absolute = box_cxcywh_to_xyxy(boxes) * torch.tensor([1280., 720., 1280., 720.])
        hidden = torch.linspace(-1., 1., 4 * 256).reshape(1, 4, 256)
        return raw, boxes, absolute, hidden

    def make_value(self, directory, classes, **options):
        path = Path(directory) / ('head%d.pt' % classes)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            torch.save(UMRHead(num_classes=classes).state_dict(), path)
        return UMR(num_classes=classes, num_queries=4, topk=3, checkpoint=path, **options).eval()

    def test_aligned_giou_matches_pairwise_diagonal(self):
        left = torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.], [1., 1., 4., 3.]])
        right = torch.tensor([[0., 0., 2., 2.], [2., 2., 3., 3.], [0., 0., 3., 3.]])
        torch.testing.assert_close(generalized_box_iou_aligned(left, right),
                                   generalized_box_iou(left, right).diagonal(), rtol=0, atol=0)

    def test_aligned_giou_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            generalized_box_iou_aligned(torch.zeros(2, 4), torch.zeros(3, 4))

    def test_empirical_cdf_uses_inclusive_right_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            criterion = self.make_criterion(directory)
            actual = criterion._ufd_empirical_cdf(torch.tensor([0., 1., 2., 4., 5.]),
                                                  torch.tensor([1., 2., 2., 4.]))
            torch.testing.assert_close(actual, torch.tensor([0., .25, .75, 1., 1.]), rtol=0, atol=0)

    def test_overlap_difficulty_is_directional_and_handles_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            criterion = self.make_criterion(directory)
            actual = criterion._ufd_max_ioa(torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.]]))
            torch.testing.assert_close(actual, torch.tensor([.25, 1.]), rtol=0, atol=0)
            self.assertEqual(criterion._ufd_max_ioa(torch.empty(0, 4)).numel(), 0)

    def test_difficulty_is_finite_bounded_and_detached(self):
        with tempfile.TemporaryDirectory() as directory:
            criterion = self.make_criterion(directory)
            boxes = torch.tensor([[.5, .5, .2, .2], [.5, .5, .1, .1]], requires_grad=True)
            difficulty = criterion._ufd_difficulty(boxes)
            self.assertTrue(torch.isfinite(difficulty).all())
            self.assertTrue(((difficulty >= 0) & (difficulty <= 1)).all())
            self.assertFalse(difficulty.requires_grad)
            self.assertTrue(torch.equal(criterion._ufd_difficulty(boxes), difficulty))

    def test_conflicts_backfill_without_duplicate_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            criterion = self.make_criterion(directory)
            queues = [[(3, .9, .9), (4, .8, .8)], [(3, .7, 1.1), (5, .6, .6)]]
            assigned = criterion._ufd_resolve_conflicts(queues)
            self.assertEqual(assigned[1][0][0], 3)
            self.assertEqual(assigned[0][0][0], 4)
            tokens = [token for group in assigned for token, _ in group]
            self.assertEqual(len(tokens), len(set(tokens)))
            self.assertEqual(assigned, criterion._ufd_resolve_conflicts(queues))

    def test_zero_auxiliary_weight_keeps_zero_gradients_connected(self):
        with tempfile.TemporaryDirectory() as directory:
            criterion = self.make_criterion(directory)
            logits = torch.zeros(1, 2, 132, requires_grad=True)
            empty = torch.empty(0, dtype=torch.long)
            losses = criterion._ufd_loss({'ufd_logits': logits}, [{'boxes': torch.empty(0, 4)}],
                                         [(empty, empty)], epoch=0, step=0, epoch_step=5)
            sum(losses.values()).backward()
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_active_auxiliary_loss_is_finite_and_backpropagates(self):
        with tempfile.TemporaryDirectory() as directory:
            criterion = self.make_criterion(directory)
            logits = torch.zeros(1, 1, 132, requires_grad=True)
            outputs = dict(ufd_logits=logits, ufd_ref_points=torch.tensor([[.5, .5, .2, .2]]),
                           ufd_valid_mask=torch.tensor([True]),
                           ufd_spatial_shapes=torch.tensor([[1, 1], [0, 0], [0, 0]]),
                           up=torch.tensor([.5]), reg_scale=torch.tensor([4.]))
            targets = [{'boxes': torch.tensor([[.5, .5, .2, .2]])}]
            indices = [(torch.tensor([0]), torch.tensor([0]))]
            losses = criterion._ufd_loss(outputs, targets, indices, epoch=1, step=0, epoch_step=5)
            total = sum(losses.values())
            self.assertTrue(torch.isfinite(total))
            total.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_competition_ties_keep_the_declared_query_order(self):
        logits = torch.tensor([[[3., -1.], [3., 2.], [1., 0.]]])
        boxes = torch.tensor([[[.5, .5, .4, .4]] * 3])
        risk = compute_competition_risk(logits, boxes)
        torch.testing.assert_close(risk, torch.tensor([[[0., 1.], [1., 0.], [1., 1.]]]), rtol=0, atol=0)

    def test_cached_pairwise_iou_matches_direct_computation(self):
        raw, boxes, _, _ = self.inputs(5)
        direct = compute_competition_risk(raw, boxes)
        cached = compute_competition_risk(raw, boxes, iou_matrix=compute_pairwise_iou(boxes))
        torch.testing.assert_close(direct, cached, rtol=0, atol=0)

    def test_mapping_formula_does_not_mutate_inputs(self):
        logits = torch.tensor([[-10., 0., 10.]])
        risk = torch.tensor([[0., .5, 1.]])
        before = logits.clone()
        mapped = apply_qccr(logits, risk, .1, 1.2, .8, .3)
        expected = .1 - 1.2 * torch.nn.functional.softplus(-logits) + .8 * torch.nn.functional.softplus(logits) - .3 * risk
        torch.testing.assert_close(mapped, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(logits, before))

    def test_features_support_both_dataset_class_counts(self):
        for classes in (3, 5):
            with self.subTest(classes=classes):
                raw, boxes, absolute, hidden = self.inputs(classes)
                risk = compute_competition_risk(raw, boxes)
                calibrated = apply_qccr(raw, risk, .1, 1.1, .9, .2)
                flat, features, labels, selected = build_image_features(
                    raw[0], boxes[0], absolute[0], hidden[0], calibrated[0],
                    calibrated[0].sigmoid(), risk[0], K=3)
                self.assertEqual(features.shape, (3, 25))
                self.assertEqual(selected.shape, (3, 256))
                self.assertTrue(torch.isfinite(features).all())
                self.assertTrue(torch.equal(labels, flat % classes))
                self.assertTrue(torch.equal(selected, hidden[0, flat // classes]))

    def test_value_heads_support_both_class_counts_without_changing_boxes(self):
        for classes in (3, 5):
            with self.subTest(classes=classes), tempfile.TemporaryDirectory() as directory:
                value = self.make_value(directory, classes)
                inputs = self.inputs(classes)
                before = tuple(tensor.clone() for tensor in inputs)
                output = value(*inputs)
                self.assertEqual(output.shape, inputs[0].shape)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(all(torch.equal(left, right) for left, right in zip(inputs, before)))
                self.assertFalse(any(parameter.requires_grad for parameter in value.head.parameters()))

    def test_zero_value_residual_recovers_the_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_value(directory, 3, beta=0., beta0=.1, beta_positive=1.1,
                                    beta_negative=.9, duplicate_penalty=.2)
            inputs = self.inputs(3)
            expected = apply_qccr(inputs[0], compute_competition_risk(inputs[0], inputs[1]), .1, 1.1, .9, .2)
            torch.testing.assert_close(value(*inputs), expected, rtol=0, atol=0)

    def test_value_wrapper_rejects_wrong_class_count(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.make_value(directory, 5)
            with self.assertRaisesRegex(ValueError, 'expected 5 classes'):
                value(*self.inputs(3))

    def test_postprocessor_scales_rectangular_images_width_then_height(self):
        for classes in (3, 5):
            with self.subTest(classes=classes):
                logits = torch.full((1, 2, classes), -5.)
                logits[0, 0, 0], logits[0, 1, classes - 1] = 4., 3.
                boxes = torch.tensor([[[.5, .25, .5, .25], [.25, .75, .25, .25]]])
                before = boxes.clone()
                result = PostProcessor(num_classes=classes, num_top_queries=2).eval()(
                    dict(pred_logits=logits, pred_boxes=boxes), torch.tensor([[1280., 720.]]))[0]
                torch.testing.assert_close(result['boxes'], torch.tensor([[320., 90., 960., 270.],
                                                                         [160., 450., 480., 630.]]), rtol=0, atol=0)
                self.assertTrue(torch.equal(result['labels'], torch.tensor([0, classes - 1])))
                self.assertTrue(torch.equal(boxes, before))

    def test_capture_records_keep_the_documented_height_width_schema(self):
        tree = ast.parse((SOURCE / 'tools/calibration/common/capture_train_val.py').read_text())
        orders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == 'orig_size':
                    if isinstance(value, ast.Call) and value.args and isinstance(value.args[0], ast.List):
                        orders.append([item.id for item in value.args[0].elts if isinstance(item, ast.Name)])
        self.assertEqual(orders, [['height', 'width']])


if __name__ == '__main__':
    unittest.main()
