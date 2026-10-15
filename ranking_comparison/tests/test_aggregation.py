import unittest

from aggregate_results import _fingerprint, _metric


class AggregationTests(unittest.TestCase):
    def test_threshold_metrics_are_read_from_nested_coco_payload(self):
        metrics = {"AP": 78.0, "threshold_ap": {"AP85": 71.0}}
        self.assertEqual(_metric(metrics, "AP"), 78.0)
        self.assertEqual(_metric(metrics, "AP85"), 71.0)

    def test_soft_nms_duplicate_runs_use_semantic_parameters(self):
        base = {
            "split": "val",
            "capture_sha256": "capture",
            "method": "soft_nms_linear",
        }
        individual = {
            **base,
            "parameters": {"iou_threshold": 0.5, "sigma": 0.5, "formal_source": None},
        }
        sweep = {
            **base,
            "parameters": {"grid_value": 0.5, "iou_threshold": 0.5, "sigma": 0.5},
        }
        self.assertEqual(_fingerprint(individual), _fingerprint(sweep))

    def test_iou_regressor_fingerprint_ignores_absolute_parent_directory(self):
        base = {
            "split": "val",
            "capture_sha256": "capture",
            "method": "iou_reg_rescore",
        }
        first = {
            **base,
            "parameters": {
                "iou_reg_checkpoint": "/server/a/iou_regressor_seed123.pt",
                "iou_reg_metadata": "/server/a/iou_regressor_seed123.metadata.json",
                "iou_reg_device": "cpu",
            },
        }
        second = {
            **base,
            "parameters": {
                "iou_reg_checkpoint": "/server/b/iou_regressor_seed123.pt",
                "iou_reg_metadata": "/server/b/iou_regressor_seed123.metadata.json",
                "iou_reg_device": "cuda",
            },
        }
        self.assertEqual(_fingerprint(first), _fingerprint(second))


if __name__ == "__main__":
    unittest.main()
