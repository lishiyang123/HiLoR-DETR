"""Offline replay must bind geometry inputs without executing detector source."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_analysis as analysis


class ReplayInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch_root = patch.object(analysis, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.write("protocol.json", {"image_count": 2, "gt_count": 2})
        self.write("annotation.json", {
            "images": [{"id": 10, "width": 640, "height": 480},
                       {"id": 20, "width": 320, "height": 240}],
            "annotations": [{"id": 1, "image_id": 10, "iscrowd": 0},
                            {"id": 2, "image_id": 20, "iscrowd": 0}]})
        ann = self.root / "annotation.json"
        ann_sha = analysis.sha256(ann)
        self.patch_ann = patch.object(analysis, "ANNOTATION_SHA", ann_sha)
        self.patch_ann.start()
        self.addCleanup(self.patch_ann.stop)
        self.bindings = {"annotation": str(ann), "protected_inputs": {str(ann): ann_sha},
                         "source_root": str(self.root / "unused_source"),
                         "source_hashes": {"old_name.py": "historical"},
                         "config_hashes": {"cdw_b0_test.yml": "b0", "cdw_native_test.yml": "gccs"},
                         "runs": []}
        # Neither current source nor a current head metadata file is an offline input.
        self.bindings["protected_inputs"][str(self.root / "unused.metadata.json")] = "historical"
        for name in ("DEIM", "GCCS"):
            for seed in (0, 1, 2):
                folder = self.root / f"captures/{name}/seed{seed}"
                folder.mkdir(parents=True)
                checkpoint = folder / "checkpoint.pth"
                checkpoint.write_bytes(f"fixture-{name}-{seed}".encode())
                digest = analysis.sha256(checkpoint)
                self.bindings["protected_inputs"][str(checkpoint)] = digest
                self.bindings["runs"].append({"model": name, "seed": seed,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": digest})
                self.save_pool(folder)
                self.write(str(folder.relative_to(self.root) / "CAPTURE_SUMMARY.json"), {
                    "model": name, "seed": seed, "checkpoint_sha256": digest,
                    "config_sha256": "b0" if name == "DEIM" else "gccs",
                    "weight_source": "ema.module", "strict_load": "PASS", "images": 2,
                    "queries_per_image": 300, "precision": "FP32", "batch_size": 8,
                    "iou_precision": "float64", "postprocessing_used_for_oracle": False,
                    "capture_sha256": analysis.sha256(folder / "full_query_boxes.npz")})
        self.write("bindings.json", self.bindings)
        self.refresh_manifest()

    def write(self, relative, data):
        (self.root / relative).write_text(json.dumps(data), encoding="utf-8")

    def save_pool(self, folder, queries=300):
        raw = np.full((2, queries, 4), 0.25, dtype=np.float32)
        pixels = np.stack([analysis.cxcywh_to_pixel_xyxy(raw[0], 640, 480),
                           analysis.cxcywh_to_pixel_xyxy(raw[1], 320, 240)])
        np.savez_compressed(folder / "full_query_boxes.npz", image_ids=np.array([10, 20]),
                            pred_boxes_cxcywh_normalized=raw, pred_boxes_xyxy_pixels=pixels)

    def refresh_manifest(self):
        self.write("MANIFEST.sha256.json", {"files": {
            p.relative_to(self.root).as_posix(): analysis.sha256(p)
            for p in self.root.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256.json"}})

    def test_offline_replay_does_not_require_current_source_or_head(self):
        analysis.check_replay_bindings(self.bindings)

    def test_changed_annotation_is_rejected(self):
        (self.root / "annotation.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "annotation SHA"):
            analysis.check_replay_bindings(self.bindings)

    def test_changed_checkpoint_is_rejected(self):
        Path(self.bindings["runs"][0]["checkpoint"]).write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "checkpoint binding"):
            analysis.check_replay_bindings(self.bindings)

    def test_changed_capture_is_rejected(self):
        (self.root / "captures/DEIM/seed0/full_query_boxes.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "artifact SHA"):
            analysis.check_replay_bindings(self.bindings)

    def test_receipt_with_wrong_seed_is_rejected(self):
        rel = "captures/DEIM/seed0/CAPTURE_SUMMARY.json"
        receipt = analysis.read_json(self.root / rel)
        receipt["seed"] = 2
        self.write(rel, receipt)
        self.refresh_manifest()
        with self.assertRaisesRegex(RuntimeError, "capture receipt"):
            analysis.check_replay_bindings(self.bindings)

    def test_topk_truncated_capture_is_rejected(self):
        folder = self.root / "captures/DEIM/seed0"
        self.save_pool(folder, queries=100)
        rel = "captures/DEIM/seed0/CAPTURE_SUMMARY.json"
        receipt = analysis.read_json(self.root / rel)
        receipt["capture_sha256"] = analysis.sha256(folder / "full_query_boxes.npz")
        self.write(rel, receipt)
        self.refresh_manifest()
        with self.assertRaisesRegex(RuntimeError, "truncated full-query"):
            analysis.check_replay_bindings(self.bindings)


if __name__ == "__main__":
    unittest.main()
