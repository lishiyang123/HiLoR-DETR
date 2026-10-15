"""Data identity, full-pool capture and Top-K boundary regressions."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import torch
from _bootstrap import ROOT
from tools.capture import (
    TOPK_BOUNDARY_TIE_POLICY, audit_topk_boundary, config_provenance,
    enforce_topk_boundary_policy, sha256, sha256_lines, validate_split_binding,
)


def annotation_payload():
    return {
        "images": [{"id": 7, "file_name": "x.jpg", "width": 10, "height": 10}],
        "annotations": [],
        "categories": [{"id": index, "name": str(index)} for index in range(5)],
    }

def split_entry(annotation, images_dir, payload):
    image_ids = sorted(int(item["id"]) for item in payload["images"])
    file_names = sorted(str(item["file_name"]) for item in payload["images"])
    tree = hashlib.sha256()
    for file_name in file_names:
        tree.update(
            f"{sha256(images_dir / file_name)}  ./{file_name}\n".encode("utf-8")
        )
    return {
        "annotation": str(annotation.resolve()),
        "annotation_sha256": sha256(annotation),
        "images_dir": str(images_dir.resolve()),
        "image_count": 1,
        "instance_count": 0,
        "image_id_set_sha256": sha256_lines(image_ids),
        "file_name_set_sha256": sha256_lines(file_names),
        "image_tree_sha256": tree.hexdigest(),
    }

class CaptureTest(unittest.TestCase):

    def test_bound_val_is_accepted_and_test_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            (images / "x.jpg").write_bytes(b"bound image")
            payload = annotation_payload()
            val = root / "val.json"
            val.write_text(json.dumps(payload), encoding="utf-8")
            test_payload = annotation_payload()
            test_payload["images"][0]["id"] = 8
            test_payload["images"][0]["file_name"] = "test.jpg"
            test = root / "test.json"
            test.write_text(json.dumps(test_payload, indent=2), encoding="utf-8")
            lock = {
                "schema": "cdw1630-rtdetr-r18-hilor-data-split-lock-v1",
                "selection_eligible_splits": ["train", "val"],
                "category_ids": list(range(5)),
                "splits": {
                    "train": split_entry(val, images, payload),
                    "val": split_entry(val, images, payload),
                    "test": {
                        "annotation_sha256": sha256(test),
                        "image_id_set_sha256": sha256_lines([8]),
                        "selection_or_fitting_allowed": False,
                    },
                },
            }
            accepted = validate_split_binding("val", val, images, payload, lock)
            self.assertEqual(accepted["image_count"], 1)

            lock["splits"]["val"] = {
                **lock["splits"]["val"],
                "annotation": str(test.resolve()),
                "annotation_sha256": sha256(test),
                "image_id_set_sha256": sha256_lines([8]),
                "file_name_set_sha256": sha256_lines(["test.jpg"]),
            }
            with self.assertRaisesRegex(RuntimeError, "Test annotation"):
                validate_split_binding("val", test, images, test_payload, lock)

    def test_split_binding_rejects_image_content_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            image = images / "x.jpg"
            image.write_bytes(b"original")
            payload = annotation_payload()
            annotation = root / "val.json"
            annotation.write_text(json.dumps(payload), encoding="utf-8")
            test_payload = annotation_payload()
            test_payload["images"][0]["id"] = 8
            test_payload["images"][0]["file_name"] = "test.jpg"
            test = root / "test.json"
            test.write_text(json.dumps(test_payload), encoding="utf-8")
            lock = {
                "schema": "cdw1630-rtdetr-r18-hilor-data-split-lock-v1",
                "selection_eligible_splits": ["train", "val"],
                "category_ids": list(range(5)),
                "splits": {
                    "train": split_entry(annotation, images, payload),
                    "val": split_entry(annotation, images, payload),
                    "test": {
                        "annotation_sha256": sha256(test),
                        "image_id_set_sha256": sha256_lines([8]),
                    },
                },
            }
            image.write_bytes(b"drifted")
            with self.assertRaisesRegex(RuntimeError, "image-tree SHA-256"):
                validate_split_binding("val", annotation, images, payload, lock)

    def test_split_binding_rejects_extra_image_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            (images / "x.jpg").write_bytes(b"original")
            payload = annotation_payload()
            annotation = root / "val.json"
            annotation.write_text(json.dumps(payload), encoding="utf-8")
            test_payload = annotation_payload()
            test_payload["images"][0]["id"] = 8
            test = root / "test.json"
            test.write_text(json.dumps(test_payload), encoding="utf-8")
            binding = split_entry(annotation, images, payload)
            lock = {
                "schema": "cdw1630-rtdetr-r18-hilor-data-split-lock-v1",
                "selection_eligible_splits": ["train", "val"],
                "category_ids": list(range(5)),
                "splits": {
                    "train": binding,
                    "val": binding,
                    "test": {
                        "annotation_sha256": sha256(test),
                        "image_id_set_sha256": sha256_lines([8]),
                    },
                },
            }
            (images / "extra.jpg").write_bytes(b"extra")
            with self.assertRaisesRegex(RuntimeError, "extras=.*extra.jpg"):
                validate_split_binding("val", annotation, images, payload, lock)

    def test_topk_boundary_audit_accepts_unique_boundary(self):
        scores = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32)
        report = audit_topk_boundary(scores, topk=2)[0]
        self.assertFalse(report["boundary_tie"])
        self.assertEqual(report["equal_boundary_count"], 1)
        self.assertEqual(report["excluded_equal_boundary_count"], 0)
        self.assertAlmostEqual(report["boundary_margin"], 0.1, places=6)
        self.assertEqual(report["policy"], TOPK_BOUNDARY_TIE_POLICY)

    def test_topk_boundary_audit_flags_exact_straddling_tie(self):
        scores = torch.tensor([[0.9, 0.8, 0.8, 0.6]], dtype=torch.float32)
        report = audit_topk_boundary(scores, topk=2)[0]
        self.assertTrue(report["boundary_tie"])
        self.assertEqual(report["strictly_above_count"], 1)
        self.assertEqual(report["equal_boundary_count"], 2)
        self.assertEqual(report["selected_equal_boundary_count"], 1)
        self.assertEqual(report["excluded_equal_boundary_count"], 1)
        self.assertEqual(report["boundary_margin"], 0.0)
        self.assertIsNone(
            enforce_topk_boundary_policy(report, split="val", image_id=7)
        )

    def test_topk_boundary_audit_rejects_nonfinite_scores(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            audit_topk_boundary(torch.tensor([[0.9, float("nan")]]), topk=1)

    def test_config_provenance_binds_resolved_config_and_include_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.yml"
            leaf = root / "leaf.yml"
            base.write_text("model:\n  depth: 18\n", encoding="utf-8")
            leaf.write_text(
                '__include__: ["./base.yml"]\nmodel:\n  classes: 5\n',
                encoding="utf-8",
            )
            resolved = {"model": {"depth": 18, "classes": 5}}
            first = config_provenance(leaf, resolved)
            second = config_provenance(leaf, resolved)
            self.assertEqual(first, second)
            self.assertEqual(len(first["include_closure"]["files"]), 2)
            self.assertEqual(
                first["include_closure"]["root"], str(leaf.resolve())
            )

            base.write_text("model:\n  depth: 34\n", encoding="utf-8")
            changed = config_provenance(leaf, resolved)
            self.assertNotEqual(
                first["include_closure_sha256"],
                changed["include_closure_sha256"],
            )
            self.assertEqual(
                first["resolved_config_sha256"],
                changed["resolved_config_sha256"],
            )
            changed_resolved = config_provenance(
                leaf, {"model": {"depth": 34, "classes": 5}}
            )
            self.assertNotEqual(
                first["resolved_config_sha256"],
                changed_resolved["resolved_config_sha256"],
            )

    def test_config_provenance_rejects_include_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.yml"
            second = root / "second.yml"
            first.write_text('__include__: ["./second.yml"]\n', encoding="utf-8")
            second.write_text('__include__: ["./first.yml"]\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "include cycle"):
                config_provenance(first, {})
