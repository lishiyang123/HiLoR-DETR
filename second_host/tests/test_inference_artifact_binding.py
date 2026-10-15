import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from _bootstrap import ROOT  # noqa: F401
from src.zoo.rtdetr.hilor_artifacts import (
    FEATURE_SCHEMA,
    SCHEMA,
    load_bound_inference_artifact,
    sha256_file,
)
from src.zoo.rtdetr.hilor_umr import UMRHead
from tools.fit_common import canonical_data_lock_sha, runtime_source_hashes


SOURCE_ROOT = ROOT / "source" / "RT-DETR_HiLoR" / "rtdetr_pytorch"
DATA_LOCK_SHA = canonical_data_lock_sha(ROOT)
RUNTIME = runtime_source_hashes(SOURCE_ROOT, require_umr=True)
IMPLEMENTATION_SHA = json.loads(
    (ROOT / "provenance/PROTOCOL_LOCK.json").read_text(encoding="utf-8")
)["bindings"]["implementation_lock"]["sha256"]


class InferenceArtifactBindingTest(unittest.TestCase):
    def make_binding(self, root: Path):
        detector_sha = "1" * 64
        val_capture_sha = "3" * 64
        qccr_path = root / "qccr.json"
        qccr_path.write_text(
            json.dumps(
                {
                    "schema": "rtdetr-hilor-qccr-frozen-artifact-v1",
                    "method": "QCCR-v1",
                    "dataset": "CDW1630",
                    "implementation_lock_sha256": IMPLEMENTATION_SHA,
                    "detector_seed": 0,
                    "source_checkpoint_sha256": detector_sha,
                    "source_capture_sha256": val_capture_sha,
                    "source_data_split_lock_sha256": DATA_LOCK_SHA,
                    "runtime_source_sha256": {
                        "hilor_qccr.py": RUNTIME["hilor_qccr.py"]
                    },
                    "selection_decision": "OOF_VAL_GO",
                    "test_used_for_selection": False,
                    "parameters": {
                        "beta0": 0.1,
                        "beta_positive": 1.1,
                        "beta_negative": 0.9,
                        "duplicate_penalty": 0.2,
                    },
                }
            ),
            encoding="utf-8",
        )
        umr_path = root / "umr.pt"
        torch.save(UMRHead(num_classes=5).state_dict(), umr_path)
        selection_path = root / "selection.json"
        selection_path.write_text(
            json.dumps(
                {
                    "schema": "rtdetr-hilor-umr-val-selection-v1",
                    "decision": "VAL_GO",
                    "test_accessed": False,
                    "official_test_accessed": False,
                    "protocol": "CDW1630 Train candidates / Val-only selection",
                    "implementation_lock_sha256": IMPLEMENTATION_SHA,
                    "detector_seed": 0,
                    "source_checkpoint_sha256": detector_sha,
                    "train_capture_sha256": "2" * 64,
                    "val_capture_sha256": val_capture_sha,
                    "source_data_split_lock_sha256": DATA_LOCK_SHA,
                    "qccr_artifact_sha256": sha256_file(qccr_path),
                    "qccr_source_capture_sha256": val_capture_sha,
                    "runtime_source_sha256": dict(RUNTIME),
                    "feature_schema": FEATURE_SCHEMA,
                    "selected_model": {"sha256": sha256_file(umr_path)},
                    "selected": {"tau": 0.0, "beta": 2.0},
                }
            ),
            encoding="utf-8",
        )
        qccr_entry = {
            "path": qccr_path.name,
            "sha256": sha256_file(qccr_path),
            "source_capture_sha256": val_capture_sha,
            "source_data_split_lock_sha256": DATA_LOCK_SHA,
            "runtime_source_sha256": {"hilor_qccr.py": RUNTIME["hilor_qccr.py"]},
        }
        manifest = {
            "schema": SCHEMA,
            "format_version": 2,
            "mode": "qccr-umr",
            "detector_seed": 0,
            "host": "RT-DETR-R18",
            "dataset": "CDW1630",
            "implementation_lock_sha256": IMPLEMENTATION_SHA,
            "test_accessed_for_fitting_or_selection": False,
            "fit_and_selection_splits": ["train", "val"],
            "detector_checkpoint_sha256": detector_sha,
            "source_data_split_lock_sha256": DATA_LOCK_SHA,
            "source_val_capture_sha256": val_capture_sha,
            "runtime_source_sha256": dict(RUNTIME),
            "captures": {
                split: {
                    "sha256": value * 64,
                    "detector_checkpoint_sha256": detector_sha,
                    "data_split_lock_sha256": DATA_LOCK_SHA,
                }
                for split, value in (("train", "2"), ("val", "3"))
            },
            "qccr_artifact": qccr_entry,
            "umr": {
                "path": umr_path.name,
                "sha256": sha256_file(umr_path),
                "selection_path": selection_path.name,
                "selection_sha256": sha256_file(selection_path),
                "feature_schema": FEATURE_SCHEMA,
                "num_classes": 5,
                "num_queries": 300,
                "topk": 300,
                "tau": 0.0,
                "beta": 2.0,
                "source_val_capture_sha256": val_capture_sha,
                "runtime_source_sha256": dict(RUNTIME),
            },
        }
        manifest_path = root / "binding.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, manifest

    def test_complete_binding_loads_and_checks_manifest_and_weight_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, manifest = self.make_binding(Path(directory))
            loaded = load_bound_inference_artifact(
                {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
                require_umr=True,
            )
            self.assertEqual(
                loaded["detector_checkpoint_sha256"],
                manifest["detector_checkpoint_sha256"],
            )
            self.assertEqual(loaded["qccr_parameters"]["duplicate_penalty"], 0.2)

            with self.assertRaisesRegex(RuntimeError, "inference binding SHA-256"):
                load_bound_inference_artifact(
                    {"path": str(manifest_path), "sha256": "4" * 64},
                    require_umr=True,
                )

            manifest["umr"]["sha256"] = "4" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "UMR checkpoint SHA-256"):
                load_bound_inference_artifact(manifest_path, require_umr=True)

    def test_runtime_loading_does_not_require_development_locks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = self.make_binding(root)
            manifest.pop("runtime_source_sha256")
            manifest.pop("implementation_lock_sha256")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            # A separately located inference bundle needs the actual fitted
            # files, not the training machine's source/protocol lock tree.
            with patch("src.zoo.rtdetr.hilor_artifacts._adapter_root",
                       return_value=root / "no_development_files"):
                loaded = load_bound_inference_artifact(manifest_path, require_umr=True)
            self.assertEqual(loaded["umr"]["checkpoint"], str(root / "umr.pt"))

    def test_mapping_from_another_detector_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = self.make_binding(root)
            mapping_path = root / manifest["qccr_artifact"]["path"]
            mapping = json.loads(mapping_path.read_text())
            mapping["source_checkpoint_sha256"] = "0" * 64
            mapping_path.write_text(json.dumps(mapping))
            manifest["qccr_artifact"]["sha256"] = sha256_file(mapping_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "another detector checkpoint"):
                load_bound_inference_artifact(manifest_path, require_umr=True)

    def test_umr_test_isolation_and_protocol_remain_runtime_guards(self):
        cases = (
            ("official_test_accessed", True, "Test exclusion"),
            ("protocol", "different protocol", "another Train/Val protocol"),
        )
        for key, value, expected in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest_path, manifest = self.make_binding(root)
                selection_path = root / manifest["umr"]["selection_path"]
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                selection[key] = value
                selection_path.write_text(json.dumps(selection), encoding="utf-8")
                manifest["umr"]["selection_sha256"] = sha256_file(selection_path)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, expected):
                    load_bound_inference_artifact(
                        manifest_path, require_umr=True
                    )

    def test_wrong_dimensions_and_nonfinite_settings_are_rejected(self):
        for key, value, message in (
            ('num_classes', 3, 'dimensions'),
            ('num_queries', 200, 'dimensions'),
            ('topk', 200, 'Top-k'),
            ('tau', float('nan'), 'finite'),
            ('beta', float('inf'), 'finite'),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                manifest_path, manifest = self.make_binding(Path(directory))
                manifest['umr'][key] = value
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, message):
                    load_bound_inference_artifact(manifest_path, require_umr=True)


if __name__ == "__main__":
    unittest.main()
