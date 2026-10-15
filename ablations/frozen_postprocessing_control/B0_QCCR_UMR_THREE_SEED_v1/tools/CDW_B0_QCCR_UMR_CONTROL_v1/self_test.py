#!/usr/bin/env python3
"""Static and semantic checks for the bounded control package."""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads(
    (ROOT / "protocol" / "B0_POSTPROCESS_CONTROL.json").read_text(encoding="utf-8")
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class B0PostprocessControlTests(unittest.TestCase):
    def test_scope_is_exactly_three_frozen_b0_seeds(self):
        self.assertEqual(PROTOCOL["detector"], "B0")
        self.assertEqual(PROTOCOL["detector_seeds"], [0, 1, 2])
        self.assertFalse(PROTOCOL["detector_training"])
        self.assertEqual(PROTOCOL["comparison_rows"], ["B0", "B0_QCCR_UMR"])

    def test_fixed_qccr_has_no_regularization_search(self):
        self.assertEqual(PROTOCOL["qccr_lambda_reg"], 0.001)
        self.assertEqual(PROTOCOL["qccr_fold_count"], 5)
        self.assertEqual(PROTOCOL["qccr_fold_seed"], 0)
        text = (ROOT / "fixed_qccr_fit.py").read_text(encoding="utf-8")
        self.assertIn("LAMBDA_REG_VALUES = (1.0e-3,)", text)

    def test_fixed_umr_has_no_tuple_search(self):
        actual = (
            PROTOCOL["umr_positive_weight"],
            PROTOCOL["umr_head_seed"],
            PROTOCOL["umr_tau"],
            PROTOCOL["umr_beta"],
            PROTOCOL["umr_topk"],
        )
        self.assertEqual(actual, (4, 0, 0.0, 2.0, 300))
        text = (ROOT / "fixed_umr_fit.py").read_text(encoding="utf-8")
        for marker in (
            "SEEDS = (0,)", "POS_WEIGHTS = (4,)",
            "TAU_GRID = (0.0,)", "BETA_GRID = (2.0,)",
        ):
            self.assertIn(marker, text)

    def test_fit_and_test_are_separate_actions(self):
        text = (ROOT / "manager.py").read_text(encoding="utf-8")
        self.assertIn('"verify-frozen", "prepare-test", "start-test"', text)
        run_fit = text[text.index("def run_fit("):text.index("def verify_frozen(")]
        self.assertNotIn("test2017", run_fit)
        self.assertNotIn("instances_test2017", run_fit)
        self.assertIn("verify_fit_lock(lock)", run_fit)
        fit_lock = text[text.index("def make_fit_lock("):text.index("def materialize_fit_configs(")]
        self.assertIn('"test_accessed": False', fit_lock)

    def test_b0_and_postprocessing_switches_are_exact(self):
        text = (ROOT / "manager.py").read_text(encoding="utf-8")
        base = text[text.index("def base_config_text("):text.index("def child_config_text(")]
        final = text[text.index("def final_test_config_text("):text.index("def make_fit_lock(")]
        self.assertGreaterEqual(base.count('"  ufd_enabled: False"'), 2)
        self.assertIn('"  lcct_enabled: False"', base)
        self.assertIn('"  qccr_enabled: False"', base)
        self.assertIn('"  umr_enabled: False"', base)
        self.assertIn('"  qccr_enabled: True"', final)
        self.assertIn('"  umr_enabled: True"', final)

    def test_no_model_dataset_or_result_binary_is_shipped(self):
        forbidden = {".pt", ".pth", ".onnx", ".jpg", ".jpeg", ".png"}
        offenders = [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden]
        self.assertEqual(offenders, [])

    def test_python_sources_parse(self):
        for path in sorted(ROOT.rglob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_payload_manifest(self):
        manifest_path = ROOT / "PAYLOAD_MANIFEST.sha256"
        records = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            records[relative] = digest
        actual = {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in sorted(ROOT.rglob("*"))
            if path.is_file()
            and path.name != "PAYLOAD_MANIFEST.sha256"
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        }
        self.assertEqual(records, actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
