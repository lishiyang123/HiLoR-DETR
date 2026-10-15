#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
MANAGER = (PACKAGE / "manager.py").read_text(encoding="utf-8")
PROTOCOL = json.loads((PACKAGE / "protocol" / "EVALUATION_PROTOCOL.json").read_text(encoding="utf-8"))


class FrozenTestTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for name in ("common.py", "manager.py", "self_test.py"):
            ast.parse((PACKAGE / name).read_text(encoding="utf-8"), filename=name)

    def test_scope_is_exactly_three_new_rows(self):
        self.assertEqual(PROTOCOL["formal_new_rows"], [
            "ufd_without_difficulty", "ufd_only", "lcct_only",
        ])
        self.assertEqual(set(PROTOCOL["reuse_rows"]), {"b0", "gccs"})

    def test_test_is_reporting_only(self):
        self.assertFalse(PROTOCOL["test_model_or_parameter_selection"])
        self.assertFalse(PROTOCOL["fitting_on_test"])
        self.assertIn("reporting", PROTOCOL["test_role"])

    def test_batch_precision_and_postprocess_are_exact(self):
        inference = PROTOCOL["inference"]
        self.assertEqual(inference["actual_total_batch_size"], 8)
        self.assertEqual(inference["precision"], "FP32")
        self.assertFalse(inference["qccr_enabled"])
        self.assertFalse(inference["umr_enabled"])
        self.assertTrue(inference["native_detector_output"])

    def test_formal_command_is_test_only(self):
        tree = ast.parse(MANAGER)
        commands = []
        for node in ast.walk(tree):
            if isinstance(node, ast.List):
                values = []
                for item in node.elts:
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        values.append(item.value)
                if "--test-only" in values:
                    commands.append(values)
        self.assertEqual(len(commands), 1)
        self.assertIn("--test-only", commands[0])

    def test_no_test_tuning_actions_exist(self):
        lowered = MANAGER.lower()
        for forbidden in ("fit_qccr", "train_umr", "grid_search", "optuna"):
            self.assertNotIn(forbidden, lowered)

    def test_resume_is_bounded_to_state_rows(self):
        self.assertIn('p["formal"] / name', MANAGER)
        self.assertIn('output.resolve().parent == root.resolve()', MANAGER)

    def test_payload_manifest(self):
        manifest = PACKAGE / "PAYLOAD_MANIFEST.sha256"
        self.assertTrue(manifest.is_file())
        rows = [line.split(None, 1) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = {
            "common.py", "manager.py", "protocol/EVALUATION_PROTOCOL.json",
            "README.md", "run_server.sh", "self_test.py",
        }
        self.assertEqual({row[1] for row in rows}, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
