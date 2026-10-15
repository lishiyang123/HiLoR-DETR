#!/usr/bin/env python3

import ast
import unittest
from pathlib import Path

import manager


class B0QCCROnlyTests(unittest.TestCase):
    def test_python_sources_parse(self):
        root = Path(__file__).resolve().parent
        for name in ("common.py", "manager.py", "self_test.py"):
            ast.parse((root / name).read_text(encoding="utf-8"), filename=name)

    def test_scope_is_exactly_three_detector_seeds(self):
        self.assertEqual(manager.SEEDS, (0, 1, 2))

    def test_table_has_exactly_three_rows(self):
        self.assertEqual(manager.ROWS, ("B0", "B0_QCCR", "B0_QCCR_UMR"))

    def test_new_row_is_qccr_only(self):
        text = (Path(__file__).resolve().parent / "manager.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"  qccr_enabled: True"', text)
        self.assertIn('"  umr_enabled: False"', text)

    def test_no_training_or_fitting_commands(self):
        text = (Path(__file__).resolve().parent / "manager.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("capture_train_val.py", "fit_qccr.py",
                          "train_umr_select_val.py", "--epochs"):
            self.assertNotIn(forbidden, text)

    def test_test_role_is_reporting_only(self):
        text = (Path(__file__).resolve().parent /
                "protocol/B0_QCCR_ONLY_PROTOCOL.json").read_text(
                    encoding="utf-8")
        self.assertIn('"test_used_for_parameter_selection": false', text)
        self.assertIn('"postprocessor_fitting": false', text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
