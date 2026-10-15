from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import manager


ROOT = Path(__file__).resolve().parent


class TrainingPackageTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in (ROOT / "common.py", ROOT / "manager.py", ROOT / "self_test.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_scope_is_exactly_three_new_seed0_runs(self):
        lock = json.loads((ROOT / "protocol" / "TRAINING_PROTOCOL.json").read_text(encoding="utf-8"))
        self.assertEqual(tuple(lock["formal_variants"]), manager.FORMAL_VARIANTS)
        self.assertEqual(lock["seed"], 0)
        self.assertEqual(lock["epochs"], 102)
        self.assertEqual(lock["physical_batch_size"], 8)
        self.assertEqual(lock["precision"], "FP32")

    def test_no_difficulty_is_a_single_controlled_change(self):
        no_diff = (ROOT / "configs" / "ufd_without_difficulty.yml.in").read_text(encoding="utf-8")
        ufd = (ROOT / "configs" / "ufd_only.yml.in").read_text(encoding="utf-8")
        changed = []
        for left, right in zip(no_diff.splitlines(), ufd.splitlines()):
            if left != right:
                changed.append((left, right))
        self.assertEqual(changed, [("  ufd_alpha_h: 0.0", "  ufd_alpha_h: 0.5")])
        self.assertIn("ufd_alpha_u_max: 0.5", no_diff)
        self.assertIn("ufd_k: 9", no_diff)
        self.assertIn("ufd_lambda_max: 0.75", no_diff)

    def test_variant_switch_matrix(self):
        for name, expected in manager.EXPECTED.items():
            text = (ROOT / "configs" / f"{name}.yml.in").read_text(encoding="utf-8")
            transformer = text.split("DFINETransformer:", 1)[1].split("DEIMCriterion:", 1)[0]
            criterion = text.split("DEIMCriterion:", 1)[1]
            self.assertIn(f"ufd_enabled: {expected['transformer_ufd']}", transformer)
            self.assertIn(f"ufd_enabled: {expected['criterion_ufd']}", criterion)
            self.assertIn(f"lcct_enabled: {expected['lcct']}", criterion)

    def test_training_configs_never_name_test_inputs(self):
        for path in sorted((ROOT / "configs").glob("*.yml.in")):
            lowered = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("test2017", lowered)
            self.assertNotIn("instances_test", lowered)
            self.assertNotIn("qccr_enabled: true", lowered)
            self.assertNotIn("umr_enabled: true", lowered)
        manager_text = (ROOT / "manager.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("instances_test2017", manager_text)
        self.assertNotIn("/test2017", manager_text)

    def test_cleanup_is_bounded_to_state_children(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state"
            target = state / "smoke"
            target.mkdir(parents=True)
            (target / "temporary.txt").write_text("x", encoding="utf-8")
            manager.exact_remove_tree(target, state)
            self.assertFalse(target.exists())
            outside = root / "outside"
            outside.mkdir()
            with self.assertRaises(RuntimeError):
                manager.exact_remove_tree(outside, state)
            self.assertTrue(outside.exists())

    def test_interrupted_queue_has_resume_path(self):
        source = (ROOT / "manager.py").read_text(encoding="utf-8")
        self.assertIn("prepare_resume(output)", source)
        self.assertIn('"-r", resume', source)
        self.assertIn("rerun --start to resume", source)

    def test_reuse_rows_are_content_bound(self):
        lock = json.loads((ROOT / "protocol" / "TRAINING_PROTOCOL.json").read_text(encoding="utf-8"))
        self.assertEqual(set(lock["reuse_rows"]), {"b0", "gccs"})
        for row in lock["reuse_rows"].values():
            self.assertEqual(len(row["checkpoint_sha256"]), 64)
            self.assertEqual(len(row["log_sha256"]), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)

