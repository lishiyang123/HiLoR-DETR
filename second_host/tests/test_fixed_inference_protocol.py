import copy
import json
import unittest
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from tools.fit_common import sha256_file
from tools.fixed_protocol_common import (
    LOCK_RELATIVE,
    POLICY_NAME,
    QCCR_LAMBDA_REG,
    UMR_BETA,
    UMR_HEAD_SEED,
    UMR_POSITIVE_WEIGHT,
    UMR_TAU,
    UMR_TOPK,
    load_fixed_protocol,
    validate_fixed_protocol_payload,
)
from tools.freeze_inference_binding_fixed import (
    mark_fixed_binding,
    validate_fixed_umr_decision,
)
from tools.freeze_qccr_fixed import build_fixed_artifact
from tools.train_umr_fixed import build_fixed_decision, validate_fixed_qccr


IMPLEMENTATION_SHA = sha256_file(ROOT / "provenance/IMPLEMENTATION_LOCK.json")
DATA_LOCK_SHA = sha256_file(ROOT / "provenance/DATA_SPLIT_LOCK.json")
QCCR_RUNTIME_SHA = sha256_file(
    ROOT
    / "source/RT-DETR_HiLoR/rtdetr_pytorch/src/zoo/rtdetr/competition_mapping.py"
)
UMR_RUNTIME_SHA = sha256_file(
    ROOT
    / "source/RT-DETR_HiLoR/rtdetr_pytorch/src/zoo/rtdetr/candidate_value.py"
)
FIXED_LOCK_PATH = ROOT / LOCK_RELATIVE
FIXED_LOCK_SHA = sha256_file(FIXED_LOCK_PATH)


def _fit_payload(checkpoint_sha="1" * 64, capture_sha="2" * 64):
    def candidate(value, beta0):
        return {
            "lambda_reg": value,
            "full_val_fit": {
                "optimizer_success": True,
                "parameters": {
                    "beta0": beta0,
                    "beta_positive": 1.1,
                    "beta_negative": 0.9,
                    "duplicate_penalty": 0.2,
                },
            },
        }

    return {
        "schema": "rtdetr-hilor-qccr-fivefold-fit-v1",
        "test_accessed": False,
        "implementation_lock_sha256": IMPLEMENTATION_SHA,
        "detector_seed": 0,
        "source_checkpoint_sha256": checkpoint_sha,
        "source_capture_sha256": capture_sha,
        "source_data_split_lock_sha256": DATA_LOCK_SHA,
        "runtime_source_sha256": {"hilor_qccr.py": QCCR_RUNTIME_SHA},
        "fit_candidate_count": 123,
        "candidates": {
            "1e-04": candidate(1.0e-4, -1.0),
            "1e-03": candidate(1.0e-3, 0.1),
            "1e-02": candidate(1.0e-2, 2.0),
        },
    }


class FixedProtocolLockTests(unittest.TestCase):
    def test_repository_lock_and_new_tools_form_one_valid_sha_chain(self):
        payload, lock_sha, implementation_sha, data_sha = load_fixed_protocol(
            FIXED_LOCK_PATH,
            FIXED_LOCK_SHA,
            adapter_root_path=ROOT,
            tool_relative="tools/freeze_qccr_fixed.py",
            required_locked_tools=("tools/fit_qccr.py",),
        )
        self.assertEqual(payload["selection_policy"]["name"], POLICY_NAME)
        self.assertEqual(lock_sha, FIXED_LOCK_SHA)
        self.assertEqual(implementation_sha, IMPLEMENTATION_SHA)
        self.assertEqual(data_sha, DATA_LOCK_SHA)
        load_fixed_protocol(
            FIXED_LOCK_PATH,
            FIXED_LOCK_SHA,
            adapter_root_path=ROOT,
            tool_relative="tools/freeze_inference_binding_fixed.py",
            required_locked_tools=("tools/freeze_inference_binding.py",),
        )

    def test_policy_drift_is_rejected(self):
        payload = json.loads(FIXED_LOCK_PATH.read_text(encoding="utf-8"))
        validate_fixed_protocol_payload(payload)
        self.assertNotIn("diagnostic_only", payload["selection_policy"])
        self.assertFalse(
            payload["selection_policy"]["adaptive_candidate_evidence_used"]
        )
        self.assertFalse(payload["selection_policy"]["val_metrics_computed"])
        payload = copy.deepcopy(payload)
        payload["umr"]["beta"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "umr.beta"):
            validate_fixed_protocol_payload(payload)

    def test_documentation_describes_fixed_parameters_and_fitting_splits(self):
        text = (ROOT / "docs/INFERENCE_FIXED_PROTOCOL.md").read_text(
            encoding="utf-8"
        )
        for phrase in ("lambda=0.001", "positive weight=4", "head seed=0",
                       "tau=0, beta=2, Top-K=300", "938", "104"):
            self.assertIn(phrase, text)
        # Machine-readable identities are validated by the preceding lock-chain
        # test; prose should explain the protocol rather than duplicate hashes.


class FixedArtifactTests(unittest.TestCase):
    def test_qccr_freeze_uses_only_predeclared_lambda(self):
        checkpoint_sha = "1" * 64
        capture_sha = "2" * 64
        artifact = build_fixed_artifact(
            _fit_payload(checkpoint_sha, capture_sha),
            detector_seed=0,
            checkpoint=Path("/bound/best_val.pth"),
            checkpoint_sha=checkpoint_sha,
            fits_sha="3" * 64,
            implementation_sha=IMPLEMENTATION_SHA,
            data_lock_sha=DATA_LOCK_SHA,
            qccr_runtime_sha=QCCR_RUNTIME_SHA,
            fixed_lock={"path": LOCK_RELATIVE.as_posix(), "sha256": FIXED_LOCK_SHA},
        )
        self.assertEqual(artifact["selected_lambda_reg"], QCCR_LAMBDA_REG)
        self.assertEqual(artifact["parameters"]["beta0"], 0.1)
        self.assertEqual(artifact["selection_decision"], "OOF_VAL_GO")
        self.assertEqual(artifact["selection_policy"]["decision"], "FIXED_BEFORE_TEST")
        self.assertFalse(artifact["selection_policy"]["oof_gate_read"])
        self.assertFalse(artifact["adaptive_candidate_evidence_used"])
        self.assertFalse(artifact["val_metrics_computed"])
        self.assertTrue(artifact["compatibility_fields_are_not_observed_gates"])
        self.assertNotIn("diagnostic_only", artifact)
        self.assertNotIn("diagnostic_only", artifact["selection_policy"])
        self.assertFalse(artifact["test_accessed"])
        self.assertNotIn("oof_selection_sha256", artifact)

    def test_qccr_freeze_rejects_absent_fixed_lambda(self):
        fits = _fit_payload()
        del fits["candidates"]["1e-03"]
        with self.assertRaisesRegex(RuntimeError, "lambda=0.001"):
            build_fixed_artifact(
                fits,
                detector_seed=0,
                checkpoint=Path("/bound/best_val.pth"),
                checkpoint_sha="1" * 64,
                fits_sha="3" * 64,
                implementation_sha=IMPLEMENTATION_SHA,
                data_lock_sha=DATA_LOCK_SHA,
                qccr_runtime_sha=QCCR_RUNTIME_SHA,
                fixed_lock={
                    "path": LOCK_RELATIVE.as_posix(),
                    "sha256": FIXED_LOCK_SHA,
                },
            )

    def test_umr_record_contains_one_fixed_tuple_and_no_grid(self):
        decision = build_fixed_decision(
            detector_seed=0,
            checkpoint_sha="1" * 64,
            train_capture_sha="4" * 64,
            val_capture_sha="2" * 64,
            data_lock_sha=DATA_LOCK_SHA,
            qccr_sha="5" * 64,
            qccr_source_capture_sha="2" * 64,
            runtime_hashes={
                "hilor_qccr.py": QCCR_RUNTIME_SHA,
                "hilor_umr.py": UMR_RUNTIME_SHA,
            },
            model_path=Path("/bound/umr_seed0_pw4_best.pt"),
            model_sha="6" * 64,
            implementation_sha=IMPLEMENTATION_SHA,
            fixed_lock={"path": LOCK_RELATIVE.as_posix(), "sha256": FIXED_LOCK_SHA},
        )
        selected = decision["selected"]
        self.assertEqual(selected["positive_weight"], UMR_POSITIVE_WEIGHT)
        self.assertEqual(selected["head_seed"], UMR_HEAD_SEED)
        self.assertEqual(selected["tau"], UMR_TAU)
        self.assertEqual(selected["beta"], UMR_BETA)
        self.assertEqual(selected["topk"], UMR_TOPK)
        self.assertEqual(decision["decision"], "VAL_GO")
        self.assertEqual(decision["selection_policy"]["decision"], "FIXED_BEFORE_TEST")
        self.assertFalse(decision["selection_policy"]["val_grid_read"])
        self.assertNotIn("grids", decision)
        self.assertNotIn("baseline_qccr_val", decision)
        self.assertNotIn("top5_by_val_score", decision)
        self.assertFalse(decision["adaptive_candidate_evidence_used"])
        self.assertFalse(decision["val_metrics_computed"])
        self.assertTrue(decision["compatibility_fields_are_not_observed_gates"])
        self.assertNotIn("diagnostic_only", decision)
        self.assertNotIn("diagnostic_only", decision["selection_policy"])
        validate_fixed_umr_decision(
            decision,
            detector_seed=0,
            checkpoint_sha="1" * 64,
            train_capture_sha="4" * 64,
            val_capture_sha="2" * 64,
            data_lock_sha=DATA_LOCK_SHA,
            qccr_sha="5" * 64,
            runtime_hashes={
                "hilor_qccr.py": QCCR_RUNTIME_SHA,
                "hilor_umr.py": UMR_RUNTIME_SHA,
            },
            umr_sha="6" * 64,
            implementation_sha=IMPLEMENTATION_SHA,
            fixed_lock_sha=FIXED_LOCK_SHA,
        )

    def test_fixed_binding_guard_rejects_legacy_or_drifted_umr_decision(self):
        decision = build_fixed_decision(
            detector_seed=0,
            checkpoint_sha="1" * 64,
            train_capture_sha="4" * 64,
            val_capture_sha="2" * 64,
            data_lock_sha=DATA_LOCK_SHA,
            qccr_sha="5" * 64,
            qccr_source_capture_sha="2" * 64,
            runtime_hashes={
                "hilor_qccr.py": QCCR_RUNTIME_SHA,
                "hilor_umr.py": UMR_RUNTIME_SHA,
            },
            model_path=Path("/bound/umr_seed0_pw4_best.pt"),
            model_sha="6" * 64,
            implementation_sha=IMPLEMENTATION_SHA,
            fixed_lock={"path": LOCK_RELATIVE.as_posix(), "sha256": FIXED_LOCK_SHA},
        )
        arguments = {
            "detector_seed": 0,
            "checkpoint_sha": "1" * 64,
            "train_capture_sha": "4" * 64,
            "val_capture_sha": "2" * 64,
            "data_lock_sha": DATA_LOCK_SHA,
            "qccr_sha": "5" * 64,
            "runtime_hashes": {
                "hilor_qccr.py": QCCR_RUNTIME_SHA,
                "hilor_umr.py": UMR_RUNTIME_SHA,
            },
            "umr_sha": "6" * 64,
            "implementation_sha": IMPLEMENTATION_SHA,
            "fixed_lock_sha": FIXED_LOCK_SHA,
        }
        legacy = copy.deepcopy(decision)
        del legacy["selection_policy"]
        with self.assertRaisesRegex(RuntimeError, "fixed pre-Test"):
            validate_fixed_umr_decision(legacy, **arguments)
        drifted = copy.deepcopy(decision)
        drifted["selected"]["beta"] = 1.0
        with self.assertRaisesRegex(RuntimeError, "tuple mismatch: beta"):
            validate_fixed_umr_decision(drifted, **arguments)

    def test_fixed_binding_visibly_carries_policy_and_lock(self):
        lock = {"path": LOCK_RELATIVE.as_posix(), "sha256": FIXED_LOCK_SHA}
        binding = mark_fixed_binding({"schema": "legacy-compatible"}, lock)
        self.assertEqual(binding["fixed_protocol_lock"], lock)
        self.assertFalse(binding["adaptive_candidate_evidence_used"])
        self.assertFalse(binding["val_metrics_computed"])
        self.assertTrue(binding["compatibility_fields_are_not_observed_gates"])
        self.assertNotIn("diagnostic_only", binding)
        self.assertNotIn("diagnostic_only", binding["selection_policy"])
        self.assertEqual(
            binding["selection_policy"]["fixed_tuple"],
            {
                "qccr_lambda_reg": QCCR_LAMBDA_REG,
                "umr_positive_weight": UMR_POSITIVE_WEIGHT,
                "umr_head_seed": UMR_HEAD_SEED,
                "umr_tau": UMR_TAU,
                "umr_beta": UMR_BETA,
                "umr_topk": UMR_TOPK,
            },
        )

    def test_fixed_qccr_is_accepted_by_fixed_umr_guard(self):
        qccr = build_fixed_artifact(
            _fit_payload(),
            detector_seed=0,
            checkpoint=Path("/bound/best_val.pth"),
            checkpoint_sha="1" * 64,
            fits_sha="3" * 64,
            implementation_sha=IMPLEMENTATION_SHA,
            data_lock_sha=DATA_LOCK_SHA,
            qccr_runtime_sha=QCCR_RUNTIME_SHA,
            fixed_lock={"path": LOCK_RELATIVE.as_posix(), "sha256": FIXED_LOCK_SHA},
        )
        parameters = validate_fixed_qccr(
            qccr,
            detector_seed=0,
            checkpoint_sha="1" * 64,
            val_capture_sha="2" * 64,
            data_lock_sha=DATA_LOCK_SHA,
            implementation_sha=IMPLEMENTATION_SHA,
            fixed_lock_sha=FIXED_LOCK_SHA,
            qccr_runtime_sha=QCCR_RUNTIME_SHA,
        )
        self.assertEqual(parameters["duplicate_penalty"], 0.2)


if __name__ == "__main__":
    unittest.main()
