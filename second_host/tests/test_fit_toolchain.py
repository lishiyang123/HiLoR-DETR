import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from _bootstrap import ROOT  # noqa: F401
from src.zoo.rtdetr.hilor_artifacts import load_bound_inference_artifact
from src.zoo.rtdetr.hilor_qccr import apply_qccr, compute_competition_risk
from src.zoo.rtdetr.hilor_umr import UMR, UMRHead, build_image_features
from tools.build_qccr_targets import build_payload
from tools.evaluate_qccr_oof_val import SELECTION_SCHEMA, decide
from tools.fit_common import (
    REQUIRED_RUNTIME_KEYS,
    formal_fit_guard,
    install_source_root,
    load_capture,
    require_output_path,
    runtime_source_hashes,
    runtime_source_path,
    sha256_file,
)
from tools.fit_qccr import fit_parameters, run_fit
from tools.freeze_inference_binding import main as freeze_binding_main
from tools.freeze_qccr import main as freeze_qccr_main
from tools.render_inference_config import main as render_config_main
from tools.train_umr_select_val import build_train_data, train_one


SOURCE_ROOT = ROOT / "source" / "RT-DETR_HiLoR" / "rtdetr_pytorch"
RUNTIME_HASHES = runtime_source_hashes(SOURCE_ROOT, require_umr=True)
TEST_RUNTIME = {name: f"test-{index}" for index, name in enumerate(REQUIRED_RUNTIME_KEYS)}


def _lines_sha(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(f"{value}\n".encode("utf-8"))
    return digest.hexdigest()


def synthetic_data_lock(*, train_count=3, val_count=10):
    splits = {}
    for split, count, offset in (
        ("train", train_count, 0),
        ("val", val_count, 100),
        ("test", 1, 1000),
    ):
        image_ids = list(range(offset, offset + count))
        file_names = [f"{split}_{image_id}.jpg" for image_id in image_ids]
        splits[split] = {
            "annotation": f"/synthetic/{split}.json",
            "annotation_sha256": hashlib.sha256(
                f"{split}-annotation".encode("utf-8")
            ).hexdigest(),
            "images_dir": f"/synthetic/{split}",
            "image_count": count,
            "instance_count": count,
            "image_id_set_sha256": _lines_sha(image_ids),
            "file_name_set_sha256": _lines_sha(file_names),
            "image_tree_sha256": hashlib.sha256(
                f"{split}-image-tree".encode("utf-8")
            ).hexdigest(),
        }
    splits["test"]["selection_or_fitting_allowed"] = False
    return {
        "schema": "cdw1630-rtdetr-r18-hilor-data-split-lock-v1",
        "category_ids": list(range(5)),
        "selection_eligible_splits": ["train", "val"],
        "splits": splits,
    }


DATA_LOCK = synthetic_data_lock()
DATA_LOCK_BYTES = (
    json.dumps(DATA_LOCK, sort_keys=True, separators=(",", ":")) + "\n"
).encode("utf-8")
DATA_LOCK_SHA = hashlib.sha256(DATA_LOCK_BYTES).hexdigest()


def make_adapter_root(root: Path, *, data_lock=None) -> Path:
    provenance = root / "provenance"
    provenance.mkdir(parents=True)
    data_name = "DATA_SPLIT_LOCK.json"
    payload = DATA_LOCK if data_lock is None else data_lock
    payload_bytes = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    lock_sha = hashlib.sha256(payload_bytes).hexdigest()
    (provenance / data_name).write_bytes(payload_bytes)
    common = root / "configs" / "hilor" / "cdw1630_rtdetr_r18_capture.yml"
    common.parent.mkdir(parents=True)
    common.write_text("model: RTDETR\n", encoding="utf-8")
    capture_configs = {}
    for split in ("train", "val"):
        relative = Path(
            f"configs/hilor/cdw1630_rtdetr_r18_capture_{split}.yml"
        )
        leaf = root / relative
        leaf.write_text(
            '__include__: ["./cdw1630_rtdetr_r18_capture.yml"]\n',
            encoding="utf-8",
        )
        closure = [
            {
                "path": common.relative_to(root).as_posix(),
                "sha256": sha256_file(common),
            },
            {"path": relative.as_posix(), "sha256": sha256_file(leaf)},
        ]
        closure.sort(key=lambda item: item["path"])
        aggregate = hashlib.sha256()
        for record in closure:
            aggregate.update(
                f"{record['sha256']}  {record['path']}\n".encode("utf-8")
            )
        capture_configs[split] = {
            "path": relative.as_posix(),
            "sha256": sha256_file(leaf),
            "include_closure": closure,
            "include_closure_aggregate_sha256": aggregate.hexdigest(),
            "include_closure_file_count": len(closure),
        }
    tool_relatives = (
        "tools/fit_common.py",
        "tools/build_qccr_targets.py",
        "tools/fit_qccr.py",
        "tools/evaluate_qccr_oof_val.py",
        "tools/freeze_qccr.py",
        "tools/train_umr_select_val.py",
        "tools/freeze_inference_binding.py",
    )
    tools = {}
    for relative in tool_relatives:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        tools[relative] = sha256_file(target)
    implementation = {
        "schema": "cdw1630-rtdetr-r18-hilor-implementation-lock-v1",
        "host": "RT-DETR-R18",
        "dataset": "CDW1630",
        "public_inference_name": "QCCR–UMR",
        "test_policy": {"selection_or_fitting_allowed": False},
        "runtime_environment": TEST_RUNTIME,
        "provenance": {"data_split_lock_sha256": lock_sha},
        "capture_configs": capture_configs,
        "fit_runtime_sources": dict(RUNTIME_HASHES),
        "tools": tools,
    }
    implementation_bytes = (
        json.dumps(implementation, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    implementation_path = provenance / "IMPLEMENTATION_LOCK.json"
    implementation_path.write_bytes(implementation_bytes)
    implementation_sha = hashlib.sha256(implementation_bytes).hexdigest()
    protocol = {
        "schema": "cdw1630-rtdetr-r18-hilor-second-host-protocol-lock-v1",
        "bindings": {
            "data_split_lock": {
                "path": f"provenance/{data_name}",
                "sha256": lock_sha,
            },
            "implementation_lock": {
                "path": "provenance/IMPLEMENTATION_LOCK.json",
                "sha256": implementation_sha,
            },
        },
    }
    (provenance / "PROTOCOL_LOCK.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    return root


def make_record(image_id):
    generator = torch.Generator().manual_seed(1000 + image_id)
    logits = torch.randn(300, 5, generator=generator) * 0.15 - 2.0
    logits[0, 0] = 8.0
    boxes = torch.rand(300, 4, generator=generator)
    boxes[:, 2:] = boxes[:, 2:] * 0.2 + 0.03
    boxes[0] = torch.tensor([0.5, 0.5, 0.4, 0.4])
    hidden = torch.randn(300, 256, generator=generator)
    flat_scores = torch.sigmoid(logits).flatten()
    scores, indices = torch.topk(flat_scores, 300)
    query = indices // 5
    labels = indices % 5
    cx, cy, width, height = boxes.unbind(-1)
    xyxy = torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=-1,
    ) * 100.0
    next_score = float(torch.topk(flat_scores, 301).values[-1])
    target_box = torch.tensor([[30.0, 30.0, 70.0, 70.0]])
    return {
        "image_id": image_id,
        "pred_logits": logits,
        "pred_boxes": boxes,
        "hL": hidden,
        "orig_size": torch.tensor([100, 100], dtype=torch.int64),
        "gt_boxes": target_box,
        "gt_labels": torch.tensor([0]),
        "gt_area": torch.tensor([1600.0]),
        "gt_iscrowd": torch.tensor([0]),
        "target_boxes": target_box,
        "target_labels": torch.tensor([0]),
        "target_area": torch.tensor([1600.0]),
        "target_iscrowd": torch.tensor([0]),
        "post_query_idx": query,
        "post_labels": labels,
        "post_scores": scores,
        "post_boxes": xyxy[query],
        "topk_boundary_audit": {
            "image_id": image_id,
            "boundary_tie": False,
            "boundary_score": float(scores[-1]),
            "boundary_margin": float(scores[-1]) - next_score,
            "policy": "record_native_torch_topk_exact_boundary_ties",
        },
    }


def make_capture(
    split, checkpoint, checkpoint_sha, count=10, *, adapter_root_path,
    data_lock=None,
    data_lock_sha=None,
):
    data_lock = DATA_LOCK if data_lock is None else data_lock
    if data_lock_sha is None:
        encoded = (
            json.dumps(data_lock, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        data_lock_sha = hashlib.sha256(encoded).hexdigest()
    offset = 0 if split == "train" else 100
    resolved = "a" * 64
    split_binding = data_lock["splits"][split]
    image_tree_sha = split_binding["image_tree_sha256"]
    root = Path(adapter_root_path).resolve()
    config = root / f"configs/hilor/cdw1630_rtdetr_r18_capture_{split}.yml"
    common = root / "configs/hilor/cdw1630_rtdetr_r18_capture.yml"
    closure = {
        "root": str(config),
        "files": sorted(
            [
                {
                    "path": str(common),
                    "sha256": sha256_file(common),
                    "includes_in_declared_order": [],
                },
                {
                    "path": str(config),
                    "sha256": sha256_file(config),
                    "includes_in_declared_order": [str(common)],
                },
            ],
            key=lambda item: item["path"],
        ),
    }
    implementation_sha = json.loads(
        (root / "provenance/PROTOCOL_LOCK.json").read_text(encoding="utf-8")
    )["bindings"]["implementation_lock"]["sha256"]
    return {
        "format_version": 4,
        "schema": "rtdetr-hilor-frozen-query-class-capture-v1",
        "method": "frozen_native_detector",
        "host": "RT-DETR-R18",
        "split": split,
        "test_accessed": False,
        "selection_eligible": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "training_run": {"implementation_lock_sha256": implementation_sha},
        "config": str(config),
        "config_sha256": sha256_file(config),
        "config_include_closure_sha256": hashlib.sha256(
            json.dumps(
                closure, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest(),
        "config_include_closure": closure,
        "data_split_lock_sha256": data_lock_sha,
        "image_tree_sha256": image_tree_sha,
        "annotation": "synthetic_val.json",
        "annotation_sha256": split_binding["annotation_sha256"],
        "image_id_set_sha256": split_binding["image_id_set_sha256"],
        "file_name_set_sha256": split_binding["file_name_set_sha256"],
        "image_count": split_binding["image_count"],
        "resolved_config_sha256": resolved,
        "postprocessor_num_classes": 5,
        "postprocessor_num_top_queries": 300,
        "records": [make_record(index + offset) for index in range(count)],
        "summary": {
            "image_count": count,
            "query_counts": [300],
            "resolved_config_sha256": resolved,
            "image_tree_sha256": image_tree_sha,
            "topk_boundary_tie_policy": "record_native_torch_topk_exact_boundary_ties",
            "topk_boundary_audit_count": count,
            "topk_boundary_tie_count": 0,
        },
    }


class FitToolchainTest(unittest.TestCase):
    def test_capture_loader_accepts_recorded_native_topk_boundary_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_adapter_root(Path(directory))
            checkpoint = root / "runs/seed2/best_val.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"detector")
            checkpoint_sha = sha256_file(checkpoint)
            capture = make_capture(
                "train", checkpoint, checkpoint_sha, count=3,
                adapter_root_path=root,
            )
            capture["records"][0]["topk_boundary_audit"]["boundary_tie"] = True
            capture["records"][0]["topk_boundary_audit"]["boundary_margin"] = 0.0
            capture["summary"]["topk_boundary_tie_count"] = 1
            capture_path = root / "train.pt"
            torch.save(capture, capture_path)
            loaded = load_capture(
                capture_path,
                expected_sha=sha256_file(capture_path),
                split="train",
                expected_checkpoint_sha=checkpoint_sha,
                adapter_root_path=root,
                detector_seed=2,
            )
            self.assertTrue(
                loaded["records"][0]["topk_boundary_audit"]["boundary_tie"]
            )

    def test_source_loader_imports_umr_only_when_requested(self):
        qccr_only = install_source_root(SOURCE_ROOT, require_umr=False)
        self.assertTrue(callable(qccr_only[0]))
        self.assertTrue(callable(qccr_only[1]))
        self.assertEqual(qccr_only[2:], (None, None))

        with_umr = install_source_root(SOURCE_ROOT, require_umr=True)
        self.assertTrue(all(callable(value) for value in with_umr))

    def test_formal_fit_guard_rejects_runtime_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_adapter_root(Path(directory))
            evidence = formal_fit_guard(
                tool_relative="tools/fit_qccr.py",
                adapter_root_path=root,
                runtime_versions=TEST_RUNTIME,
            )
            self.assertEqual(len(evidence["implementation_lock_sha256"]), 64)
            drifted = dict(TEST_RUNTIME)
            drifted["torch"] = "wrong"
            with self.assertRaisesRegex(RuntimeError, "runtime environment differs"):
                formal_fit_guard(
                    tool_relative="tools/fit_qccr.py",
                    adapter_root_path=root,
                    runtime_versions=drifted,
                )

    def test_formal_fit_guard_rejects_tool_and_runtime_source_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_adapter_root(Path(directory))
            evidence = formal_fit_guard(
                tool_relative="tools/build_qccr_targets.py",
                adapter_root_path=root,
                source_root=SOURCE_ROOT,
                check_runtime=False,
            )
            self.assertEqual(
                evidence["runtime_source_sha256"]["hilor_qccr.py"],
                RUNTIME_HASHES["hilor_qccr.py"],
            )
            tool = root / "tools/build_qccr_targets.py"
            tool.write_text("# drifted\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "formal fit tool.*mismatch"):
                formal_fit_guard(
                    tool_relative="tools/build_qccr_targets.py",
                    adapter_root_path=root,
                    source_root=SOURCE_ROOT,
                    check_runtime=False,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = make_adapter_root(Path(directory))
            source_root = root / "runtime-source"
            runtime_dir = source_root / "src/zoo/rtdetr"
            runtime_dir.mkdir(parents=True)
            for name in ("hilor_qccr.py", "hilor_umr.py"):
                (runtime_dir / name).write_bytes(
                    runtime_source_path(SOURCE_ROOT, name).read_bytes()
                )
            (runtime_dir / "hilor_qccr.py").write_text(
                "# drifted runtime\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "runtime source differs"):
                formal_fit_guard(
                    tool_relative="tools/build_qccr_targets.py",
                    adapter_root_path=root,
                    source_root=source_root,
                    check_runtime=False,
                )

    def test_formal_fit_clis_expose_no_test_argument(self):
        for name in (
            "build_qccr_targets.py",
            "fit_qccr.py",
            "evaluate_qccr_oof_val.py",
            "freeze_qccr.py",
            "train_umr_select_val.py",
        ):
            source = (ROOT / "tools" / name).read_text(encoding="utf-8")
            self.assertNotIn('"--test', source, msg=name)
            self.assertNotIn("'--test", source, msg=name)

    def test_optimizer_failure_is_fatal(self):
        fake = SimpleNamespace(
            x=np.asarray((0.0, 1.0, 1.0, 0.0)), fun=1.0, success=False,
            status=2, message="failed", nit=1,
        )
        with patch("tools.fit_qccr.minimize", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                fit_parameters(
                    np.ones((2, 4)), np.asarray((0.0, 1.0)),
                    np.asarray((0.0, 0.0)), 1.0e-4,
                )

    def test_output_namespace_is_seed_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / "artifacts/inference/seed1/qccr/value.json"
            self.assertEqual(
                require_output_path(
                    allowed, namespace="inference", detector_seed=1,
                    adapter_root_path=root,
                ),
                allowed.resolve(),
            )
            with self.assertRaisesRegex(RuntimeError, "seed1"):
                require_output_path(
                    root / "artifacts/inference/seed0/value.json",
                    namespace="inference", detector_seed=1,
                    adapter_root_path=root,
                )

    def test_synthetic_qccr_fit_freeze_and_both_binding_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_adapter_root(Path(directory))
            checkpoint = root / "runs/seed0/best_val.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"detector")
            checkpoint_sha = sha256_file(checkpoint)
            inference = root / "artifacts/inference/seed0"
            qccr_dir, umr_dir = inference / "qccr", inference / "umr"
            qccr_dir.mkdir(parents=True)
            umr_dir.mkdir()

            val_capture = make_capture(
                "val", checkpoint, checkpoint_sha, adapter_root_path=root
            )
            val_path = root / "val.pt"
            torch.save(val_capture, val_path)
            val_sha = sha256_file(val_path)
            loaded = load_capture(
                val_path, expected_sha=val_sha, split="val",
                expected_checkpoint_sha=checkpoint_sha,
                adapter_root_path=root, detector_seed=0,
            )
            implementation_sha = json.loads(
                (root / "provenance/PROTOCOL_LOCK.json").read_text(
                    encoding="utf-8"
                )
            )["bindings"]["implementation_lock"]["sha256"]
            with self.assertRaisesRegex(RuntimeError, "implementation-lock SHA-256"):
                build_payload(
                    loaded,
                    val_path,
                    val_sha,
                    SOURCE_ROOT,
                    implementation_lock_sha256="",
                    detector_seed=0,
                )
            targets = build_payload(
                loaded, val_path, val_sha, SOURCE_ROOT, detector_seed=0,
                implementation_lock_sha256=implementation_sha,
            )
            fit = run_fit(
                targets, target_path=qccr_dir / "targets.pt",
                target_sha="4" * 64, seed=0, folds=5,
            )
            self.assertEqual(fit["source_capture_sha256"], val_sha)
            self.assertEqual(fit["source_data_split_lock_sha256"], DATA_LOCK_SHA)
            candidate_key = next(iter(fit["candidates"]))
            parameters = fit["candidates"][candidate_key]["full_val_fit"]["parameters"]
            baseline = {
                "AP": 0.5, "AP50": 0.7, "AP75": 0.5, "APm": 0.5,
                "APl": 0.5, "AR100": 0.6, "AP90": 0.2, "AP95": 0.1,
            }
            improved = {**baseline, "AP": 0.502, "AP75": 0.501, "APm": 0.501}
            _, selected, _, _, decision = decide(
                {candidate_key: improved}, baseline, fit
            )
            self.assertEqual((selected, decision), (candidate_key, "OOF_VAL_GO"))

            fit_path = qccr_dir / "fits.json"
            fit_path.write_text(json.dumps(fit), encoding="utf-8")
            fit_sha = sha256_file(fit_path)
            oof = {
                "schema": SELECTION_SCHEMA,
                "protocol": "image-level-five-fold-OOF-Val",
                "test_accessed": False,
                "decision": "OOF_VAL_GO",
                "fits_sha256": fit_sha,
                "checkpoint_sha256": checkpoint_sha,
                "detector_seed": 0,
                "implementation_lock_sha256": implementation_sha,
                "capture_sha256": val_sha,
                "source_capture_sha256": val_sha,
                "source_data_split_lock_sha256": DATA_LOCK_SHA,
                "runtime_source_sha256": {
                    "hilor_qccr.py": RUNTIME_HASHES["hilor_qccr.py"]
                },
                "selected_candidate_key": candidate_key,
                "selected_lambda_reg": fit["candidates"][candidate_key]["lambda_reg"],
                "selected_full_val_parameters": parameters,
                "baseline_metrics": baseline,
                "candidate_decisions": {candidate_key: {"metrics": improved}},
            }
            oof_path = qccr_dir / "oof.json"
            oof_path.write_text(json.dumps(oof), encoding="utf-8")
            frozen = qccr_dir / "frozen.json"
            with patch.object(
                sys,
                "argv",
                [
                    "freeze_qccr.py", "--adapter-root", str(root),
                    "--detector-seed", "0", "--oof-selection", str(oof_path),
                    "--expected-oof-selection-sha", sha256_file(oof_path),
                    "--fits", str(fit_path), "--expected-fits-sha", fit_sha,
                    "--checkpoint", str(checkpoint),
                    "--expected-checkpoint-sha", checkpoint_sha,
                    "--output", str(frozen),
                ],
            ):
                freeze_qccr_main()
            frozen_payload = json.loads(frozen.read_text(encoding="utf-8"))
            self.assertEqual(frozen_payload["source_capture_sha256"], val_sha)
            self.assertGreater(frozen_payload["parameters"]["beta_positive"], 0.0)
            self.assertEqual(
                targets["implementation_lock_sha256"], implementation_sha
            )
            self.assertEqual(fit["implementation_lock_sha256"], implementation_sha)
            self.assertEqual(oof["implementation_lock_sha256"], implementation_sha)
            self.assertEqual(
                frozen_payload["implementation_lock_sha256"], implementation_sha
            )

            stale_fit = dict(fit)
            stale_fit["implementation_lock_sha256"] = "9" * 64
            stale_fit_path = qccr_dir / "stale_fits.json"
            stale_fit_path.write_text(json.dumps(stale_fit), encoding="utf-8")
            stale_fit_sha = sha256_file(stale_fit_path)
            stale_oof = dict(oof)
            stale_oof["implementation_lock_sha256"] = "9" * 64
            stale_oof["fits_sha256"] = stale_fit_sha
            stale_oof_path = qccr_dir / "stale_oof.json"
            stale_oof_path.write_text(json.dumps(stale_oof), encoding="utf-8")
            with patch.object(
                sys,
                "argv",
                [
                    "freeze_qccr.py", "--adapter-root", str(root),
                    "--detector-seed", "0",
                    "--oof-selection", str(stale_oof_path),
                    "--expected-oof-selection-sha", sha256_file(stale_oof_path),
                    "--fits", str(stale_fit_path),
                    "--expected-fits-sha", stale_fit_sha,
                    "--checkpoint", str(checkpoint),
                    "--expected-checkpoint-sha", checkpoint_sha,
                    "--output", str(qccr_dir / "stale_frozen.json"),
                ],
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "another implementation lock"
                ):
                    freeze_qccr_main()

            qccr_binding = inference / "qccr_binding.json"
            with patch.object(
                sys,
                "argv",
                [
                    "freeze_inference_binding.py", "--adapter-root", str(root),
                    "--source-root", str(SOURCE_ROOT), "--mode", "qccr",
                    "--detector-seed", "0", "--detector-checkpoint", str(checkpoint),
                    "--expected-detector-sha", checkpoint_sha,
                    "--val-capture", str(val_path),
                    "--expected-val-capture-sha", val_sha,
                    "--qccr-artifact", str(frozen),
                    "--expected-qccr-sha", sha256_file(frozen),
                    "--output", str(qccr_binding),
                ],
            ):
                freeze_binding_main()
            tampered_qccr = dict(frozen_payload)
            tampered_qccr["implementation_lock_sha256"] = "2" * 64
            tampered_qccr_path = qccr_dir / "tampered_implementation.json"
            tampered_qccr_path.write_text(
                json.dumps(tampered_qccr), encoding="utf-8"
            )
            with patch.object(
                sys,
                "argv",
                [
                    "freeze_inference_binding.py", "--adapter-root", str(root),
                    "--source-root", str(SOURCE_ROOT), "--mode", "qccr",
                    "--detector-seed", "0", "--detector-checkpoint", str(checkpoint),
                    "--expected-detector-sha", checkpoint_sha,
                    "--val-capture", str(val_path),
                    "--expected-val-capture-sha", val_sha,
                    "--qccr-artifact", str(tampered_qccr_path),
                    "--expected-qccr-sha", sha256_file(tampered_qccr_path),
                    "--output", str(inference / "tampered_qccr_binding.json"),
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid QCCR artifact"):
                    freeze_binding_main()
            qccr_bound = load_bound_inference_artifact(
                {"path": str(qccr_binding), "sha256": sha256_file(qccr_binding)},
                require_umr=False,
            )
            self.assertEqual(qccr_bound["detector_seed"], 0)
            qccr_binding_payload = json.loads(
                qccr_binding.read_text(encoding="utf-8")
            )
            self.assertNotIn("protocol_lock_sha256", qccr_binding_payload)
            self.assertNotIn("runtime_environment", qccr_binding_payload)
            self.assertNotIn("fit_evidence", qccr_binding_payload["qccr_artifact"])
            stale_binding_payload = dict(qccr_binding_payload)
            stale_binding_payload["implementation_lock_sha256"] = "8" * 64
            stale_binding = inference / "stale_qccr_binding.json"
            stale_binding.write_text(
                json.dumps(stale_binding_payload), encoding="utf-8"
            )
            with patch.object(
                sys,
                "argv",
                [
                    "render_inference_config.py", "--adapter-root", str(root),
                    "--detector-seed", "0", "--mode", "qccr",
                    "--binding", str(stale_binding),
                    "--expected-binding-sha", sha256_file(stale_binding),
                    "--output", str(
                        root / "configs/hilor/generated/seed0/stale.yml"
                    ),
                ],
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "another implementation lock"
                ):
                    render_config_main()
            generated = root / "configs/hilor/generated/seed0/qccr.yml"
            with patch.object(
                sys,
                "argv",
                [
                    "render_inference_config.py", "--adapter-root", str(root),
                    "--detector-seed", "0", "--mode", "qccr",
                    "--binding", str(qccr_binding),
                    "--expected-binding-sha", sha256_file(qccr_binding),
                    "--output", str(generated),
                ],
            ):
                render_config_main()
            rendered = generated.read_text(encoding="utf-8")
            self.assertIn(sha256_file(qccr_binding), rendered)
            self.assertIn("artifacts/inference/seed0/qccr_binding.json", rendered)
            self.assertNotIn(str(root).replace("\\", "/"), rendered)

            train_capture = make_capture(
                "train", checkpoint, checkpoint_sha, count=3,
                adapter_root_path=root,
            )
            train_path = root / "train.pt"
            torch.save(train_capture, train_path)
            train_sha = sha256_file(train_path)
            qccr = frozen_payload["parameters"]
            data, ids, positives, _ = build_train_data(
                train_capture, qccr, apply_qccr, compute_competition_risk,
                build_image_features,
            )
            self.assertGreater(positives, 0)
            model, _, _, _, _, _ = train_one(
                UMRHead, data, ids, seed=0, positive_weight=1, max_epochs=2
            )
            umr_path = umr_dir / "umr.pt"
            torch.save(model.state_dict(), umr_path)
            runtime = UMR(
                checkpoint=str(umr_path), num_classes=5, num_queries=300,
                topk=300, tau=0.1, beta=0.25, **qccr,
            )
            record = train_capture["records"][0]
            result = runtime(
                record["pred_logits"].unsqueeze(0),
                record["pred_boxes"].unsqueeze(0),
                (record["pred_boxes"] * 100.0).unsqueeze(0),
                record["hL"].unsqueeze(0),
            )
            self.assertEqual(tuple(result.shape), (1, 300, 5))

            selection = {
                "schema": "rtdetr-hilor-umr-val-selection-v1",
                "decision": "VAL_GO",
                "test_accessed": False,
                "official_test_accessed": False,
                "protocol": "CDW1630 Train candidates / Val-only selection",
                "implementation_lock_sha256": frozen_payload[
                    "implementation_lock_sha256"
                ],
                "detector_seed": 0,
                "feature_schema": "umr-25d-rtdetr-final-hidden-v1",
                "source_checkpoint_sha256": checkpoint_sha,
                "train_capture_sha256": train_sha,
                "val_capture_sha256": val_sha,
                "source_data_split_lock_sha256": DATA_LOCK_SHA,
                "qccr_artifact_sha256": sha256_file(frozen),
                "qccr_source_capture_sha256": val_sha,
                "runtime_source_sha256": RUNTIME_HASHES,
                "selected_model": {
                    "path": str(umr_path),
                    "sha256": sha256_file(umr_path),
                },
                "selected": {"tau": 0.1, "beta": 0.25},
            }
            selection_path = umr_dir / "selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            full_binding = inference / "qccr_umr_binding.json"
            with patch.object(
                sys,
                "argv",
                [
                    "freeze_inference_binding.py", "--adapter-root", str(root),
                    "--source-root", str(SOURCE_ROOT), "--mode", "qccr-umr",
                    "--detector-seed", "0", "--detector-checkpoint", str(checkpoint),
                    "--expected-detector-sha", checkpoint_sha,
                    "--train-capture", str(train_path),
                    "--expected-train-capture-sha", train_sha,
                    "--val-capture", str(val_path),
                    "--expected-val-capture-sha", val_sha,
                    "--qccr-artifact", str(frozen),
                    "--expected-qccr-sha", sha256_file(frozen),
                    "--umr-checkpoint", str(umr_path),
                    "--expected-umr-sha", sha256_file(umr_path),
                    "--umr-selection", str(selection_path),
                    "--expected-umr-selection-sha", sha256_file(selection_path),
                    "--output", str(full_binding),
                ],
            ):
                freeze_binding_main()
            test_tainted = dict(selection)
            test_tainted["official_test_accessed"] = True
            test_tainted_path = umr_dir / "test_tainted_selection.json"
            test_tainted_path.write_text(
                json.dumps(test_tainted), encoding="utf-8"
            )
            with patch.object(
                sys,
                "argv",
                [
                    "freeze_inference_binding.py", "--adapter-root", str(root),
                    "--source-root", str(SOURCE_ROOT), "--mode", "qccr-umr",
                    "--detector-seed", "0", "--detector-checkpoint", str(checkpoint),
                    "--expected-detector-sha", checkpoint_sha,
                    "--train-capture", str(train_path),
                    "--expected-train-capture-sha", train_sha,
                    "--val-capture", str(val_path),
                    "--expected-val-capture-sha", val_sha,
                    "--qccr-artifact", str(frozen),
                    "--expected-qccr-sha", sha256_file(frozen),
                    "--umr-checkpoint", str(umr_path),
                    "--expected-umr-sha", sha256_file(umr_path),
                    "--umr-selection", str(test_tainted_path),
                    "--expected-umr-selection-sha", sha256_file(test_tainted_path),
                    "--output", str(inference / "test_tainted_binding.json"),
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "Test exclusion"):
                    freeze_binding_main()
            bound = load_bound_inference_artifact(
                {"path": str(full_binding), "sha256": sha256_file(full_binding)},
                require_umr=True,
            )
            self.assertEqual(bound["detector_checkpoint_sha256"], checkpoint_sha)
            self.assertEqual(bound["umr"]["tau"], 0.1)
            self.assertNotIn("training_report", bound["umr"])
            self.assertNotIn("grids", bound["umr"])

    def test_hardened_capture_rejects_noncanonical_data_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_adapter_root(Path(directory))
            checkpoint = root / "runs/seed0/best_val.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"detector")
            capture = make_capture(
                "val", checkpoint, sha256_file(checkpoint), count=1,
                adapter_root_path=root,
            )
            capture["data_split_lock_sha256"] = "2" * 64
            path = root / "bad.pt"
            torch.save(capture, path)
            with self.assertRaisesRegex(RuntimeError, "protocol-bound"):
                load_capture(
                    path, expected_sha=sha256_file(path), split="val",
                    expected_checkpoint_sha=sha256_file(checkpoint),
                    adapter_root_path=root, detector_seed=0,
                )

    def test_hardened_capture_rejects_image_tree_provenance_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_adapter_root(Path(directory))
            checkpoint = root / "runs/seed0/best_val.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"detector")
            checkpoint_sha = sha256_file(checkpoint)
            capture = make_capture(
                "val", checkpoint, checkpoint_sha, adapter_root_path=root
            )
            capture["image_tree_sha256"] = "2" * 64
            path = root / "bad_tree.pt"
            torch.save(capture, path)
            with self.assertRaisesRegex(RuntimeError, "image-tree provenance"):
                load_capture(
                    path,
                    expected_sha=sha256_file(path),
                    split="val",
                    expected_checkpoint_sha=checkpoint_sha,
                    adapter_root_path=root,
                    detector_seed=0,
                )

    def test_hardened_capture_rejects_forged_record_image_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_adapter_root(Path(directory))
            checkpoint = root / "runs/seed0/best_val.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"detector")
            checkpoint_sha = sha256_file(checkpoint)
            capture = make_capture(
                "val", checkpoint, checkpoint_sha, adapter_root_path=root
            )
            capture["records"][0]["image_id"] = 9999
            path = root / "forged_ids.pt"
            torch.save(capture, path)
            with self.assertRaisesRegex(RuntimeError, "record image ids"):
                load_capture(
                    path,
                    expected_sha=sha256_file(path),
                    split="val",
                    expected_checkpoint_sha=checkpoint_sha,
                    adapter_root_path=root,
                    detector_seed=0,
                )

    def test_hardened_capture_rejects_missing_image_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_adapter_root(Path(directory))
            checkpoint = root / "runs/seed0/best_val.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"detector")
            checkpoint_sha = sha256_file(checkpoint)
            capture = make_capture(
                "val", checkpoint, checkpoint_sha, adapter_root_path=root
            )
            capture["records"].pop()
            capture["summary"]["image_count"] = len(capture["records"])
            path = root / "missing_record.pt"
            torch.save(capture, path)
            with self.assertRaisesRegex(RuntimeError, "canonical image_count"):
                load_capture(
                    path,
                    expected_sha=sha256_file(path),
                    split="val",
                    expected_checkpoint_sha=checkpoint_sha,
                    adapter_root_path=root,
                    detector_seed=0,
                )

    def test_hardened_capture_rejects_old_capture_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_adapter_root(Path(directory))
            checkpoint = root / "runs/seed0/best_val.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"detector")
            checkpoint_sha = sha256_file(checkpoint)
            capture = make_capture(
                "val", checkpoint, checkpoint_sha, adapter_root_path=root
            )
            capture["config_sha256"] = "2" * 64
            path = root / "old_config.pt"
            torch.save(capture, path)
            with self.assertRaisesRegex(RuntimeError, "config SHA-256"):
                load_capture(
                    path,
                    expected_sha=sha256_file(path),
                    split="val",
                    expected_checkpoint_sha=checkpoint_sha,
                    adapter_root_path=root,
                    detector_seed=0,
                )

    def test_hardened_capture_rejects_old_implementation_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_adapter_root(Path(directory))
            checkpoint = root / "runs/seed0/best_val.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"detector")
            checkpoint_sha = sha256_file(checkpoint)
            capture = make_capture(
                "val", checkpoint, checkpoint_sha, adapter_root_path=root
            )
            capture["training_run"]["implementation_lock_sha256"] = "2" * 64
            path = root / "old_implementation.pt"
            torch.save(capture, path)
            with self.assertRaisesRegex(RuntimeError, "another implementation lock"):
                load_capture(
                    path,
                    expected_sha=sha256_file(path),
                    split="val",
                    expected_checkpoint_sha=checkpoint_sha,
                    adapter_root_path=root,
                    detector_seed=0,
                )

    def test_hardened_capture_rejects_image_tree_summary_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_adapter_root(Path(directory))
            checkpoint = root / "runs/seed0/best_val.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"detector")
            checkpoint_sha = sha256_file(checkpoint)
            capture = make_capture(
                "val", checkpoint, checkpoint_sha, adapter_root_path=root
            )
            capture["summary"]["image_tree_sha256"] = "2" * 64
            path = root / "bad_tree_summary.pt"
            torch.save(capture, path)
            with self.assertRaisesRegex(RuntimeError, "image-tree summary"):
                load_capture(
                    path,
                    expected_sha=sha256_file(path),
                    split="val",
                    expected_checkpoint_sha=checkpoint_sha,
                    adapter_root_path=root,
                    detector_seed=0,
                )


if __name__ == "__main__":
    unittest.main()
