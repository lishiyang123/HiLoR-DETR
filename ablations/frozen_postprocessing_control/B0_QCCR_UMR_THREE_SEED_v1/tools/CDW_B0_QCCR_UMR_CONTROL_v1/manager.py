#!/usr/bin/env python3
"""Three-seed B0 + QCCR + UMR frozen-postprocessing control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import (
    atomic_json,
    atomic_text,
    package_root,
    read_json,
    require,
    sha256,
    state_root,
    yaml_path,
)


PROTOCOL_PATH = package_root() / "protocol" / "B0_POSTPROCESS_CONTROL.json"
SEEDS = (0, 1, 2)
METRICS = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
    "AP80", "AP85", "AP90", "AP95",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def protocol():
    return read_json(PROTOCOL_PATH)


def locate_final_package(project):
    candidates = (
        project / "HiLoR-DETR" / "main_host",
        project / "main_host",
    )
    for candidate in candidates:
        if (candidate / "checkpoints" / "CDW1630" / "B0").is_dir():
            return candidate.resolve()
    raise RuntimeError("Unable to locate main_host")


def locate_source(main):
    candidates = (
        main / "source" / "FINAL_CODE",
        main / "source" / "FINAL_CODE",
    )
    required = (
        "train.py",
        "tools/calibration/common/capture_train_val.py",
        "tools/calibration/common/train_umr_select_val.py",
        "tools/calibration/common/freeze_qccr.py",
        "tools/calibration/fixed_protocol/fix_qccr.py",
        "tools/calibration/fixed_protocol/fix_umr.py",
        "tools/qccr/build_qccr_targets.py",
        "tools/qccr/fit_qccr.py",
        "tools/qccr/evaluate_qccr_oof_val.py",
    )
    for candidate in candidates:
        if all((candidate / relative).is_file() for relative in required):
            return candidate.resolve()
    raise RuntimeError("Unable to locate runnable clean source")


def locate_dataset(main, project):
    candidates = (
        main / "datasets" / "CDW1630",
        project / "data" / "CDW1630_official" / "coco_deim",
    )
    for candidate in candidates:
        if (candidate / "train2017").is_dir() and (
            candidate / "annotations" / "instances_val2017.json"
        ).is_file():
            return candidate.resolve()
    raise RuntimeError("Unable to locate CDW1630 dataset")


def source_manifest(source):
    for name in ("CLEAN_SOURCE_MANIFEST.json", "SOURCE_MANIFEST.json"):
        candidate = source / name
        if candidate.is_file():
            return candidate
    raise RuntimeError("Clean-source manifest is missing")


def checkpoint_for(main, seed):
    root = main / "checkpoints" / "CDW1630" / "B0" / ("seed%d" % seed)
    candidates = sorted(path for path in root.glob("*.pth") if path.is_file())
    require(len(candidates) == 1, "Expected exactly one B0 checkpoint in %s" % root)
    return candidates[0].resolve()


def baseline_eval_for(main, seed):
    path = (
        main
        / "results"
        / "CDW1630"
        / "B0"
        / ("seed%d" % seed)
        / "official_test"
        / "eval.pth"
    )
    require(path.is_file(), "Missing formal B0 Test result: %s" % path)
    return path.resolve()


def expected_checkpoint_sha(main, seed):
    metadata = read_json(
        main
        / "training_records"
        / "CDW1630"
        / "B0"
        / ("seed%d" % seed)
        / "metadata.json"
    )
    if seed == 0:
        return str(metadata["canonical_checkpoint"]["sha256"])
    return str(metadata["checkpoint"]["sha256"])


def annotation_info(path, expected_images):
    value = read_json(path)
    require(len(value.get("images", [])) == expected_images, "Unexpected image count: %s" % path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "image_count": len(value.get("images", [])),
        "annotation_count": len(value.get("annotations", [])),
    }


def directory_record(path):
    import hashlib

    path = path.resolve()
    require(path.is_dir(), "Missing image directory: %s" % path)
    digest = hashlib.sha256()
    count = 0
    total = 0
    for item in sorted((entry for entry in path.rglob("*") if entry.is_file()), key=lambda p: p.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        content = sha256(item)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content.encode("ascii"))
        digest.update(b"\n")
        count += 1
        total += size
    return {
        "path": str(path),
        "file_count": count,
        "total_size": total,
        "tree_sha256": digest.hexdigest(),
    }


def verify_directory_record(record):
    require(directory_record(Path(record["path"])) == record, "Dataset image tree changed: %s" % record["path"])


def paths(project):
    main = locate_final_package(project)
    source = locate_source(main)
    dataset = locate_dataset(main, project)
    state = state_root(project)
    return {
        "project": project,
        "main": main,
        "source": source,
        "dataset": dataset,
        "state": state,
        "fit_lock": state / "FIT_INPUT_LOCK.json",
        "fit_status": state / "FIT_STATUS.json",
        "test_lock": state / "TEST_INPUT_LOCK.json",
        "test_status": state / "TEST_STATUS.json",
        "tables": state / "paper_tables",
    }


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def update_status(path, **updates):
    value = read_json(path) if path.is_file() else {}
    value.update(updates)
    value["updated_utc"] = utc_now()
    atomic_json(path, value)


def gpu_idle():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    require(not result.stdout.strip(), "GPU compute process already exists:\n%s" % result.stdout)


def base_config_text(source, image_dir, annotation, output):
    return "\n".join(
        [
            "__include__: [%s]" % yaml_path(source / "configs/deim_dfine/deim_hgnetv2_m_coco.yml"),
            "output_dir: %s" % yaml_path(output),
            "",
            "num_classes: 5",
            "remap_mscoco_category: False",
            "",
            "HGNetv2:",
            "  pretrained: False",
            "",
            "DFINETransformer:",
            "  ufd_enabled: False",
            "",
            "DEIMCriterion:",
            "  ufd_enabled: False",
            "  lcct_enabled: False",
            "",
            "PostProcessor:",
            "  qccr_enabled: False",
            "  umr_enabled: False",
            "",
            "val_dataloader:",
            "  dataset:",
            "    img_folder: %s" % yaml_path(image_dir),
            "    ann_file: %s" % yaml_path(annotation),
            "  total_batch_size: 8",
            "  num_workers: 4",
            "",
        ]
    )


def child_config_text(base, image_dir, annotation, output):
    return "\n".join(
        [
            "__include__: [%s]" % yaml_path(base),
            "output_dir: %s" % yaml_path(output),
            "val_dataloader:",
            "  dataset:",
            "    img_folder: %s" % yaml_path(image_dir),
            "    ann_file: %s" % yaml_path(annotation),
            "  total_batch_size: 8",
            "  num_workers: 4",
            "",
        ]
    )


def final_test_config_text(base, image_dir, annotation, output, qccr, umr_weight):
    parameters = read_json(qccr)["parameters"]
    p = protocol()
    return "\n".join(
        [
            "__include__: [%s]" % yaml_path(base),
            "output_dir: %s" % yaml_path(output),
            "val_dataloader:",
            "  dataset:",
            "    img_folder: %s" % yaml_path(image_dir),
            "    ann_file: %s" % yaml_path(annotation),
            "  total_batch_size: 8",
            "  num_workers: 4",
            "PostProcessor:",
            "  qccr_enabled: True",
            "  qccr_beta0: %r" % float(parameters["beta0"]),
            "  qccr_beta_positive: %r" % float(parameters["beta_positive"]),
            "  qccr_beta_negative: %r" % float(parameters["beta_negative"]),
            "  qccr_duplicate_penalty: %r" % float(parameters["duplicate_penalty"]),
            "  umr_enabled: True",
            "  umr_topk: %d" % int(p["umr_topk"]),
            "  umr_tau: %r" % float(p["umr_tau"]),
            "  umr_beta: %r" % float(p["umr_beta"]),
            "  umr_checkpoint: %s" % yaml_path(umr_weight),
            "",
        ]
    )


def make_fit_lock(project):
    p = paths(project)
    train_ann = p["dataset"] / "annotations" / "instances_train2017.json"
    val_ann = p["dataset"] / "annotations" / "instances_val2017.json"
    lock = {
        "schema": "cdw1630-b0-postprocess-fit-input-lock-v1",
        "project": str(project),
        "main_package": str(p["main"]),
        "source": {
            "path": str(p["source"]),
            "manifest": str(source_manifest(p["source"])),
            "manifest_sha256": sha256(source_manifest(p["source"])),
        },
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        "dataset": {
            "train_images": directory_record(p["dataset"] / "train2017"),
            "train_annotation": annotation_info(train_ann, 1144),
            "val_images": directory_record(p["dataset"] / "val2017"),
            "val_annotation": annotation_info(val_ann, 249),
        },
        "seeds": {},
        "test_accessed": False,
        "created_utc": utc_now(),
    }
    for seed in SEEDS:
        checkpoint = checkpoint_for(p["main"], seed)
        observed = sha256(checkpoint)
        require(observed == expected_checkpoint_sha(p["main"], seed), "B0 checkpoint/metadata SHA mismatch: seed%d" % seed)
        lock["seeds"][str(seed)] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": observed,
        }
    return lock


def verify_fit_lock(lock):
    require(lock.get("test_accessed") is False, "Fit lock accessed Test")
    require(sha256(Path(lock["protocol"]["path"])) == lock["protocol"]["sha256"], "Protocol changed")
    source = Path(lock["source"]["path"])
    require(sha256(Path(lock["source"]["manifest"])) == lock["source"]["manifest_sha256"], "Source changed")
    for key in ("train_images", "val_images"):
        verify_directory_record(lock["dataset"][key])
    for key in ("train_annotation", "val_annotation"):
        record = lock["dataset"][key]
        require(sha256(Path(record["path"])) == record["sha256"], "Annotation changed")
    for seed in SEEDS:
        record = lock["seeds"][str(seed)]
        require(sha256(Path(record["checkpoint"])) == record["checkpoint_sha256"], "Checkpoint changed: seed%d" % seed)
    return source


def materialize_fit_configs(project, lock):
    p = paths(project)
    source = Path(lock["source"]["path"])
    train_images = Path(lock["dataset"]["train_images"]["path"])
    val_images = Path(lock["dataset"]["val_images"]["path"])
    train_ann = Path(lock["dataset"]["train_annotation"]["path"])
    val_ann = Path(lock["dataset"]["val_annotation"]["path"])
    for seed in SEEDS:
        root = p["state"] / "fit" / ("seed%d" % seed)
        configs = root / "configs"
        base = configs / "b0_base.yml"
        atomic_text(base, base_config_text(source, val_images, val_ann, root / "native"))
        atomic_text(
            configs / "capture_train.yml",
            child_config_text(base, train_images, train_ann, root / "capture_train"),
        )
        atomic_text(
            configs / "capture_val.yml",
            child_config_text(base, val_images, val_ann, root / "capture_val"),
        )


def preflight(project, prepare=False):
    p = protocol()
    require(p["detector_seeds"] == [0, 1, 2], "Protocol seeds changed")
    require(p["comparison_rows"] == ["B0", "B0_QCCR_UMR"], "Protocol rows changed")
    require(float(p["qccr_lambda_reg"]) == 0.001, "QCCR lambda changed")
    require((int(p["umr_positive_weight"]), int(p["umr_head_seed"]), float(p["umr_tau"]), float(p["umr_beta"]), int(p["umr_topk"])) == (4, 0, 0.0, 2.0, 300), "UMR tuple changed")
    lock = make_fit_lock(project)
    verify_fit_lock(lock)
    source = Path(lock["source"]["path"])
    verifier = source / "tools" / "reproduction" / "verify_source.py"
    if verifier.is_file():
        subprocess.run([sys.executable, "-B", str(verifier), "--source", str(source)], check=True)
    if prepare:
        root = state_root(project)
        root.mkdir(parents=True, exist_ok=True)
        if (root / "FIT_INPUT_LOCK.json").is_file():
            existing = read_json(root / "FIT_INPUT_LOCK.json")
            existing.pop("created_utc", None)
            candidate = dict(lock)
            candidate.pop("created_utc", None)
            require(existing == candidate, "Existing fit input lock differs")
        else:
            atomic_json(root / "FIT_INPUT_LOCK.json", lock)
        materialize_fit_configs(project, lock)
    print("===== B0 + QCCR + UMR THREE-SEED CONTROL PREFLIGHT =====")
    print("Detector seeds : 0 / 1 / 2 existing Val-selected B0 checkpoints")
    print("Detector train : NO")
    print("QCCR           : checkpoint-specific Val fit / fixed lambda=1e-3")
    print("UMR            : checkpoint-specific Train fit / pw4 / head seed0")
    print("Fusion          : tau=0 / beta=2 / topk=300")
    print("Test access     : NO")
    print("PREPARE         : %s" % ("PASS" if prepare else "NOT REQUESTED"))
    print("PREFLIGHT: PASS")
    return lock


def reset_bounded(path, parent):
    if not path.exists():
        return
    path = path.resolve()
    parent = parent.resolve()
    require(path.parent == parent, "Unsafe cleanup target: %s" % path)
    shutil.rmtree(path)


def run_logged(command, cwd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\nCOMMAND " + " ".join(str(value) for value in command) + "\n")
        handle.flush()
        result = subprocess.run(
            [str(value) for value in command],
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
        )
    require(result.returncode == 0, "Command exited %d; inspect %s" % (result.returncode, log))


def file_record(path):
    require(path.is_file() and path.stat().st_size > 0, "Missing output: %s" % path)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256(path)}


def verify_file_record(record):
    path = Path(record["path"])
    require(path.is_file(), "Bound file missing: %s" % path)
    require(path.stat().st_size == int(record["size"]), "Bound file size changed: %s" % path)
    require(sha256(path) == record["sha256"], "Bound file hash changed: %s" % path)


def capture_seed(project, lock, seed):
    p = paths(project)
    source = Path(lock["source"]["path"])
    root = p["state"] / "fit" / ("seed%d" % seed)
    output = root / "capture"
    binding_path = root / "CAPTURE_BINDING.json"
    if binding_path.is_file():
        value = read_json(binding_path)
        for record in value["files"].values():
            verify_file_record(record)
        return value
    reset_bounded(output, root)
    checkpoint = Path(lock["seeds"][str(seed)]["checkpoint"])
    prefix = "b0_seed%d" % seed
    command = [
        sys.executable,
        "-u",
        "-B",
        str(source / "tools/calibration/common/capture_train_val.py"),
        "--branch-root",
        str(source),
        "--checkpoint",
        str(checkpoint),
        "--train-config",
        str(root / "configs/capture_train.yml"),
        "--val-config",
        str(root / "configs/capture_val.yml"),
        "--output-dir",
        str(output),
        "--expected-checkpoint-sha",
        lock["seeds"][str(seed)]["checkpoint_sha256"],
        "--num-classes",
        "5",
        "--capture-prefix",
        prefix,
    ]
    run_logged(command, source, root / "logs/capture.log")
    files = {
        "train_capture": file_record(output / (prefix + "_train_capture.pt")),
        "train_baseline_eval": file_record(output / (prefix + "_train_baseline_eval.pth")),
        "val_capture": file_record(output / (prefix + "_val_capture.pt")),
        "val_baseline_eval": file_record(output / (prefix + "_val_baseline_eval.pth")),
        "summary": file_record(output / "CAPTURE_SUMMARY.json"),
    }
    value = {
        "schema": "cdw1630-b0-train-val-capture-binding-v1",
        "seed": seed,
        "checkpoint_sha256": lock["seeds"][str(seed)]["checkpoint_sha256"],
        "files": files,
        "test_accessed": False,
    }
    atomic_json(binding_path, value)
    return value


def fit_qccr(project, lock, seed, capture):
    p = paths(project)
    source = Path(lock["source"]["path"])
    root = p["state"] / "fit" / ("seed%d" % seed)
    output = root / "qccr"
    binding_path = root / "QCCR_BINDING.json"
    if binding_path.is_file():
        value = read_json(binding_path)
        for record in value["files"].values():
            verify_file_record(record)
        return value
    reset_bounded(output, root)
    output.mkdir(parents=True, exist_ok=True)
    val_capture = Path(capture["files"]["val_capture"]["path"])
    val_eval = Path(capture["files"]["val_baseline_eval"]["path"])
    target = output / "qccr_targets.pt"
    target_summary = output / "QCCR_TARGET_SUMMARY.json"
    fits = output / "qccr_fits.json"
    oof_dir = output / "oof"
    raw_oof = oof_dir / "oof_summary.json"
    fixed_oof = output / "oof_summary_fixed.json"
    artifact = output / "qccr_params.json"
    log = root / "logs/qccr.log"
    run_logged(
        [sys.executable, "-B", str(source / "tools/qccr/build_qccr_targets.py"), "--capture", str(val_capture), "--output", str(target), "--summary", str(target_summary)],
        source,
        log,
    )
    run_logged(
        [sys.executable, "-B", str(package_root() / "fixed_qccr_fit.py"), "--official-script", str(source / "tools/qccr/fit_qccr.py"), "--targets", str(target), "--output", str(fits), "--seed", "0", "--folds", "5"],
        source,
        log,
    )
    run_logged(
        [sys.executable, "-B", str(source / "tools/qccr/evaluate_qccr_oof_val.py"), "--capture", str(val_capture), "--baseline-eval", str(val_eval), "--fits", str(fits), "--annotation", lock["dataset"]["val_annotation"]["path"], "--output-dir", str(oof_dir)],
        source,
        log,
    )
    run_logged(
        [sys.executable, "-B", str(source / "tools/calibration/fixed_protocol/fix_qccr.py"), "--summary", str(raw_oof), "--fits", str(fits), "--policy", str(PROTOCOL_PATH), "--output", str(fixed_oof)],
        source,
        log,
    )
    checkpoint = Path(lock["seeds"][str(seed)]["checkpoint"])
    run_logged(
        [sys.executable, "-B", str(source / "tools/calibration/common/freeze_qccr.py"), "--oof-summary", str(fixed_oof), "--fits", str(fits), "--checkpoint", str(checkpoint), "--output", str(artifact), "--expected-checkpoint-sha", lock["seeds"][str(seed)]["checkpoint_sha256"], "--dataset-name", "CDW1630", "--source-method", "B0_seed%d" % seed],
        source,
        log,
    )
    frozen = read_json(artifact)
    require(abs(float(frozen["selected_lambda_reg"]) - 0.001) < 1e-12, "Frozen QCCR lambda changed")
    require(frozen["source_checkpoint_sha256"] == lock["seeds"][str(seed)]["checkpoint_sha256"], "QCCR/checkpoint mismatch")
    files = {
        "targets": file_record(target),
        "target_summary": file_record(target_summary),
        "fits": file_record(fits),
        "fixed_oof": file_record(fixed_oof),
        "artifact": file_record(artifact),
    }
    value = {
        "schema": "cdw1630-b0-fixed-qccr-binding-v1",
        "seed": seed,
        "lambda_reg": 0.001,
        "checkpoint_sha256": lock["seeds"][str(seed)]["checkpoint_sha256"],
        "files": files,
        "test_accessed": False,
    }
    atomic_json(binding_path, value)
    return value


def fit_umr(project, lock, seed, capture, qccr):
    p = paths(project)
    source = Path(lock["source"]["path"])
    root = p["state"] / "fit" / ("seed%d" % seed)
    output = root / "umr"
    binding_path = root / "UMR_BINDING.json"
    if binding_path.is_file():
        value = read_json(binding_path)
        for record in value["files"].values():
            verify_file_record(record)
        return value
    reset_bounded(output, root)
    candidates = output / "candidates"
    checkpoint = Path(lock["seeds"][str(seed)]["checkpoint"])
    qccr_artifact = Path(qccr["files"]["artifact"]["path"])
    log = root / "logs/umr.log"
    run_logged(
        [
            sys.executable,
            "-B",
            str(package_root() / "fixed_umr_fit.py"),
            "--official-script",
            str(source / "tools/calibration/common/train_umr_select_val.py"),
            "--source-root",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--train-capture",
            capture["files"]["train_capture"]["path"],
            "--val-capture",
            capture["files"]["val_capture"]["path"],
            "--qccr-artifact",
            str(qccr_artifact),
            "--val-annotation",
            lock["dataset"]["val_annotation"]["path"],
            "--output-dir",
            str(candidates),
            "--expected-checkpoint-sha",
            lock["seeds"][str(seed)]["checkpoint_sha256"],
            "--num-classes",
            "5",
            "--dataset-name",
            "CDW1630_B0_seed%d" % seed,
        ],
        source,
        log,
    )
    fixed = output / "UMR_FIXED.json"
    run_logged(
        [sys.executable, "-B", str(source / "tools/calibration/fixed_protocol/fix_umr.py"), "--candidate-dir", str(candidates), "--policy", str(PROTOCOL_PATH), "--checkpoint-sha", lock["seeds"][str(seed)]["checkpoint_sha256"], "--output", str(fixed)],
        source,
        log,
    )
    weight = candidates / "models" / "umr_seed0_pw4_best.pt"
    provenance = candidates / "results" / "INPUT_PROVENANCE.json"
    grid = candidates / "results" / "VAL_GRID_MEAN.csv"
    metadata_path = output / "UMR_METADATA.json"
    metadata = {
        "schema": "cdw1630-b0-fixed-umr-metadata-v1",
        "detector_seed": seed,
        "head_seed": 0,
        "positive_weight": 4,
        "fusion": {"tau": 0.0, "beta": 2.0, "topk": 300},
        "weight_sha256": sha256(weight),
        "source_checkpoint_sha256": lock["seeds"][str(seed)]["checkpoint_sha256"],
        "qccr_sha256": sha256(qccr_artifact),
        "fit_split": "Train",
        "diagnostic_split": "Val",
        "test_used_for_selection": False,
    }
    atomic_json(metadata_path, metadata)
    files = {
        "weight": file_record(weight),
        "metadata": file_record(metadata_path),
        "fixed_decision": file_record(fixed),
        "provenance": file_record(provenance),
        "fixed_val_row": file_record(grid),
    }
    value = {
        "schema": "cdw1630-b0-fixed-umr-binding-v1",
        "seed": seed,
        "checkpoint_sha256": lock["seeds"][str(seed)]["checkpoint_sha256"],
        "qccr_sha256": sha256(qccr_artifact),
        "tuple": {"positive_weight": 4, "head_seed": 0, "tau": 0.0, "beta": 2.0, "topk": 300},
        "files": files,
        "test_accessed": False,
    }
    atomic_json(binding_path, value)
    return value


def frozen_binding_path(project, seed):
    return state_root(project) / "fit" / ("seed%d" % seed) / "FROZEN_BINDING.json"


def build_frozen_binding(project, lock, seed, capture, qccr, umr):
    path = frozen_binding_path(project, seed)
    value = {
        "schema": "cdw1630-b0-qccr-umr-frozen-binding-v1",
        "seed": seed,
        "checkpoint": lock["seeds"][str(seed)],
        "capture_binding": {"path": str((path.parent / "CAPTURE_BINDING.json").resolve()), "sha256": sha256(path.parent / "CAPTURE_BINDING.json")},
        "qccr_binding": {"path": str((path.parent / "QCCR_BINDING.json").resolve()), "sha256": sha256(path.parent / "QCCR_BINDING.json")},
        "umr_binding": {"path": str((path.parent / "UMR_BINDING.json").resolve()), "sha256": sha256(path.parent / "UMR_BINDING.json")},
        "qccr_artifact": qccr["files"]["artifact"],
        "umr_weight": umr["files"]["weight"],
        "umr_metadata": umr["files"]["metadata"],
        "fixed_protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        "test_accessed": False,
        "frozen_utc": utc_now(),
    }
    if path.is_file():
        old = read_json(path)
        old.pop("frozen_utc", None)
        candidate = dict(value)
        candidate.pop("frozen_utc", None)
        require(old == candidate, "Frozen binding changed: seed%d" % seed)
    else:
        atomic_json(path, value)
    return value


def verify_frozen_seed(project, lock, seed):
    path = frozen_binding_path(project, seed)
    value = read_json(path)
    require(value["seed"] == seed and value.get("test_accessed") is False, "Invalid frozen seed binding")
    require(value["checkpoint"]["checkpoint_sha256"] == lock["seeds"][str(seed)]["checkpoint_sha256"], "Frozen checkpoint mismatch")
    for key in ("capture_binding", "qccr_binding", "umr_binding", "qccr_artifact", "umr_weight", "umr_metadata", "fixed_protocol"):
        record = value[key]
        require(sha256(Path(record["path"])) == record["sha256"], "Frozen asset changed: %s" % key)
    return value


def run_fit(project):
    p = paths(project)
    if not p["fit_lock"].is_file():
        preflight(project, prepare=True)
    lock = read_json(p["fit_lock"])
    verify_fit_lock(lock)
    completed = []
    try:
        for seed in SEEDS:
            update_status(p["fit_status"], state="RUNNING", pid=os.getpid(), current="seed%d" % seed, completed=completed, pending=[value for value in SEEDS if value not in completed], error=None)
            capture = capture_seed(project, lock, seed)
            qccr = fit_qccr(project, lock, seed, capture)
            umr = fit_umr(project, lock, seed, capture, qccr)
            build_frozen_binding(project, lock, seed, capture, qccr, umr)
            verify_frozen_seed(project, lock, seed)
            completed.append(seed)
            print("seed%d B0-specific QCCR/UMR freeze: PASS" % seed, flush=True)
        update_status(p["fit_status"], state="COMPLETE", pid=os.getpid(), current=None, completed=list(SEEDS), pending=[], error=None)
        print("B0 + QCCR + UMR THREE-SEED FIT/FREEZE: COMPLETE")
    except BaseException as exc:
        update_status(p["fit_status"], state="FAILED", pid=os.getpid(), completed=completed, pending=[value for value in SEEDS if value not in completed], error="%s: %s" % (type(exc).__name__, exc))
        raise


def verify_frozen(project):
    p = paths(project)
    require(p["fit_lock"].is_file(), "Fit input lock is missing")
    lock = read_json(p["fit_lock"])
    verify_fit_lock(lock)
    for seed in SEEDS:
        verify_frozen_seed(project, lock, seed)
    status = read_json(p["fit_status"])
    require(status.get("state") == "COMPLETE" and status.get("completed") == list(SEEDS), "Fit queue is incomplete")
    print("===== B0 + QCCR + UMR FROZEN-ASSET VERIFICATION =====")
    print("Detector seeds : 3/3")
    print("QCCR artifacts : 3/3 checkpoint-specific / lambda=1e-3")
    print("UMR heads      : 3/3 checkpoint-specific / pw4 / head seed0")
    print("Fusion tuple   : tau=0 / beta=2 / topk=300")
    print("Detector train : NO")
    print("Test access    : NO")
    print("VERIFICATION: PASS")


def make_test_lock(project):
    verify_frozen(project)
    p = paths(project)
    test_ann = p["dataset"] / "annotations" / "instances_test2017.json"
    value = {
        "schema": "cdw1630-b0-postprocess-test-input-lock-v1",
        "fit_input_lock": {"path": str(p["fit_lock"]), "sha256": sha256(p["fit_lock"])},
        "test_images": directory_record(p["dataset"] / "test2017"),
        "test_annotation": annotation_info(test_ann, 237),
        "seeds": {},
        "test_role": "reporting only after all three seeds were frozen",
        "test_used_for_parameter_selection": False,
        "created_utc": utc_now(),
    }
    for seed in SEEDS:
        frozen = verify_frozen_seed(project, read_json(p["fit_lock"]), seed)
        value["seeds"][str(seed)] = {
            "frozen_binding": {"path": str(frozen_binding_path(project, seed)), "sha256": sha256(frozen_binding_path(project, seed))},
            "baseline_eval": file_record(baseline_eval_for(p["main"], seed)),
        }
    return value


def prepare_test(project):
    p = paths(project)
    value = make_test_lock(project)
    if p["test_lock"].is_file():
        old = read_json(p["test_lock"])
        old.pop("created_utc", None)
        candidate = dict(value)
        candidate.pop("created_utc", None)
        require(old == candidate, "Existing Test input lock differs")
    else:
        atomic_json(p["test_lock"], value)
    for seed in SEEDS:
        frozen = read_json(frozen_binding_path(project, seed))
        base = p["state"] / "fit" / ("seed%d" % seed) / "configs" / "b0_base.yml"
        test_root = p["state"] / "test" / ("seed%d" % seed)
        config = test_root / "config" / "b0_qccr_umr_test.yml"
        atomic_text(
            config,
            final_test_config_text(
                base,
                Path(value["test_images"]["path"]),
                Path(value["test_annotation"]["path"]),
                test_root / "output",
                Path(frozen["qccr_artifact"]["path"]),
                Path(frozen["umr_weight"]["path"]),
            ),
        )
    print("===== B0 + QCCR + UMR FORMAL TEST PREPARATION =====")
    print("Frozen seeds : 3/3 verified before Test binding")
    print("Test images  : 237 / SHA tree bound")
    print("Test role    : reporting only")
    print("PREPARATION: PASS")
    return value


def extract_metrics(path):
    import numpy as np
    import torch

    require(path.is_file() and path.stat().st_size > 0, "Missing eval.pth: %s" % path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "eval" in payload and "precision" not in payload:
        payload = payload["eval"]
    require(isinstance(payload, dict) and {"precision", "recall", "params"} <= set(payload), "Unrecognized eval payload")
    precision = np.asarray(payload["precision"], dtype=np.float64)
    recall = np.asarray(payload["recall"], dtype=np.float64)
    params = payload["params"]
    field = (lambda key: params[key]) if isinstance(params, dict) else (lambda key: getattr(params, key))
    ious = np.asarray(field("iouThrs"), dtype=np.float64)
    areas = list(field("areaRngLbl"))
    max_dets = [int(value) for value in field("maxDets")]

    def mean_valid(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[values > -1]
        return None if values.size == 0 else float(values.mean())

    def threshold(value):
        found = np.flatnonzero(np.isclose(ious, value, rtol=0, atol=1e-12))
        require(len(found) == 1, "Missing IoU threshold: %s" % value)
        return int(found[0])

    def ap(iou=None, area="all"):
        area_index = areas.index(area)
        max_index = max_dets.index(100)
        values = precision[:, :, :, area_index, max_index]
        if iou is not None:
            values = precision[threshold(iou), :, :, area_index, max_index]
        return mean_valid(values)

    def ar(area="all", max_det=100):
        return mean_valid(recall[:, :, areas.index(area), max_dets.index(max_det)])

    result = {
        "AP": ap(), "AP50": ap(0.50), "AP75": ap(0.75),
        "APs": ap(area="small"), "APm": ap(area="medium"), "APl": ap(area="large"),
        "AR1": ar(max_det=1), "AR10": ar(max_det=10), "AR100": ar(max_det=100),
        "ARs": ar(area="small"), "ARm": ar(area="medium"), "ARl": ar(area="large"),
        "AP80": ap(0.80), "AP85": ap(0.85), "AP90": ap(0.90), "AP95": ap(0.95),
    }
    for key, value in result.items():
        require(value is None or math.isfinite(value), "Non-finite metric: %s" % key)
    return result


def metric_close(left, right):
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= 1e-10


def test_record_path(project, seed):
    return state_root(project) / "test" / ("seed%d" % seed) / "TEST_RECORD.json"


def verify_test_seed(project, seed):
    p = paths(project)
    lock = read_json(p["test_lock"])
    record = read_json(test_record_path(project, seed))
    require(record["seed"] == seed, "Test seed identity changed")
    require(record.get("test_used_for_parameter_selection") is False, "Test used for selection")
    for key in ("config", "frozen_binding", "baseline_eval", "postprocessed_eval"):
        verify_file_record(record[key])
    require(record["frozen_binding"]["sha256"] == lock["seeds"][str(seed)]["frozen_binding"]["sha256"], "Test/frozen binding mismatch")
    observed = extract_metrics(Path(record["postprocessed_eval"]["path"]))
    baseline = extract_metrics(Path(record["baseline_eval"]["path"]))
    for key in METRICS:
        require(metric_close(observed[key], record["metrics"]["B0_QCCR_UMR"][key]), "Postprocessed metric changed: %s" % key)
        require(metric_close(baseline[key], record["metrics"]["B0"][key]), "Baseline metric changed: %s" % key)
    return record


def run_test_seed(project, seed):
    p = paths(project)
    if test_record_path(project, seed).is_file():
        return verify_test_seed(project, seed)
    test_lock = read_json(p["test_lock"])
    fit_lock = read_json(p["fit_lock"])
    source = Path(fit_lock["source"]["path"])
    checkpoint = Path(fit_lock["seeds"][str(seed)]["checkpoint"])
    root = p["state"] / "test" / ("seed%d" % seed)
    output = root / "output"
    if output.exists():
        reset_bounded(output, root)
    config = root / "config" / "b0_qccr_umr_test.yml"
    command = [
        sys.executable,
        "-u",
        "-B",
        str(source / "train.py"),
        "-c",
        str(config),
        "--seed",
        str(seed),
        "--output-dir",
        str(output),
        "-r",
        str(checkpoint),
        "--test-only",
    ]
    run_logged(command, source, root / "console.log")
    post_eval = output / "eval.pth"
    baseline_eval = Path(test_lock["seeds"][str(seed)]["baseline_eval"]["path"])
    value = {
        "schema": "cdw1630-b0-qccr-umr-formal-test-record-v1",
        "seed": seed,
        "checkpoint_sha256": fit_lock["seeds"][str(seed)]["checkpoint_sha256"],
        "config": file_record(config),
        "frozen_binding": file_record(frozen_binding_path(project, seed)),
        "baseline_eval": file_record(baseline_eval),
        "postprocessed_eval": file_record(post_eval),
        "metrics": {"B0": extract_metrics(baseline_eval), "B0_QCCR_UMR": extract_metrics(post_eval)},
        "test_role": "final reporting only",
        "test_used_for_parameter_selection": False,
        "completed_utc": utc_now(),
    }
    atomic_json(test_record_path(project, seed), value)
    return verify_test_seed(project, seed)


def run_test(project):
    p = paths(project)
    if not p["test_lock"].is_file():
        prepare_test(project)
    completed = []
    try:
        for seed in SEEDS:
            update_status(p["test_status"], state="RUNNING", pid=os.getpid(), current="seed%d" % seed, completed=completed, pending=[value for value in SEEDS if value not in completed], error=None)
            record = run_test_seed(project, seed)
            completed.append(seed)
            print("seed%d B0 + QCCR + UMR Test AP=%.9f" % (seed, record["metrics"]["B0_QCCR_UMR"]["AP"]), flush=True)
        update_status(p["test_status"], state="COMPLETE", pid=os.getpid(), current=None, completed=list(SEEDS), pending=[], error=None)
        build_tables(project)
        print("B0 + QCCR + UMR THREE-SEED FORMAL TEST: COMPLETE")
    except BaseException as exc:
        update_status(p["test_status"], state="FAILED", pid=os.getpid(), completed=completed, pending=[value for value in SEEDS if value not in completed], error="%s: %s" % (type(exc).__name__, exc))
        raise


def mean_std(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def build_tables(project):
    p = paths(project)
    records = [verify_test_seed(project, seed) for seed in SEEDS]
    p["tables"].mkdir(parents=True, exist_ok=True)
    per_seed = []
    for record in records:
        baseline = record["metrics"]["B0"]
        final = record["metrics"]["B0_QCCR_UMR"]
        for name, metrics in (("B0", baseline), ("B0_QCCR_UMR", final)):
            row = {"seed": record["seed"], "row": name}
            row.update(metrics)
            row.update({"delta_%s_pp" % key: 100.0 * (metrics[key] - baseline[key]) for key in METRICS})
            per_seed.append(row)
    columns = ["seed", "row"] + list(METRICS) + ["delta_%s_pp" % key for key in METRICS]
    with (p["tables"] / "B0_POSTPROCESS_PER_SEED.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(per_seed)
    summary_rows = []
    for name in ("B0", "B0_QCCR_UMR"):
        selected = [row for row in per_seed if row["row"] == name]
        row = {"row": name, "metrics": {}, "delta_vs_B0_pp": {}}
        for key in METRICS:
            row["metrics"][key] = mean_std([float(value[key]) for value in selected])
            row["delta_vs_B0_pp"][key] = mean_std([float(value["delta_%s_pp" % key]) for value in selected])
        summary_rows.append(row)
    summary = {
        "schema": "cdw1630-b0-postprocess-three-seed-summary-v1",
        "dataset": "CDW1630 official Test",
        "detector_seeds": list(SEEDS),
        "rows": summary_rows,
        "detector_retrained": False,
        "test_used_for_parameter_selection": False,
    }
    atomic_json(p["tables"] / "B0_POSTPROCESS_MEAN_STD.json", summary)
    lines = [
        "# CDW1630 B0 frozen-postprocessing control",
        "",
        "All values are percentages; mean ± sample standard deviation across detector seeds 0/1/2.",
        "",
        "| Row | AP | Delta AP vs B0 | AP50 | AP75 | AR100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        metric = row["metrics"]
        delta = row["delta_vs_B0_pp"]
        lines.append(
            "| %s | %.3f ± %.3f | %+.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f |"
            % (
                row["row"],
                100.0 * metric["AP"]["mean"],
                100.0 * metric["AP"]["sample_std"],
                delta["AP"]["mean"],
                delta["AP"]["sample_std"],
                100.0 * metric["AP50"]["mean"],
                100.0 * metric["AP50"]["sample_std"],
                100.0 * metric["AP75"]["mean"],
                100.0 * metric["AP75"]["sample_std"],
                100.0 * metric["AR100"]["mean"],
                100.0 * metric["AR100"]["sample_std"],
            )
        )
    lines.extend(["", "QCCR and UMR were fitted separately for each frozen B0 checkpoint. Test was reporting-only.", ""])
    atomic_text(p["tables"] / "B0_POSTPROCESS_MAIN.md", "\n".join(lines))
    return summary


def write_result_manifest(project):
    p = paths(project)
    files = [PROTOCOL_PATH, p["fit_lock"], p["test_lock"]]
    files.extend(frozen_binding_path(project, seed) for seed in SEEDS)
    files.extend(test_record_path(project, seed) for seed in SEEDS)
    files.extend(sorted(path for path in p["tables"].glob("*") if path.is_file()))
    for seed in SEEDS:
        frozen = read_json(frozen_binding_path(project, seed))
        files.extend(Path(frozen[key]["path"]) for key in ("qccr_artifact", "umr_weight", "umr_metadata"))
        files.append(Path(read_json(test_record_path(project, seed))["postprocessed_eval"]["path"]))
    unique = sorted(set(path.resolve() for path in files), key=str)
    state = p["state"].resolve()
    records = []
    for path in unique:
        require(path.is_file(), "Result-manifest input missing: %s" % path)
        try:
            relative = path.relative_to(state).as_posix()
        except ValueError:
            relative = "external::" + str(path)
        records.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(p["state"] / "RESULT_MANIFEST.sha256.json", {"schema": 1, "files": records})


def verify(project):
    p = paths(project)
    verify_frozen(project)
    require(p["test_lock"].is_file(), "Test input lock is missing")
    test_lock = read_json(p["test_lock"])
    verify_directory_record(test_lock["test_images"])
    require(sha256(Path(test_lock["test_annotation"]["path"])) == test_lock["test_annotation"]["sha256"], "Test annotation changed")
    for seed in SEEDS:
        verify_test_seed(project, seed)
    status = read_json(p["test_status"])
    require(status.get("state") == "COMPLETE" and status.get("completed") == list(SEEDS), "Test queue is incomplete")
    build_tables(project)
    write_result_manifest(project)
    print("===== B0 + QCCR + UMR THREE-SEED CONTROL VERIFICATION =====")
    print("Frozen B0 checkpoints : 3/3")
    print("Checkpoint-specific QCCR/UMR: 3/3")
    print("Formal Test results   : 3/3")
    print("Rows                  : B0 / B0 + QCCR + UMR")
    print("Detector training     : NO")
    print("Test selection        : NO")
    print("Tables/manifest       : PASS")
    print("VERIFICATION: PASS")


def summary(project):
    verify(project)
    value = read_json(state_root(project) / "paper_tables" / "B0_POSTPROCESS_MEAN_STD.json")
    print("===== CDW1630 B0 FROZEN-POSTPROCESSING CONTROL / THREE-SEED TEST =====")
    print("%-20s %18s %18s %18s %18s %18s" % ("row", "AP", "dAP", "AP50", "AP75", "AR100"))
    for row in value["rows"]:
        metrics = row["metrics"]
        deltas = row["delta_vs_B0_pp"]
        print(
            "%-20s %8.3f ± %-6.3f %+8.3f ± %-6.3f %8.3f ± %-6.3f %8.3f ± %-6.3f %8.3f ± %-6.3f"
            % (
                row["row"],
                100.0 * metrics["AP"]["mean"], 100.0 * metrics["AP"]["sample_std"],
                deltas["AP"]["mean"], deltas["AP"]["sample_std"],
                100.0 * metrics["AP50"]["mean"], 100.0 * metrics["AP50"]["sample_std"],
                100.0 * metrics["AP75"]["mean"], 100.0 * metrics["AP75"]["sample_std"],
                100.0 * metrics["AR100"]["mean"], 100.0 * metrics["AR100"]["sample_std"],
            )
        )


def start_background(project, action, status_path, log_name):
    if status_path.is_file():
        old = read_json(status_path)
        require(not (old.get("state") in ("STARTING", "RUNNING") and process_alive(old.get("pid"))), "Queue is already running: PID %s" % old.get("pid"))
    gpu_idle()
    state_root(project).mkdir(parents=True, exist_ok=True)
    log = state_root(project) / log_name
    stream = log.open("a", encoding="utf-8", newline="\n")
    process = subprocess.Popen(
        [sys.executable, "-u", "-B", str(Path(__file__).resolve()), action, "--project", str(project)],
        cwd=str(package_root()),
        stdout=stream,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )
    stream.close()
    update_status(status_path, state="STARTING", pid=process.pid, current=None, completed=[], pending=list(SEEDS), error=None)
    print("Launched %s PID=%d" % (action, process.pid))


def show_status(project, kind):
    p = paths(project)
    status_path = p["fit_status"] if kind == "fit" else p["test_status"]
    print("===== B0 POSTPROCESS %s STATUS =====" % kind.upper())
    if not status_path.is_file():
        print("state: NOT_STARTED")
        return
    value = read_json(status_path)
    for key in ("state", "pid", "current", "completed", "pending", "error"):
        print("%s: %s" % (key, value.get(key)))
    if value.get("state") in ("STARTING", "RUNNING") and not process_alive(value.get("pid")):
        print("Process is not alive; relaunch the same bounded queue to resume.")
    log = p["state"] / ("fit_supervisor.log" if kind == "fit" else "test_supervisor.log")
    if log.is_file():
        print("----- supervisor tail -----")
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "preflight", "prepare", "start-fit", "run-fit", "fit-status",
            "verify-frozen", "prepare-test", "start-test", "run-test",
            "test-status", "verify", "summary",
        ),
    )
    parser.add_argument("--project", type=Path, default=Path("/root/autodl-tmp/BTC_DEIM_project"))
    args = parser.parse_args()
    project = args.project.resolve(strict=True)
    p = paths(project)
    if args.action == "preflight":
        preflight(project, prepare=False)
    elif args.action == "prepare":
        preflight(project, prepare=True)
    elif args.action == "start-fit":
        preflight(project, prepare=True)
        start_background(project, "run-fit", p["fit_status"], "fit_supervisor.log")
    elif args.action == "run-fit":
        run_fit(project)
    elif args.action == "fit-status":
        show_status(project, "fit")
    elif args.action == "verify-frozen":
        verify_frozen(project)
    elif args.action == "prepare-test":
        prepare_test(project)
    elif args.action == "start-test":
        prepare_test(project)
        start_background(project, "run-test", p["test_status"], "test_supervisor.log")
    elif args.action == "run-test":
        run_test(project)
    elif args.action == "test-status":
        show_status(project, "test")
    elif args.action == "verify":
        verify(project)
    elif args.action == "summary":
        summary(project)


if __name__ == "__main__":
    main()
