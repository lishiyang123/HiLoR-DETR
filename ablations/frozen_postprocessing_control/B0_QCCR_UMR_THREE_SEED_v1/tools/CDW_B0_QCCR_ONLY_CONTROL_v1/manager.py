#!/usr/bin/env python3
"""Evaluate the missing B0 + QCCR row without retraining or refitting."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
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
    upstream_state_root,
    yaml_path,
)


PROTOCOL_PATH = package_root() / "protocol/B0_QCCR_ONLY_PROTOCOL.json"
SEEDS = (0, 1, 2)
ROWS = ("B0", "B0_QCCR", "B0_QCCR_UMR")
METRICS = (
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
    "AP80", "AP85", "AP90", "AP95",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def locate_main(project):
    candidates = (
        project / "HiLoR-DETR/main_host",
        project / "main_host",
    )
    for candidate in candidates:
        if (candidate / "checkpoints/CDW1630/B0").is_dir():
            return candidate.resolve()
    raise RuntimeError("Unable to locate main_host")


def paths(project):
    state = state_root(project)
    upstream = upstream_state_root(project)
    return {
        "project": project,
        "main": locate_main(project),
        "state": state,
        "upstream": upstream,
        "fit_lock": upstream / "FIT_INPUT_LOCK.json",
        "fit_status": upstream / "FIT_STATUS.json",
        "test_lock": upstream / "TEST_INPUT_LOCK.json",
        "upstream_test_status": upstream / "TEST_STATUS.json",
        "input_lock": state / "INPUT_LOCK.json",
        "status": state / "TEST_STATUS.json",
        "tables": state / "paper_tables",
    }


def file_record(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size > 0, "Missing file: %s" % path)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_file_record(record):
    path = Path(record["path"])
    require(path.is_file(), "Bound file missing: %s" % path)
    if "size" in record:
        require(path.stat().st_size == int(record["size"]), "Bound size changed: %s" % path)
    require(sha256(path) == record["sha256"], "Bound SHA changed: %s" % path)
    return path


def upstream_test_record_path(upstream, seed):
    return upstream / ("test/seed%d/TEST_RECORD.json" % seed)


def frozen_binding_path(upstream, seed):
    return upstream / ("fit/seed%d/FROZEN_BINDING.json" % seed)


def result_record_path(project, seed):
    return state_root(project) / ("test/seed%d/TEST_RECORD.json" % seed)


def qccr_config_text(base, image_dir, annotation, output, qccr):
    parameters = read_json(qccr)["parameters"]
    required = ("beta0", "beta_positive", "beta_negative", "duplicate_penalty")
    require(all(key in parameters for key in required), "Incomplete frozen QCCR parameters")
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
            "  umr_enabled: False",
            "",
        ]
    )


def inspect_upstream(project):
    p = paths(project)
    require(p["fit_lock"].is_file() and p["test_lock"].is_file(), "Upstream locks are incomplete")
    fit_status = read_json(p["fit_status"])
    test_status = read_json(p["upstream_test_status"])
    require(fit_status.get("state") == "COMPLETE" and fit_status.get("completed") == [0, 1, 2], "Upstream fit/freeze is incomplete")
    require(test_status.get("state") == "COMPLETE" and test_status.get("completed") == [0, 1, 2], "Upstream B0+QCCR+UMR Test is incomplete")
    fit_lock = read_json(p["fit_lock"])
    test_lock = read_json(p["test_lock"])
    require(fit_lock.get("test_accessed") is False, "Upstream fitting accessed Test")
    require(test_lock.get("test_used_for_parameter_selection") is False, "Upstream Test role changed")
    source = Path(fit_lock["source"]["path"])
    require((source / "train.py").is_file(), "Runnable source missing")
    require(sha256(Path(fit_lock["source"]["manifest"])) == fit_lock["source"]["manifest_sha256"], "Source manifest changed")
    require(Path(test_lock["test_images"]["path"]).is_dir(), "Test image directory missing")
    require(int(test_lock["test_images"]["file_count"]) == 237, "Unexpected Test image count")
    require(sha256(Path(test_lock["test_annotation"]["path"])) == test_lock["test_annotation"]["sha256"], "Test annotation changed")
    records = {}
    for seed in SEEDS:
        checkpoint = Path(fit_lock["seeds"][str(seed)]["checkpoint"])
        require(checkpoint.is_file() and sha256(checkpoint) == fit_lock["seeds"][str(seed)]["checkpoint_sha256"], "B0 checkpoint changed: seed%d" % seed)
        frozen_path = frozen_binding_path(p["upstream"], seed)
        frozen = read_json(frozen_path)
        require(frozen.get("seed") == seed and frozen.get("test_accessed") is False, "Frozen binding changed: seed%d" % seed)
        qccr = verify_file_record(frozen["qccr_artifact"])
        require(read_json(qccr).get("source_checkpoint_sha256") == sha256(checkpoint), "QCCR/checkpoint mismatch: seed%d" % seed)
        upstream_record_path = upstream_test_record_path(p["upstream"], seed)
        upstream_record = read_json(upstream_record_path)
        require(upstream_record.get("seed") == seed and upstream_record.get("test_used_for_parameter_selection") is False, "Upstream Test record changed: seed%d" % seed)
        for key in ("baseline_eval", "postprocessed_eval", "frozen_binding"):
            verify_file_record(upstream_record[key])
        base_config = p["upstream"] / ("fit/seed%d/configs/b0_base.yml" % seed)
        require(base_config.is_file(), "Upstream B0 base config missing: seed%d" % seed)
        records[seed] = {
            "checkpoint": checkpoint,
            "frozen_binding": frozen_path,
            "qccr": qccr,
            "base_config": base_config,
            "upstream_test_record": upstream_record_path,
            "upstream_metrics": upstream_record["metrics"],
        }
    return {
        "paths": p,
        "fit_lock": fit_lock,
        "test_lock": test_lock,
        "source": source,
        "records": records,
    }


def make_input_lock(context):
    p = context["paths"]
    value = {
        "schema": "cdw1630-b0-qccr-only-input-lock-v1",
        "protocol": file_record(PROTOCOL_PATH),
        "source": {
            "path": str(context["source"]),
            "manifest": file_record(Path(context["fit_lock"]["source"]["manifest"])),
        },
        "test_images": context["test_lock"]["test_images"],
        "test_annotation": context["test_lock"]["test_annotation"],
        "seeds": {},
        "detector_training": False,
        "postprocessor_fitting": False,
        "test_used_for_parameter_selection": False,
        "created_utc": utc_now(),
    }
    for seed in SEEDS:
        item = context["records"][seed]
        value["seeds"][str(seed)] = {
            "checkpoint": file_record(item["checkpoint"]),
            "qccr_artifact": file_record(item["qccr"]),
            "frozen_binding": file_record(item["frozen_binding"]),
            "upstream_test_record": file_record(item["upstream_test_record"]),
            "base_config": file_record(item["base_config"]),
        }
    return value


def compare_lock(existing, candidate):
    left, right = dict(existing), dict(candidate)
    left.pop("created_utc", None)
    right.pop("created_utc", None)
    require(left == right, "Existing QCCR-only input lock differs")


def prepare_test(project):
    context = inspect_upstream(project)
    p = context["paths"]
    p["state"].mkdir(parents=True, exist_ok=True)
    lock = make_input_lock(context)
    if p["input_lock"].is_file():
        compare_lock(read_json(p["input_lock"]), lock)
        lock = read_json(p["input_lock"])
    else:
        atomic_json(p["input_lock"], lock)
    for seed in SEEDS:
        root = p["state"] / ("test/seed%d" % seed)
        config = root / "config/b0_qccr_test.yml"
        atomic_text(
            config,
            qccr_config_text(
                Path(lock["seeds"][str(seed)]["base_config"]["path"]),
                Path(lock["test_images"]["path"]),
                Path(lock["test_annotation"]["path"]),
                root / "output",
                Path(lock["seeds"][str(seed)]["qccr_artifact"]["path"]),
            ),
        )
    print("===== B0 + QCCR FORMAL TEST PREPARATION =====")
    print("Frozen B0/QCCR assets : 3/3")
    print("New row               : B0 + QCCR")
    print("UMR                   : OFF")
    print("Detector train/refit  : NO / NO")
    print("Test role             : reporting only")
    print("PREPARATION: PASS")
    return lock


def preflight(project):
    inspect_upstream(project)
    print("===== B0 + QCCR THREE-SEED CONTROL PREFLIGHT =====")
    print("Detector seeds : 0 / 1 / 2 existing B0 checkpoints")
    print("QCCR artifacts : checkpoint-specific / already frozen")
    print("UMR            : OFF")
    print("Detector train : NO")
    print("QCCR refit     : NO")
    print("Test role      : reporting only")
    print("PREFLIGHT: PASS")


def reset_bounded(path, parent):
    if not path.exists():
        return
    path = path.resolve()
    parent = parent.resolve()
    require(path.parent == parent, "Unsafe cleanup target: %s" % path)
    shutil.rmtree(str(path))


def run_logged(command, cwd, log):
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\nCOMMAND " + " ".join(str(value) for value in command) + "\n")
        handle.flush()
        result = subprocess.run(
            [str(value) for value in command],
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
        )
    require(result.returncode == 0, "Command exited %d; inspect %s" % (result.returncode, log))


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


def close_metrics(left, right):
    for key in METRICS:
        if left[key] is None or right[key] is None:
            require(left[key] is None and right[key] is None, "Metric differs: %s" % key)
        else:
            require(abs(float(left[key]) - float(right[key])) <= 1e-10, "Metric differs: %s" % key)


def run_seed(project, seed):
    p = paths(project)
    record_path = result_record_path(project, seed)
    if record_path.is_file():
        return verify_seed(project, seed)
    lock = read_json(p["input_lock"])
    root = p["state"] / ("test/seed%d" % seed)
    output = root / "output"
    reset_bounded(output, root)
    config = root / "config/b0_qccr_test.yml"
    checkpoint = Path(lock["seeds"][str(seed)]["checkpoint"]["path"])
    source = Path(lock["source"]["path"])
    command = [
        sys.executable, "-u", "-B", str(source / "train.py"),
        "-c", str(config), "--seed", str(seed),
        "--output-dir", str(output), "-r", str(checkpoint), "--test-only",
    ]
    run_logged(command, source, root / "console.log")
    eval_path = output / "eval.pth"
    upstream_record_path = Path(lock["seeds"][str(seed)]["upstream_test_record"]["path"])
    upstream = read_json(upstream_record_path)
    value = {
        "schema": "cdw1630-b0-qccr-only-formal-test-record-v1",
        "seed": seed,
        "checkpoint": lock["seeds"][str(seed)]["checkpoint"],
        "qccr_artifact": lock["seeds"][str(seed)]["qccr_artifact"],
        "config": file_record(config),
        "qccr_eval": file_record(eval_path),
        "upstream_test_record": lock["seeds"][str(seed)]["upstream_test_record"],
        "metrics": {
            "B0": upstream["metrics"]["B0"],
            "B0_QCCR": extract_metrics(eval_path),
            "B0_QCCR_UMR": upstream["metrics"]["B0_QCCR_UMR"],
        },
        "detector_retrained": False,
        "qccr_refitted": False,
        "umr_enabled": False,
        "test_role": "final reporting only",
        "test_used_for_parameter_selection": False,
        "completed_utc": utc_now(),
    }
    atomic_json(record_path, value)
    return verify_seed(project, seed)


def verify_seed(project, seed):
    record = read_json(result_record_path(project, seed))
    require(record.get("seed") == seed, "Seed identity changed")
    require(record.get("detector_retrained") is False and record.get("qccr_refitted") is False, "Training/refit scope changed")
    require(record.get("umr_enabled") is False, "UMR unexpectedly enabled")
    require(record.get("test_used_for_parameter_selection") is False, "Test selection changed")
    for key in ("checkpoint", "qccr_artifact", "config", "qccr_eval", "upstream_test_record"):
        verify_file_record(record[key])
    config = Path(record["config"]["path"]).read_text(encoding="utf-8")
    require("qccr_enabled: True" in config and "umr_enabled: False" in config, "QCCR-only config changed")
    close_metrics(extract_metrics(Path(record["qccr_eval"]["path"])), record["metrics"]["B0_QCCR"])
    upstream = read_json(Path(record["upstream_test_record"]["path"]))
    close_metrics(upstream["metrics"]["B0"], record["metrics"]["B0"])
    close_metrics(upstream["metrics"]["B0_QCCR_UMR"], record["metrics"]["B0_QCCR_UMR"])
    return record


def update_status(path, **updates):
    value = read_json(path) if path.is_file() else {}
    value.update(updates)
    value["updated_utc"] = utc_now()
    atomic_json(path, value)


def run_test(project):
    p = paths(project)
    require(p["input_lock"].is_file(), "Run prepare-test first")
    completed = []
    try:
        for seed in SEEDS:
            update_status(p["status"], state="RUNNING", pid=os.getpid(), current="seed%d" % seed, completed=completed, pending=[value for value in SEEDS if value not in completed], error=None)
            record = run_seed(project, seed)
            completed.append(seed)
            print("seed%d B0 + QCCR Test AP=%.9f" % (seed, record["metrics"]["B0_QCCR"]["AP"]), flush=True)
        update_status(p["status"], state="COMPLETE", pid=os.getpid(), current=None, completed=list(SEEDS), pending=[], error=None)
        build_tables(project)
        print("B0 + QCCR THREE-SEED FORMAL TEST: COMPLETE")
    except BaseException as exc:
        update_status(p["status"], state="FAILED", pid=os.getpid(), current="seed%d" % (SEEDS[len(completed)] if len(completed) < len(SEEDS) else SEEDS[-1]), completed=completed, pending=[value for value in SEEDS if value not in completed], error="%s: %s" % (type(exc).__name__, exc))
        raise


def mean_std(values):
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def build_tables(project):
    p = paths(project)
    records = [verify_seed(project, seed) for seed in SEEDS]
    p["tables"].mkdir(parents=True, exist_ok=True)
    per_seed = []
    for record in records:
        baseline = record["metrics"]["B0"]
        for name in ROWS:
            metrics = record["metrics"][name]
            row = {"seed": record["seed"], "row": name}
            row.update(metrics)
            row.update({"delta_%s_vs_B0_pp" % key: 100.0 * (metrics[key] - baseline[key]) for key in METRICS})
            per_seed.append(row)
    columns = ["seed", "row"] + list(METRICS) + ["delta_%s_vs_B0_pp" % key for key in METRICS]
    with (p["tables"] / "B0_QCCR_BRANCH_PER_SEED.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(per_seed)
    summary_rows = []
    for name in ROWS:
        selected = [row for row in per_seed if row["row"] == name]
        item = {"row": name, "metrics": {}, "delta_vs_B0_pp": {}}
        for key in METRICS:
            item["metrics"][key] = mean_std([float(row[key]) for row in selected])
            item["delta_vs_B0_pp"][key] = mean_std([float(row["delta_%s_vs_B0_pp" % key]) for row in selected])
        summary_rows.append(item)
    summary = {
        "schema": "cdw1630-b0-qccr-branch-three-seed-summary-v1",
        "dataset": "CDW1630 official Test",
        "detector_seeds": list(SEEDS),
        "rows": summary_rows,
        "detector_retrained": False,
        "postprocessor_refitted": False,
        "test_used_for_parameter_selection": False,
    }
    atomic_json(p["tables"] / "B0_QCCR_BRANCH_MEAN_STD.json", summary)
    lines = [
        "# CDW1630 B0 frozen-postprocessing branch / three detector seeds",
        "",
        "| Row | AP | Delta AP vs B0 | AP50 | AP75 | AR100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        metric, delta = row["metrics"], row["delta_vs_B0_pp"]
        lines.append(
            "| %s | %.3f ± %.3f | %+.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f |"
            % (
                row["row"],
                100.0 * metric["AP"]["mean"], 100.0 * metric["AP"]["sample_std"],
                delta["AP"]["mean"], delta["AP"]["sample_std"],
                100.0 * metric["AP50"]["mean"], 100.0 * metric["AP50"]["sample_std"],
                100.0 * metric["AP75"]["mean"], 100.0 * metric["AP75"]["sample_std"],
                100.0 * metric["AR100"]["mean"], 100.0 * metric["AR100"]["sample_std"],
            )
        )
    lines.extend(["", "B0 and B0+QCCR+UMR are reused from the bound upstream experiment. Only B0+QCCR is newly evaluated.", ""])
    atomic_text(p["tables"] / "B0_QCCR_BRANCH_MAIN.md", "\n".join(lines))
    return summary


def write_result_manifest(project):
    p = paths(project)
    files = [PROTOCOL_PATH, p["input_lock"]]
    files.extend(result_record_path(project, seed) for seed in SEEDS)
    files.extend(Path(read_json(result_record_path(project, seed))["qccr_eval"]["path"]) for seed in SEEDS)
    files.extend(sorted(item for item in p["tables"].glob("*") if item.is_file()))
    records = []
    for path in sorted(set(item.resolve() for item in files), key=str):
        records.append(file_record(path))
    atomic_json(p["state"] / "RESULT_MANIFEST.sha256.json", {"schema": 1, "files": records})


def verify(project):
    p = paths(project)
    inspect_upstream(project)
    require(p["input_lock"].is_file(), "Input lock is missing")
    for seed in SEEDS:
        verify_seed(project, seed)
    status = read_json(p["status"])
    require(status.get("state") == "COMPLETE" and status.get("completed") == [0, 1, 2], "QCCR-only Test queue is incomplete")
    build_tables(project)
    write_result_manifest(project)
    print("===== B0 + QCCR THREE-SEED CONTROL VERIFICATION =====")
    print("Frozen B0 checkpoints : 3/3")
    print("Frozen QCCR artifacts : 3/3")
    print("New QCCR-only evals   : 3/3")
    print("Rows                  : B0 / B0 + QCCR / B0 + QCCR + UMR")
    print("Detector train/refit  : NO / NO")
    print("Test selection        : NO")
    print("Tables/manifest       : PASS")
    print("VERIFICATION: PASS")


def summary(project):
    verify(project)
    value = read_json(state_root(project) / "paper_tables/B0_QCCR_BRANCH_MEAN_STD.json")
    print("===== CDW1630 B0 FROZEN-POSTPROCESSING BRANCH / THREE-SEED TEST =====")
    print("%-20s %18s %18s %18s %18s %18s" % ("row", "AP", "dAP", "AP50", "AP75", "AR100"))
    for row in value["rows"]:
        metrics, deltas = row["metrics"], row["delta_vs_B0_pp"]
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


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def gpu_idle():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    require(not result.stdout.strip(), "GPU compute process already exists:\n%s" % result.stdout)


def start_background(project):
    p = paths(project)
    if p["status"].is_file():
        old = read_json(p["status"])
        require(not (old.get("state") in ("STARTING", "RUNNING") and process_alive(old.get("pid"))), "Queue is already running: PID %s" % old.get("pid"))
    gpu_idle()
    p["state"].mkdir(parents=True, exist_ok=True)
    log = p["state"] / "test_supervisor.log"
    stream = log.open("a", encoding="utf-8", newline="\n")
    process = subprocess.Popen(
        [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "run-test", "--project", str(project)],
        cwd=str(package_root()), stdout=stream, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True, text=True,
    )
    stream.close()
    update_status(p["status"], state="STARTING", pid=process.pid, current=None, completed=[], pending=list(SEEDS), error=None)
    print("Launched run-test PID=%d" % process.pid)


def show_status(project):
    p = paths(project)
    print("===== B0 + QCCR TEST STATUS =====")
    if not p["status"].is_file():
        print("state: NOT_STARTED")
        return
    value = read_json(p["status"])
    for key in ("state", "pid", "current", "completed", "pending", "error"):
        print("%s: %s" % (key, value.get(key)))
    if value.get("state") in ("STARTING", "RUNNING") and not process_alive(value.get("pid")):
        print("Process is not alive; relaunch start-test to resume.")
    log = p["state"] / "test_supervisor.log"
    if log.is_file():
        print("----- supervisor tail -----")
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "prepare-test", "start-test", "run-test", "test-status", "verify", "summary"))
    parser.add_argument("--project", type=Path, default=Path("/root/autodl-tmp/BTC_DEIM_project"))
    args = parser.parse_args()
    project = args.project.resolve(strict=True)
    if args.action == "preflight":
        preflight(project)
    elif args.action == "prepare-test":
        prepare_test(project)
    elif args.action == "start-test":
        prepare_test(project)
        start_background(project)
    elif args.action == "run-test":
        run_test(project)
    elif args.action == "test-status":
        show_status(project)
    elif args.action == "verify":
        verify(project)
    else:
        summary(project)


if __name__ == "__main__":
    main()
