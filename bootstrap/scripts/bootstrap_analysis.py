#!/usr/bin/env python3
"""Build a paired image-level bootstrap analysis for the canonical CDW1630 seed0 pair."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


TARGET_NAME = "bootstrap"
REPLICATES = 5000
RNG_SEED = 20260915
CONFIDENCE = 0.95
METRICS = ("AP", "AP50", "AP75", "AR100")
SEEDS = (0, 1, 2)
METHODS = ("B0", "Ours")


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    if not path.is_file():
        fail(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def eval_path(main: Path, method: str, seed: int) -> Path:
    base = main / "results/CDW1630"
    if method == "B0":
        return base / f"B0/seed{seed}/official_test/eval.pth"
    return base / f"Ours/seed{seed}/final/eval.pth"


def paths(project: Path) -> dict[str, Path]:
    main = project / "main_host"
    qualitative = project / "qualitative_evidence"
    source = main / "source/FINAL_CODE"
    target = project / TARGET_NAME
    return {
        "project": project,
        "main": main,
        "source": source,
        "main_manifest": main / "manifests/FILE_MANIFEST.sha256.json",
        "annotation": main / "datasets/CDW1630/annotations/instances_test2017.json",
        "qualitative": qualitative,
        "b0_predictions": qualitative / "predictions/B0/seed0/predictions.json",
        "ours_predictions": qualitative / "predictions/Ours/seed0/predictions.json",
        "b0_prediction_summary": qualitative / "predictions/B0/seed0/PREDICTION_SUMMARY.json",
        "ours_prediction_summary": qualitative / "predictions/Ours/seed0/PREDICTION_SUMMARY.json",
        "qualitative_manifest": qualitative / "manifests/PACKAGE_MANIFEST.sha256.json",
        "target": target,
        "input_annotation": target / "inputs/instances_test2017.json",
        "input_b0": target / "inputs/B0_seed0_predictions.json",
        "input_ours": target / "inputs/Ours_seed0_predictions.json",
        "bindings": target / "manifests/INPUT_BINDINGS.json",
        "protocol": target / "protocol/BOOTSTRAP_PROTOCOL_LOCK.json",
        "results": target / "results/BOOTSTRAP_SUMMARY.json",
    }


def package_rows(root: Path) -> list[dict]:
    manifest = root / "manifests/PACKAGE_MANIFEST.sha256.json"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest and "__pycache__" not in path.parts and path.suffix != ".pyc":
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return rows


def rebuild_manifest(root: Path) -> None:
    write_json(root / "manifests/PACKAGE_MANIFEST.sha256.json", {
        "algorithm": "sha256", "format_version": 1, "package": root.name,
        "files": package_rows(root),
    })


def verify_manifest(root: Path) -> int:
    manifest = read_json(root / "manifests/PACKAGE_MANIFEST.sha256.json")
    expected = {row["path"]: row for row in manifest.get("files", [])}
    observed = {row["path"]: row for row in package_rows(root)}
    if set(expected) != set(observed):
        fail(
            "Bootstrap manifest path-set mismatch; missing=%s extra=%s"
            % (sorted(set(expected) - set(observed))[:4], sorted(set(observed) - set(expected))[:4])
        )
    for relative, row in expected.items():
        current = observed[relative]
        if row.get("sha256") != current["sha256"] or int(row.get("bytes", -1)) != current["bytes"]:
            fail(f"Bootstrap package manifest mismatch: {relative}")
    return len(expected)


def main_manifest_index(manifest_path: Path) -> dict[str, dict]:
    payload = read_json(manifest_path)
    return {row["path"]: row for row in payload.get("files", [])}


def verify_main_assets(p: dict[str, Path]) -> dict:
    required = [
        p["main_manifest"], p["annotation"], p["b0_predictions"], p["ours_predictions"],
        p["b0_prediction_summary"], p["ours_prediction_summary"], p["qualitative_manifest"],
        p["source"] / "tools/qccr/metrics.py",
    ]
    required.extend(eval_path(p["main"], method, seed) for method in METHODS for seed in SEEDS)
    for path in required:
        if not path.is_file():
            fail(f"Missing required bootstrap input: {path}")

    annotation = read_json(p["annotation"])
    image_ids = sorted(int(row["id"]) for row in annotation.get("images", []))
    if len(image_ids) != 237 or len(set(image_ids)) != 237:
        fail(f"Expected 237 unique CDW1630 Test images, got {len(image_ids)}")

    predictions = {}
    summaries = {}
    for method, pred_key, summary_key in (
        ("B0", "b0_predictions", "b0_prediction_summary"),
        ("Ours", "ours_predictions", "ours_prediction_summary"),
    ):
        pred = read_json(p[pred_key])
        summary = read_json(p[summary_key])
        if pred.get("method") != method or int(pred.get("detector_seed", -1)) != 0:
            fail(f"Unexpected {method} prediction identity")
        if pred.get("image_ids") != image_ids or int(pred.get("image_count", -1)) != 237:
            fail(f"{method} prediction image set does not match CDW1630 Test")
        if summary.get("formal_metric_replay") != "PASS":
            fail(f"{method} predictions were not metric-exactly replayed")
        if summary.get("predictions_sha256") != sha256(p[pred_key]):
            fail(f"{method} prediction/summary SHA mismatch")
        if summary.get("formal_eval_sha256") != sha256(eval_path(p["main"], method, 0)):
            fail(f"{method} formal-eval SHA mismatch")
        if pred.get("test_used_for_parameter_or_model_selection") is not False:
            fail(f"{method} prediction provenance claims Test selection")
        predictions[method] = pred
        summaries[method] = summary

    indexed = main_manifest_index(p["main_manifest"])
    for path in [p["annotation"], *(eval_path(p["main"], m, s) for m in METHODS for s in SEEDS)]:
        relative = path.relative_to(p["main"]).as_posix()
        row = indexed.get(relative)
        if row is None or row.get("sha256") != sha256(path):
            fail(f"Main-package manifest binding failed: {relative}")
    return {"annotation": annotation, "image_ids": image_ids, "predictions": predictions, "summaries": summaries}


def build_protocol(p: dict[str, Path]) -> dict:
    return {
        "protocol_name": "CDW1630_PAIRED_IMAGE_BOOTSTRAP_v1",
        "protocol_version": "1.0",
        "status": "FROZEN_BEFORE_BOOTSTRAP_COMPUTATION",
        "dataset": "CDW1630 official Test",
        "methods": ["B0", "Ours"],
        "canonical_detector_seed": 0,
        "resampling_unit": "image",
        "pairing": "the identical bootstrap image multiplicities are used for B0 and Ours",
        "sample_size_per_replicate": 237,
        "sampling": "ordinary nonparametric sampling with replacement from all 237 Test images",
        "replicates": REPLICATES,
        "rng": {"algorithm": "NumPy PCG64", "seed": RNG_SEED},
        "metrics": {
            "primary": "COCO AP@[0.50:0.95]",
            "secondary_descriptive": ["AP50", "AP75", "AR100"],
            "units": "Ours minus B0 in percentage points",
        },
        "interval": {
            "type": "two-sided percentile bootstrap",
            "confidence": CONFIDENCE,
            "quantiles": [0.025, 0.975],
        },
        "tail_probability": {
            "type": "two-sided empirical sign-tail probability with plus-one correction",
            "claim_scope": "primary AP only; secondary metrics are descriptive and not separate confirmatory tests",
        },
        "coco_computation": "per-image COCO matching is computed once; sampled evaluation records are repeated with replacement and COCO accumulation is rerun exactly",
        "seed_scope": {
            "bootstrap": "canonical detector seed0; conditional image-sampling uncertainty",
            "training_randomness": "reported separately using exact seed0/1/2 results; not estimated by this image bootstrap",
        },
        "selection_policy": "analysis is reporting-only after model/parameter freezing; no result may alter the model, parameter, or case selection",
        "test_used_for_model_or_parameter_selection": False,
        "gpu_required": False,
        "input_binding_policy": "copied prediction/annotation inputs plus SHA-256 bindings to formal evals and source packages",
    }


def protocol_markdown() -> str:
    return f"""# CDW1630 paired image-level Bootstrap protocol

Status: **frozen before Bootstrap computation**.

The canonical seed0 B0 and Ours predictions are compared with an ordinary
paired nonparametric Bootstrap over the 237 official-Test images. Every
replicate samples 237 images with replacement, and the same image
multiplicities are applied to both methods. Per-image COCO matching is cached,
but COCO accumulation and global detection-score ranking are recomputed for
every resample. This is not an average of ad-hoc per-image AP values.

- Replicates: `{REPLICATES}`.
- RNG: NumPy PCG64, seed `{RNG_SEED}`.
- Primary endpoint: paired delta COCO AP@[0.50:0.95].
- Descriptive secondary endpoints: AP50, AP75, and AR100.
- Interval: two-sided 95% percentile interval.
- Tail summary: empirical two-sided sign-tail probability with plus-one
  correction, interpreted confirmatorily only for the predeclared AP endpoint.

The image Bootstrap conditions on the canonical trained seed0 pair. Exact
seed0/1/2 results are retained separately to describe training-seed
variability; three seeds are not misrepresented as a large-sample Bootstrap.
The Test split is used only for final statistical reporting and never for
model, parameter, checkpoint, or qualitative-case selection.
"""


def readme_text() -> str:
    return """# Bootstrap statistical analysis reproduction package v1

This package contains the reviewer-facing paired image-level Bootstrap for the
canonical CDW1630 seed0 B0/Ours comparison. It copies only the annotation and
metric-exact prediction lists needed to rerun the statistic. The original
formal evals, qualitative prediction package, and three-seed context are bound
by SHA-256.

The Bootstrap measures Test-image sampling uncertainty conditional on the
trained seed0 pair. Training-seed variability remains a distinct three-seed
mean/standard-deviation result. This distinction is deliberate and is stated
in every summary. COCO matching and accumulation use the final source's
`faster-coco-eval` backend. No GPU, inference, fitting, or parameter selection
is used.
"""


def preflight(project: Path) -> tuple[dict[str, Path], dict]:
    p = paths(project)
    checked = verify_main_assets(p)
    print("===== CDW1630 BOOTSTRAP PREFLIGHT =====")
    print("Formal inputs : B0/Ours seed0 metric-exact Test predictions")
    print("Images        : 237 paired resampling units")
    print("Replicates    :", REPLICATES, "; RNG seed=", RNG_SEED)
    print("Endpoints     : AP primary; AP50/AP75/AR100 descriptive")
    print("Seed scope    : seed0 image uncertainty; three-seed context separate")
    print("Test purpose  : final reporting only; model/parameter selection NO")
    print("GPU/inference : NOT REQUIRED / NOT EXECUTED")
    print("Writes        : NO")
    print("CDW1630 BOOTSTRAP PREFLIGHT: PASS")
    return p, checked


def prepare(project: Path) -> None:
    p, checked = preflight(project)
    target = p["target"]
    if target.exists() and p["bindings"].is_file() and p["protocol"].is_file():
        verify_prepared(p)
        installed = target / "scripts/bootstrap_analysis.py"
        current = Path(__file__).resolve()
        if not installed.is_file() or sha256(installed) != sha256(current):
            shutil.copy2(current, installed)
            write_text(target / "README.md", readme_text())
            rebuild_manifest(target)
            print("Existing preparation retained; COCO backend script safely refreshed")
        else:
            print("Existing Bootstrap package preparation retained; writes performed: NO")
        return
    if target.exists() and any(target.iterdir()):
        fail(f"Refusing to overwrite partial target package: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for relative in ("inputs", "protocol", "results", "environment", "manifests", "scripts"):
        (target / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(p["annotation"], p["input_annotation"])
    shutil.copy2(p["b0_predictions"], p["input_b0"])
    shutil.copy2(p["ours_predictions"], p["input_ours"])
    shutil.copy2(Path(__file__).resolve(), target / "scripts/bootstrap_analysis.py")
    protocol = build_protocol(p)
    write_json(p["protocol"], protocol)
    write_text(target / "protocol/BOOTSTRAP_PROTOCOL.md", protocol_markdown())
    write_text(target / "README.md", readme_text())
    bindings = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "main_package_manifest": {
            "project_relative_path": p["main_manifest"].relative_to(project).as_posix(),
            "sha256": sha256(p["main_manifest"]),
        },
        "qualitative_package_manifest": {
            "project_relative_path": p["qualitative_manifest"].relative_to(project).as_posix(),
            "sha256": sha256(p["qualitative_manifest"]),
        },
        "copied_inputs": {
            "annotation": {"path": "inputs/instances_test2017.json", "source_sha256": sha256(p["annotation"])},
            "B0": {"path": "inputs/B0_seed0_predictions.json", "source_sha256": sha256(p["b0_predictions"])},
            "Ours": {"path": "inputs/Ours_seed0_predictions.json", "source_sha256": sha256(p["ours_predictions"])},
        },
        "formal_evals": {
            f"{method}_seed{seed}": {
                "project_relative_path": eval_path(p["main"], method, seed).relative_to(project).as_posix(),
                "sha256": sha256(eval_path(p["main"], method, seed)),
            }
            for method in METHODS for seed in SEEDS
        },
        "prediction_summaries": {
            method: {
                "project_relative_path": p[key].relative_to(project).as_posix(),
                "sha256": sha256(p[key]),
                "checkpoint_sha256": checked["summaries"][method]["checkpoint_sha256"],
                "formal_metric_replay": "PASS",
            }
            for method, key in (("B0", "b0_prediction_summary"), ("Ours", "ours_prediction_summary"))
        },
        "test_used_for_model_or_parameter_selection": False,
    }
    write_json(p["bindings"], bindings)
    write_json(target / "environment/PREPARE_ENVIRONMENT.json", {
        "python": sys.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_required": False,
        "bootstrap_executed": False,
    })
    rebuild_manifest(target)
    verify_prepared(p)
    print("===== CDW1630 BOOTSTRAP PREPARATION =====")
    print("Package       :", target)
    print("Protocol      : FROZEN BEFORE COMPUTATION")
    print("Local inputs  : annotation + B0/Ours seed0 predictions")
    print("External binds: six formal evals + qualitative/main manifests")
    print("Bootstrap     : NOT EXECUTED")
    print("CDW1630 BOOTSTRAP PREPARATION: PASS")


def verify_prepared(p: dict[str, Path]) -> None:
    protocol = read_json(p["protocol"])
    if protocol.get("status") != "FROZEN_BEFORE_BOOTSTRAP_COMPUTATION":
        fail("Bootstrap protocol is not frozen")
    if int(protocol.get("replicates", -1)) != REPLICATES or protocol["rng"].get("seed") != RNG_SEED:
        fail("Bootstrap replicate/RNG lock mismatch")
    if protocol.get("test_used_for_model_or_parameter_selection") is not False:
        fail("Bootstrap protocol claims Test selection")
    bindings = read_json(p["bindings"])
    for key, path in (("annotation", p["input_annotation"]), ("B0", p["input_b0"]), ("Ours", p["input_ours"])):
        row = bindings["copied_inputs"][key]
        if not path.is_file() or sha256(path) != row["source_sha256"]:
            fail(f"Copied Bootstrap input mismatch: {key}")
    annotation = read_json(p["input_annotation"])
    ids = sorted(int(row["id"]) for row in annotation.get("images", []))
    if len(ids) != 237:
        fail("Prepared annotation does not contain 237 images")
    for key, path in (("B0", p["input_b0"]), ("Ours", p["input_ours"])):
        payload = read_json(path)
        if payload.get("method") != key or payload.get("image_ids") != ids:
            fail(f"Prepared {key} image set mismatch")


def metric_values(eval_payload: dict, params) -> dict[str, float]:
    import numpy as np

    precision = np.asarray(eval_payload["precision"], dtype=np.float64)
    recall = np.asarray(eval_payload["recall"], dtype=np.float64)
    ious = np.asarray(params.iouThrs, dtype=np.float64)
    areas = list(params.areaRngLbl)
    max_dets = [int(value) for value in params.maxDets]
    area = areas.index("all")
    max100 = max_dets.index(100)

    def average(values) -> float:
        valid = values[values > -1]
        if not valid.size:
            return float("nan")
        return float(valid.mean())

    def ap(iou: float | None = None) -> float:
        values = precision[:, :, :, area, max100]
        if iou is not None:
            index = int(np.flatnonzero(np.isclose(ious, iou, rtol=0, atol=1e-12))[0])
            values = values[index]
        return average(values)

    return {"AP": ap(), "AP50": ap(0.50), "AP75": ap(0.75), "AR100": average(recall[:, :, area, max100])}


def make_evaluator(annotation: Path, detections: list[dict], image_ids: list[int]):
    """Use the exact faster-coco-eval backend already locked by the final source."""
    from faster_coco_eval import COCO, COCOeval_faster

    gt = COCO(str(annotation))
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(detections)
        evaluator = COCOeval_faster(
            gt, iouType="bbox", print_function=lambda *_args, **_kwargs: None,
            separate_eval=True,
        )
        evaluator.cocoDt = dt
        evaluator.params.imgIds = list(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
    return evaluator


def resampled_metrics(evaluator, sample_indices) -> dict[str, float]:
    """Repeat cached faster-coco-eval image records and rerun global accumulation."""
    import numpy as np
    from faster_coco_eval import COCOeval_faster

    original = evaluator._paramsEval
    image_count = len(original.imgIds)
    category_count = len(original.catIds) if original.useCats else 1
    area_count = len(original.areaRng)
    cached = np.asarray(evaluator._evalImgs_cpp, dtype=object)
    if cached.size != category_count * area_count * image_count:
        fail("Unexpected COCO per-image evaluation cache shape")
    selected = cached.reshape(category_count, area_count, image_count)[:, :, sample_indices].ravel().tolist()
    boot = COCOeval_faster(
        evaluator.cocoGt, iouType="bbox",
        print_function=lambda *_args, **_kwargs: None, separate_eval=True,
    )
    boot.cocoDt = evaluator.cocoDt
    boot.params = copy.deepcopy(evaluator.params)
    boot._paramsEval = copy.deepcopy(evaluator._paramsEval)
    pseudo_ids = list(range(image_count))
    boot.params.imgIds = pseudo_ids
    boot._paramsEval.imgIds = pseudo_ids
    boot._evalImgs_cpp = selected
    with contextlib.redirect_stdout(io.StringIO()):
        boot.accumulate()
    return metric_values(boot.eval, boot.params)


def percentile_summary(values, confidence: float = CONFIDENCE) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        fail("Invalid Bootstrap distribution")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(array, [alpha, 1.0 - alpha])
    n = int(array.size)
    nonpositive = int(np.count_nonzero(array <= 0.0))
    nonnegative = int(np.count_nonzero(array >= 0.0))
    two_sided = min(1.0, 2.0 * (min(nonpositive, nonnegative) + 1.0) / (n + 1.0))
    return {
        "bootstrap_mean_pp": float(array.mean()),
        "bootstrap_median_pp": float(np.median(array)),
        "ci_lower_pp": float(lower),
        "ci_upper_pp": float(upper),
        "probability_delta_gt_zero": float(np.mean(array > 0.0)),
        "empirical_two_sided_sign_tail_p_plus_one": float(two_sided),
    }


def load_exact_extractor(source: Path):
    path = source / "tools/qccr/metrics.py"
    spec = importlib.util.spec_from_file_location("cdw_final_exact_metrics", path)
    if spec is None or spec.loader is None:
        fail("Could not load final exact-metric extractor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_exact_metrics


def load_eval(path: Path):
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "eval" in payload and "precision" not in payload:
        payload = payload["eval"]
    return payload


def three_seed_context(p: dict[str, Path]) -> dict:
    extractor = load_exact_extractor(p["source"])
    rows = []
    for seed in SEEDS:
        b0 = extractor(load_eval(eval_path(p["main"], "B0", seed)))["overall"]
        ours = extractor(load_eval(eval_path(p["main"], "Ours", seed)))["overall"]
        row = {"seed": seed}
        for metric in METRICS:
            row[f"B0_{metric}"] = float(b0[metric])
            row[f"Ours_{metric}"] = float(ours[metric])
            row[f"delta_{metric}_pp"] = 100.0 * (float(ours[metric]) - float(b0[metric]))
        rows.append(row)
    aggregate = {}
    import statistics
    for metric in METRICS:
        b0_values = [row[f"B0_{metric}"] for row in rows]
        ours_values = [row[f"Ours_{metric}"] for row in rows]
        deltas = [row[f"delta_{metric}_pp"] for row in rows]
        aggregate[metric] = {
            "B0_mean": statistics.fmean(b0_values),
            "B0_sample_std": statistics.stdev(b0_values),
            "Ours_mean": statistics.fmean(ours_values),
            "Ours_sample_std": statistics.stdev(ours_values),
            "paired_delta_mean_pp": statistics.fmean(deltas),
            "paired_delta_sample_std_pp": statistics.stdev(deltas),
            "positive_seed_count": sum(value > 0 for value in deltas),
        }
    return {
        "role": "separate training-seed robustness context; not the image-bootstrap sampling distribution",
        "seeds": list(SEEDS), "per_seed": rows, "aggregate": aggregate,
    }




def run(project: Path) -> None:
    p = paths(project)
    verify_prepared(p)
    if p["results"].is_file():
        verify(project)
        print("Existing verified Bootstrap result retained; writes performed: NO")
        return
    annotation = read_json(p["input_annotation"])
    b0_payload = read_json(p["input_b0"])
    ours_payload = read_json(p["input_ours"])
    image_ids = sorted(int(row["id"]) for row in annotation["images"])
    if b0_payload["image_ids"] != image_ids or ours_payload["image_ids"] != image_ids:
        fail("Paired prediction image order mismatch")

    print("Building exact per-image COCO matching caches ...", flush=True)
    evaluators = {
        "B0": make_evaluator(p["input_annotation"], b0_payload["detections"], image_ids),
        "Ours": make_evaluator(p["input_annotation"], ours_payload["detections"], image_ids),
    }
    point = {method: metric_values(evaluators[method].eval, evaluators[method].params) for method in METHODS}
    bindings = read_json(p["bindings"])
    for method, payload in (("B0", b0_payload), ("Ours", ours_payload)):
        summary_path = p["project"] / bindings["prediction_summaries"][method]["project_relative_path"]
        formal = read_json(summary_path)["exact_metrics"]["overall"]
        for metric in METRICS:
            if not math.isclose(point[method][metric], float(formal[metric]), rel_tol=0.0, abs_tol=1e-12):
                fail(f"{method} full-sample COCO replay mismatch: {metric}")

    import numpy as np
    rng = np.random.Generator(np.random.PCG64(RNG_SEED))
    distributions = {metric: [] for metric in METRICS}
    replicate_rows = []
    for replicate in range(REPLICATES):
        sample = rng.integers(0, len(image_ids), size=len(image_ids), endpoint=False)
        b0 = resampled_metrics(evaluators["B0"], sample)
        ours = resampled_metrics(evaluators["Ours"], sample)
        row = {"replicate": replicate}
        for metric in METRICS:
            delta = 100.0 * (ours[metric] - b0[metric])
            if not math.isfinite(delta):
                fail(f"Non-finite Bootstrap value: replicate={replicate} metric={metric}")
            distributions[metric].append(delta)
            row[f"delta_{metric}_pp"] = delta
        replicate_rows.append(row)
        if (replicate + 1) % 250 == 0:
            print(f"Bootstrap progress: {replicate + 1}/{REPLICATES}", flush=True)

    results_dir = p["target"] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    replicate_path = results_dir / "BOOTSTRAP_REPLICATES.csv"
    fields = ["replicate", *[f"delta_{metric}_pp" for metric in METRICS]]
    with replicate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(replicate_rows)

    summary = {}
    for metric in METRICS:
        row = percentile_summary(distributions[metric])
        row.update({
            "B0_point": point["B0"][metric],
            "Ours_point": point["Ours"][metric],
            "point_delta_pp": 100.0 * (point["Ours"][metric] - point["B0"][metric]),
            "confirmatory_endpoint": metric == "AP",
        })
        summary[metric] = row
    context = three_seed_context(p)
    write_json(results_dir / "THREE_SEED_CONTEXT.json", context)
    report = {
        "status": "COMPLETE__PAIRED_IMAGE_BOOTSTRAP__REPORTING_ONLY",
        "protocol_sha256": sha256(p["protocol"]),
        "replicates_sha256": sha256(replicate_path),
        "dataset": "CDW1630 official Test", "image_count": len(image_ids),
        "detector_seed": 0, "replicates": REPLICATES, "rng_seed": RNG_SEED,
        "confidence": CONFIDENCE, "summary": summary,
        "interpretation_scope": "image-sampling uncertainty conditional on canonical seed0 pair",
        "training_seed_context_path": "THREE_SEED_CONTEXT.json",
        "training_seed_uncertainty_bootstrapped": False,
        "test_used_for_statistical_reporting": True,
        "test_used_for_model_or_parameter_selection": False,
        "gpu_used": False, "training_executed": False, "inference_executed": False,
    }
    write_json(p["results"], report)
    with (results_dir / "BOOTSTRAP_SUMMARY.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "metric", "B0_point", "Ours_point", "point_delta_pp", "ci_lower_pp", "ci_upper_pp",
            "bootstrap_mean_pp", "bootstrap_median_pp", "probability_delta_gt_zero",
            "empirical_two_sided_sign_tail_p_plus_one", "confirmatory_endpoint",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in METRICS:
            writer.writerow({"metric": metric, **{key: summary[metric][key] for key in fields if key != "metric"}})
    write_text(results_dir / "INTERPRETATION.md", interpretation_text(summary))
    rebuild_manifest(p["target"])
    verify(project)
    print("CDW1630 PAIRED IMAGE BOOTSTRAP: PASS")


def interpretation_text(summary: dict) -> str:
    ap = summary["AP"]
    direction = "excludes zero" if ap["ci_lower_pp"] > 0 or ap["ci_upper_pp"] < 0 else "includes zero"
    return f"""# Statistical interpretation

The canonical seed0 paired AP difference is `{ap['point_delta_pp']:+.6f}`
percentage points. Its predeclared 95% paired image-Bootstrap interval is
`[{ap['ci_lower_pp']:+.6f}, {ap['ci_upper_pp']:+.6f}]` percentage points and
{direction}.

This interval quantifies variation from the finite set of 237 Test images,
conditional on the already trained seed0 B0/Ours pair. It does not replace the
paper's three-seed mean and sample standard deviation, and it must not be
described as a Bootstrap over training runs. AP50, AP75, and AR100 are
predeclared descriptive secondary endpoints; no uncorrected family of four
confirmatory claims is made. The Test split was used for final reporting only.
"""


def summary_command(project: Path) -> None:
    p = paths(project)
    report = read_json(p["results"])
    print("===== CDW1630 BOOTSTRAP SUMMARY =====")
    for metric in METRICS:
        row = report["summary"][metric]
        suffix = "primary" if metric == "AP" else "descriptive"
        print(
            f"{metric:5s}: delta={row['point_delta_pp']:+.6f} pp; "
            f"95% CI=[{row['ci_lower_pp']:+.6f}, {row['ci_upper_pp']:+.6f}]; {suffix}"
        )
    print("Replicates     :", report["replicates"], "; paired images; seed0")
    print("Three-seed data: separate exact context (not bootstrapped)")
    print("Test selection : NO")


def verify_external_bindings(p: dict[str, Path]) -> None:
    bindings = read_json(p["bindings"])
    for group in ("formal_evals", "prediction_summaries"):
        for label, row in bindings[group].items():
            path = p["project"] / row["project_relative_path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                fail(f"External Bootstrap binding changed: {group}/{label}")
    for key in ("main_package_manifest", "qualitative_package_manifest"):
        row = bindings[key]
        path = p["project"] / row["project_relative_path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            fail(f"External Bootstrap manifest binding changed: {key}")


def verify(project: Path) -> None:
    p = paths(project)
    verify_prepared(p)
    verify_external_bindings(p)
    report = read_json(p["results"])
    if report.get("status") != "COMPLETE__PAIRED_IMAGE_BOOTSTRAP__REPORTING_ONLY":
        fail("Bootstrap result status mismatch")
    if report.get("replicates") != REPLICATES or report.get("rng_seed") != RNG_SEED:
        fail("Bootstrap result protocol mismatch")
    if report.get("test_used_for_statistical_reporting") is not True:
        fail("Bootstrap Test reporting disclosure is missing")
    if report.get("test_used_for_model_or_parameter_selection") is not False:
        fail("Bootstrap result claims Test selection")
    replicate_path = p["target"] / "results/BOOTSTRAP_REPLICATES.csv"
    if sha256(replicate_path) != report.get("replicates_sha256"):
        fail("Bootstrap replicate-table SHA mismatch")
    with replicate_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != REPLICATES:
        fail(f"Expected {REPLICATES} Bootstrap rows, got {len(rows)}")
    for metric in METRICS:
        row = report["summary"].get(metric)
        if not row or not all(math.isfinite(float(row[key])) for key in (
            "point_delta_pp", "ci_lower_pp", "ci_upper_pp", "bootstrap_mean_pp"
        )):
            fail(f"Invalid Bootstrap summary metric: {metric}")
        if float(row["ci_lower_pp"]) > float(row["ci_upper_pp"]):
            fail(f"Reversed Bootstrap interval: {metric}")
        if bool(row.get("confirmatory_endpoint")) != (metric == "AP"):
            fail(f"Confirmatory endpoint marker mismatch: {metric}")
    context = read_json(p["target"] / "results/THREE_SEED_CONTEXT.json")
    if context.get("seeds") != [0, 1, 2]:
        fail("Three-seed context is incomplete")
    count = verify_manifest(p["target"])
    print("===== CDW1630 BOOTSTRAP VERIFICATION =====")
    print("Paired images       : PASS (237/237)")
    print("Replicates/RNG      : PASS (%d / %d)" % (REPLICATES, RNG_SEED))
    print("Primary AP endpoint : PASS / predeclared")
    print("Secondary endpoints : AP50/AP75/AR100 descriptive")
    print("Three-seed context  : PASS / kept separate")
    print("External SHA binds  : PASS")
    print("GPU/train/inference : NOT USED / NOT EXECUTED / NOT EXECUTED")
    print("Test model/param use: NO")
    print("Package manifest    : PASS (%d files)" % count)
    print("CDW1630 BOOTSTRAP VERIFICATION: PASS")


def status(project: Path) -> None:
    p = paths(project)
    print("===== CDW1630 BOOTSTRAP STATUS =====")
    print("Prepared :", p["protocol"].is_file() and p["bindings"].is_file())
    print("Complete :", p["results"].is_file())
    print("Package  :", p["target"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "prepare", "run", "summary", "verify", "status"))
    parser.add_argument("--project", type=Path, default=Path(os.environ.get("FINAL_PAPER_RELEASE_ROOT", str(Path(__file__).resolve().parents[2]))))
    args = parser.parse_args()
    project = args.project.resolve()
    actions = {
        "preflight": lambda: preflight(project), "prepare": lambda: prepare(project),
        "run": lambda: run(project), "summary": lambda: summary_command(project),
        "verify": lambda: verify(project), "status": lambda: status(project),
    }
    actions[args.action]()


if __name__ == "__main__":
    main()
