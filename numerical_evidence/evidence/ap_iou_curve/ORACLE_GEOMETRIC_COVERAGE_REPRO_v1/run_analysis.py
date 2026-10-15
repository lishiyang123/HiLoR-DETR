#!/usr/bin/env python3
"""Replay saved full-query geometry on CPU, or capture into a new workspace."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from oracle_geometry import best_iou, cxcywh_to_pixel_xyxy, summarize

SCRIPT_ROOT = Path(__file__).resolve().parent
ROOT = SCRIPT_ROOT
ANNOTATION_SHA = "8130fa5706cd260a49df43d73074b2f3a55a5d2e8525c1f1b7489005a0524e05"
METRICS = ("mean_best_iou", "median_best_iou", "oracle_coverage_iou_ge_075")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def relocated_bindings(bindings, release):
    """Resolve package-internal inputs after copying the release, preserving original SHA receipts."""
    old_root = bindings["release_root"]
    new_root = str(release)
    if old_root == new_root:
        return bindings

    def relocate(value):
        if isinstance(value, str) and value.startswith(old_root + "/"):
            return new_root + value[len(old_root):]
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, dict):
            return {relocate(key): relocate(item) for key, item in value.items()}
        return value

    result = relocate(bindings)
    result["release_root"] = new_root
    return result


def package_manifest():
    files = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(ROOT.rglob("*"))
             if p.is_file() and p.name not in ("MANIFEST.sha256.json", "RUN_LOG.txt")
             and "__pycache__" not in p.parts}
    complete = all((ROOT / f"captures/{m}/seed{s}/CAPTURE_SUMMARY.json").is_file()
                   for m in ("DEIM", "GCCS") for s in (0, 1, 2))
    payload = {"status": "CAPTURES_COMPLETE" if complete else "IN_PROGRESS", "files": files,
               "log_excluded": "RUN_LOG.txt is an appendable execution log, not a frozen result artifact"}
    (ROOT / "MANIFEST.sha256.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def source_hashes(source):
    return {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".yml", ".json")}


def check_bindings(bindings, check_source=True):
    for filename, digest in bindings["protected_inputs"].items():
        if sha256(filename) != digest:
            raise RuntimeError(f"Protected input changed: {filename}")
    if check_source and source_hashes(Path(bindings["source_root"])) != bindings["source_hashes"]:
        raise RuntimeError("Detector source changed after preflight")


def check_replay_bindings(bindings):
    """Bind the inputs actually read by offline geometry, not current model code.

    The original capture bindings and receipts remain capture provenance. No
    detector or CCVR head is executed while replaying the stored box arrays.
    """
    files = read_json(ROOT / "MANIFEST.sha256.json")["files"]

    def checked(relative):
        path = ROOT / relative
        if files.get(relative) != sha256(path):
            raise RuntimeError(f"Replay artifact SHA mismatch: {relative}")
        return path

    checked("bindings.json")
    protocol = read_json(checked("protocol.json"))
    annotation = Path(bindings["annotation"])
    if (bindings["protected_inputs"].get(str(annotation)) != ANNOTATION_SHA
            or sha256(annotation) != ANNOTATION_SHA):
        raise RuntimeError("Replay annotation SHA mismatch")
    ann = read_json(annotation)
    if (len(ann["images"]) != protocol["image_count"]
            or len(ann["annotations"]) != protocol["gt_count"]
            or any(a.get("iscrowd", 0) for a in ann["annotations"])):
        raise RuntimeError("Replay image/GT scope mismatch")
    images = {int(i["id"]): i for i in ann["images"]}
    if len(images) != protocol["image_count"]:
        raise RuntimeError("Duplicate annotation image ID")
    runs = {(r["model"], r["seed"]): r for r in bindings["runs"]}
    expected_runs = {(m, s) for m in ("DEIM", "GCCS") for s in (0, 1, 2)}
    if set(runs) != expected_runs or len(bindings["runs"]) != 6:
        raise RuntimeError("Replay detector/seed scope mismatch")
    for (name, seed), run in runs.items():
        if (bindings["protected_inputs"].get(run["checkpoint"]) != run["checkpoint_sha256"]
                or sha256(run["checkpoint"]) != run["checkpoint_sha256"]):
            raise RuntimeError(f"Replay checkpoint binding mismatch: {name} seed {seed}")
        folder = f"captures/{name}/seed{seed}"
        receipt = read_json(checked(folder + "/CAPTURE_SUMMARY.json"))
        config = "cdw_b0_test.yml" if name == "DEIM" else "cdw_native_test.yml"
        expected = {"model": name, "seed": seed,
                    "checkpoint_sha256": run["checkpoint_sha256"],
                    "config_sha256": bindings["config_hashes"][config],
                    "weight_source": "ema.module", "strict_load": "PASS",
                    "images": protocol["image_count"], "queries_per_image": 300,
                    "precision": "FP32", "batch_size": 8,
                    "iou_precision": "float64", "postprocessing_used_for_oracle": False}
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Replay capture receipt mismatch: {name} seed {seed}")
        capture_path = checked(folder + "/full_query_boxes.npz")
        if sha256(capture_path) != receipt["capture_sha256"]:
            raise RuntimeError("Capture SHA mismatch")
        with np.load(capture_path, allow_pickle=False) as payload:
            ids = payload["image_ids"]
            normalized = payload["pred_boxes_cxcywh_normalized"]
            pixels = payload["pred_boxes_xyxy_pixels"]
            if (ids.ndim != 1 or len(set(ids.tolist())) != len(ids)
                    or set(ids.tolist()) != set(images)
                    or normalized.shape != (len(ids), 300, 4)
                    or pixels.shape != normalized.shape):
                raise RuntimeError("Incomplete or truncated full-query pool")
            if not np.isfinite(normalized).all() or not np.isfinite(pixels).all():
                raise RuntimeError("Nonfinite prediction in full-query pool")
            for image_id, raw, boxes in zip(ids, normalized, pixels):
                image = images[int(image_id)]
                converted = cxcywh_to_pixel_xyxy(raw, image["width"], image["height"])
                if not np.array_equal(converted, boxes):
                    raise RuntimeError("Saved normalized/pixel box coordinates disagree")


def preflight(release):
    binding_path = ROOT / "bindings.json"
    if binding_path.exists():
        bindings = read_json(binding_path)
        bindings = relocated_bindings(bindings, release)
        check_bindings(bindings)
        print("Existing preflight bindings verified", flush=True)
        return bindings
    formal = release / "main_host"
    source = formal / "source/FINAL_CODE"
    manifest = {entry["path"]: entry["sha256"] for entry in
                read_json(formal / "manifests/FILE_MANIFEST.sha256.json")["files"]}
    annotation = formal / "datasets/CDW1630/annotations/instances_test2017.json"
    if sha256(annotation) != ANNOTATION_SHA:
        raise RuntimeError("Test annotation SHA mismatch")
    ann = read_json(annotation)
    protocol = read_json(ROOT / "protocol.json")
    if len(ann["images"]) != protocol["image_count"] or len(ann["annotations"]) != protocol["gt_count"]:
        raise RuntimeError("Unexpected image/GT count")
    if any(a.get("iscrowd", 0) != 0 for a in ann["annotations"]):
        raise RuntimeError("Unexpected crowd annotation; do not silently exclude GTs")
    protected = {str(annotation): ANNOTATION_SHA}
    for image in ann["images"]:
        p = formal / "datasets/CDW1630/test2017" / image["file_name"]
        expected = manifest.get(p.relative_to(formal).as_posix())
        actual = sha256(p)
        if expected is None or actual != expected:
            raise RuntimeError(f"Test image binding mismatch: {p.name}")
        protected[str(p)] = actual
    runs = []
    for seed in protocol["seeds"]:
        for name, method in (("DEIM", "B0"), ("GCCS", "Ours")):
            stage = "best_stg1.pth" if name == "DEIM" and seed == 0 else "best_stg2.pth"
            checkpoint = formal / f"checkpoints/CDW1630/{method}/seed{seed}/{stage}"
            relative = checkpoint.relative_to(formal).as_posix()
            digest = sha256(checkpoint)
            if digest != manifest.get(relative):
                raise RuntimeError(f"Checkpoint binding mismatch: {relative}")
            protected[str(checkpoint)] = digest
            run = {"model": name, "seed": seed, "checkpoint": str(checkpoint), "checkpoint_sha256": digest}
            if name == "GCCS":
                frozen_path = formal / f"training_records/CDW1630/Ours/seed{seed}/calibration_evidence/FROZEN_INFERENCE.json"
                frozen = read_json(frozen_path)
                if frozen["checkpoint"]["sha256"] != digest or frozen["seed"] != seed:
                    raise RuntimeError("CCVR frozen inference belongs to another checkpoint")
                # Seed 1/2 freeze receipts retain their original historical paths/schema.
                # Resolve installed assets by canonical package paths, and verify the old SHA binding.
                calibration = formal / f"checkpoints/CDW1630/Ours/seed{seed}/calibration"
                qccr_path = calibration / "qccr_params.json"
                head_path = calibration / "umr_mlp_head_seed0_pw4.pt"
                meta_path = calibration / "umr_mlp_head_seed0_pw4.metadata.json"
                head_sha = (frozen["umr"]["checkpoint"]["sha256"] if "checkpoint" in frozen["umr"]
                            else frozen["umr"]["weight_sha256"])
                for p, expected in ((qccr_path, frozen["qccr"]["sha256"]), (head_path, head_sha),
                                    (meta_path, manifest.get(meta_path.relative_to(formal).as_posix()))):
                    if sha256(p) != expected:
                        raise RuntimeError(f"CCVR artifact mismatch: {p}")
                    protected[str(p)] = expected
                protected[str(frozen_path)] = sha256(frozen_path)
                run["frozen_inference"] = str(frozen_path)
                run["ccvr"] = {"qccr_parameters": frozen["qccr"]["parameters"],
                    "head_path": str(head_path), "head_sha256": head_sha,
                    **{k: frozen["umr"][k] for k in ("beta", "tau", "topk")}}
            runs.append(run)
    configs = ROOT / "configs"
    if configs.exists():
        raise RuntimeError("Config directory already exists without completed bindings; inspect before retry")
    subprocess.run([sys.executable, "-B", str(source / "tools/reproduction/materialize_configs.py"),
                    "--data-root", str(formal / "datasets/CDW1630"), "--output-root", str(ROOT / "unused_formal_outputs"),
                    "--destination", str(configs)], check=True)
    bindings = {"release_root": str(release), "formal_root": str(formal), "source_root": str(source),
                "annotation": str(annotation), "runs": runs, "protected_inputs": protected,
                "source_hashes": source_hashes(source), "protocol_sha256": sha256(ROOT / "protocol.json"),
                "config_hashes": {p.name: sha256(p) for p in configs.iterdir() if p.is_file()},
                "existing_gccs_seed0_cache": {"path": str(release.parent / "ranking_comparison/state/captures/gccs_seed0_test_capture.pt"),
                    "reused": False, "reason": "six runs recaptured with a common FP32/EMA/Test preprocessing path"}}
    write_json_new(binding_path, bindings)
    print("Preflight PASS: 237 images, 1071 GT, six formal checkpoints", flush=True)
    return bindings


def capture(bindings, name, seed):
    import torch
    import torchvision

    run = next(r for r in bindings["runs"] if r["model"] == name and r["seed"] == seed)
    destination = ROOT / f"captures/{name}/seed{seed}"
    summary_path = destination / "CAPTURE_SUMMARY.json"
    if summary_path.exists():
        existing = read_json(summary_path)
        if existing["checkpoint_sha256"] != run["checkpoint_sha256"] or sha256(destination / "full_query_boxes.npz") != existing["capture_sha256"]:
            raise RuntimeError("Existing capture binding mismatch")
        print(f"Reuse verified completed capture: {name} seed {seed}", flush=True)
        return
    if destination.exists():
        raise RuntimeError(f"Incomplete capture directory already exists: {destination}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to capture the detector")
    source = Path(bindings["source_root"])
    sys.path.insert(0, str(source))
    from engine.core import YAMLConfig
    from engine.deim.postprocessor import PostProcessor
    from engine.misc import dist_utils

    dist_utils.setup_seed(seed)
    config_name = "cdw_b0_test.yml" if name == "DEIM" else "cdw_native_test.yml"
    config = ROOT / "configs" / config_name
    for filename, digest in bindings["config_hashes"].items():
        if sha256(ROOT / "configs" / filename) != digest:
            raise RuntimeError("Materialized config changed")
    cfg = YAMLConfig(str(config), use_amp=False)
    model = cfg.model
    checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
    state = checkpoint.get("ema", {}).get("module")
    if not isinstance(state, dict):
        raise RuntimeError("Formal EMA state is missing; do not fall back silently")
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    model = model.to("cuda:0").eval()
    loader = cfg.val_dataloader
    if bool(cfg.yaml_cfg.get("use_amp")) or loader.batch_size != 8:
        raise RuntimeError("FP32/batch 8 protocol mismatch")
    if cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] != bindings["annotation"]:
        raise RuntimeError("Data loader is not bound to canonical Test")
    ann = read_json(bindings["annotation"])
    images = {int(i["id"]): i for i in ann["images"]}
    full_post = None
    if name == "GCCS":
        umr = run["ccvr"]
        params = umr["qccr_parameters"]
        full_post = PostProcessor(num_classes=5, use_focal_loss=True, num_top_queries=300,
            remap_mscoco_category=False, qccr_enabled=True,
            qccr_beta0=params["beta0"], qccr_beta_positive=params["beta_positive"],
            qccr_beta_negative=params["beta_negative"], qccr_duplicate_penalty=params["duplicate_penalty"],
            umr_enabled=True, umr_topk=umr["topk"], umr_tau=umr["tau"], umr_beta=umr["beta"],
            umr_checkpoint=umr["head_path"]).to("cuda:0").eval()
    records = {}
    tested_output_boxes = 0
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for batch_index, (samples, targets) in enumerate(loader):
            outputs = model(samples.to("cuda:0"))
            raw = outputs["pred_boxes"]
            if raw.shape[1:] != (300, 4) or raw.dtype != torch.float32:
                raise RuntimeError(f"Unexpected final query pool: {raw.shape}, {raw.dtype}")
            raw_before = raw.clone()
            ids = [int(t["image_id"].item()) for t in targets]
            sizes = torch.tensor([[images[i]["width"], images[i]["height"]] for i in ids],
                                 dtype=torch.float32, device="cuda:0")
            if full_post is not None:
                # Diagnostic postprocessing only: oracle metrics never use its selected outputs.
                returned = full_post(outputs, sizes)
                if not torch.equal(outputs["pred_boxes"], raw_before):
                    raise RuntimeError("CCVR mutated raw coordinates")
                pool_px = torchvision.ops.box_convert(raw_before, "cxcywh", "xyxy") * sizes.repeat(1, 2).unsqueeze(1)
                for index, result in enumerate(returned):
                    unchanged = (result["boxes"][:, None, :] == pool_px[index][None, :, :]).all(-1).any(-1)
                    if not bool(unchanged.all()):
                        raise RuntimeError("CCVR generated or changed a returned box")
                    tested_output_boxes += len(result["boxes"])
            normalized = raw_before.cpu().numpy()
            for local_index, image_id in enumerate(ids):
                if image_id in records:
                    raise RuntimeError("Duplicate Test image")
                width, height = images[image_id]["width"], images[image_id]["height"]
                records[image_id] = (normalized[local_index], cxcywh_to_pixel_xyxy(normalized[local_index], width, height))
            if batch_index % 10 == 0 or batch_index == len(loader) - 1:
                print(f"{name} seed {seed}: batch {batch_index + 1}/{len(loader)}, images {len(records)}", flush=True)
    if set(records) != set(images):
        raise RuntimeError("Incomplete Test coverage")
    image_ids = np.array(sorted(records), dtype=np.int64)
    normalized = np.stack([records[i][0] for i in image_ids])
    pixel_boxes = np.stack([records[i][1] for i in image_ids])
    destination.mkdir(parents=True)
    capture_path = destination / "full_query_boxes.npz"
    np.savez_compressed(capture_path, image_ids=image_ids,
                        pred_boxes_cxcywh_normalized=normalized, pred_boxes_xyxy_pixels=pixel_boxes)
    summary = {"model": name, "seed": seed, "checkpoint_sha256": run["checkpoint_sha256"],
        "config_sha256": sha256(config), "weight_source": "ema.module", "strict_load": "PASS",
        "images": len(image_ids), "queries_per_image": 300, "capture_sha256": sha256(capture_path),
        "precision": "FP32", "batch_size": 8, "iou_precision": "float64", "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "postprocessing_used_for_oracle": False,
        "ccvr_coordinate_check": ({"status": "PASS", "raw_boxes_bitwise_unchanged": True,
             "images_checked": len(image_ids), "returned_boxes_checked": tested_output_boxes,
             "returned_boxes_are_unchanged_pool_members": True,
             "oracle_metrics_before_after": "identical because the entire box pool is bitwise identical"}
             if full_post is not None else {"status": "not applicable"})}
    write_json_new(summary_path, summary)
    print(f"Capture PASS: {name} seed {seed}", flush=True)


def csv_new(path, fields, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(bindings):
    check_replay_bindings(bindings)
    ann = read_json(bindings["annotation"])
    gt = sorted(ann["annotations"], key=lambda a: (int(a["image_id"]), int(a["id"])))
    per_seed, per_gt, invalid_counts = [], [], []
    for seed in (0, 1, 2):
        values, argmax = {}, {}
        for name in ("DEIM", "GCCS"):
            destination = ROOT / f"captures/{name}/seed{seed}"
            receipt = read_json(destination / "CAPTURE_SUMMARY.json")
            if sha256(destination / "full_query_boxes.npz") != receipt["capture_sha256"]:
                raise RuntimeError("Capture SHA mismatch")
            with np.load(destination / "full_query_boxes.npz", allow_pickle=False) as payload:
                pool = {int(i): b for i, b in zip(payload["image_ids"], payload["pred_boxes_xyxy_pixels"])}
            invalid_counts.append({"model": name, "seed": seed,
                "nonpositive_width_or_height": sum(int((b[:, 2:] <= b[:, :2]).any(axis=1).sum()) for b in pool.values()),
                "total_queries": sum(len(b) for b in pool.values())})
            best_by_id, query_by_id = {}, {}
            for image in ann["images"]:
                targets = [a for a in gt if a["image_id"] == image["id"]]
                xyxy = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                                   a["bbox"][1] + a["bbox"][3]] for a in targets], dtype=np.float64).reshape(-1, 4)
                u, q = best_iou(pool[image["id"]], xyxy)
                best_by_id.update({a["id"]: float(v) for a, v in zip(targets, u)})
                query_by_id.update({a["id"]: int(v) for a, v in zip(targets, q)})
            values[name] = np.array([best_by_id[a["id"]] for a in gt])
            argmax[name] = query_by_id
            per_seed.append({"seed": seed, "model": name, **summarize(values[name])})
        for index, target in enumerate(gt):
            per_gt.append({"seed": seed, "image_id": target["image_id"], "annotation_id": target["id"],
                "category_id": target["category_id"], "DEIM_best_iou": values["DEIM"][index],
                "GCCS_best_iou": values["GCCS"][index], "delta": values["GCCS"][index] - values["DEIM"][index],
                "DEIM_argmax_query_0based": argmax["DEIM"][target["id"]],
                "GCCS_argmax_query_0based": argmax["GCCS"][target["id"]]})
    summary_rows = []
    for metric in METRICS:
        left = np.array([r[metric] for r in per_seed if r["model"] == "DEIM"])
        right = np.array([r[metric] for r in per_seed if r["model"] == "GCCS"])
        summary_rows.append({"metric": metric, "DEIM_mean": float(left.mean()), "DEIM_sample_sd": float(left.std(ddof=1)),
            "GCCS_mean": float(right.mean()), "GCCS_sample_sd": float(right.std(ddof=1)),
            "paired_delta_mean": float((right - left).mean()), "paired_delta_sample_sd": float((right - left).std(ddof=1))})
    output = ROOT / "results"
    if output.exists():
        saved = read_json(output / "SUMMARY.json")
        if saved["per_seed"] != per_seed or saved["three_seed"] != summary_rows:
            raise RuntimeError("Offline replay disagrees with saved metrics; no files overwritten")
        with (output / "PER_GT_PAIRED.csv").open(newline="", encoding="utf-8") as handle:
            saved_gt = list(csv.DictReader(handle))
        if len(saved_gt) != len(per_gt):
            raise RuntimeError("Saved per-GT row count differs")
        for old, new in zip(saved_gt, per_gt):
            if any(float(old[key]) != float(value) for key, value in new.items()):
                raise RuntimeError("Saved per-GT row differs; no files overwritten")
        print("OFFLINE REPLAY PASS: all 3213 paired GT rows and all six per-seed metrics reproduced exactly", flush=True)
        return
    output.mkdir(exist_ok=False)
    csv_new(output / "PER_GT_PAIRED.csv", list(per_gt[0]), per_gt)
    csv_new(output / "PER_SEED_METRICS.csv", list(per_seed[0]), per_seed)
    csv_new(output / "THREE_SEED_SUMMARY.csv", list(summary_rows[0]), summary_rows)
    summary = {"status": "COMPLETE", "protocol": read_json(ROOT / "protocol.json"),
               "per_seed": per_seed, "three_seed": summary_rows,
               "nonpositive_prediction_counts": invalid_counts,
               "effective_analysis_protocol_sha256": sha256(ROOT / "protocol.json"),
               "original_capture_protocol_sha256": bindings["protocol_sha256"],
               "offline_inputs_verified": True, "detector_executed_for_summary": False,
               "ccvr_checks": [read_json(ROOT / f"captures/GCCS/seed{s}/CAPTURE_SUMMARY.json")["ccvr_coordinate_check"] for s in (0, 1, 2)]}
    write_json_new(output / "SUMMARY.json", summary)
    lines = ["# Full-query oracle geometry analysis", "", "All values are GT-weighted; mean ± sample SD across detector seeds 0/1/2.", "",
             "| Metric | DEIM | GCCS | Paired delta |", "|---|---:|---:|---:|"]
    for r in summary_rows:
        scale = 100 if "coverage" in r["metric"] else 1
        lines.append(f"| {r['metric']} | {r['DEIM_mean'] * scale:.6f} ± {r['DEIM_sample_sd'] * scale:.6f} | "
                     f"{r['GCCS_mean'] * scale:.6f} ± {r['GCCS_sample_sd'] * scale:.6f} | {r['paired_delta_mean'] * scale:+.6f} |")
    lines += ["", "IoU values use the 0–1 scale. Coverage is shown as a percentage; its delta is in percentage points.",
              "", "This is oracle geometric coverage, not actual detection recall: one query may independently cover multiple GTs.",
              "No category, confidence, Top-K, NMS, or one-to-one matching is used. Results do not decompose AP gains.",
              "CCVR retained the entire raw box pool bitwise unchanged for all 237 images of each GCCS seed.",
              "All six saved full-query captures, their checkpoint bindings, and the Test annotations were verified before aggregation."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary_rows, indent=2), flush=True)
    print("COMPLETE: six captures, 3213 paired GT records, three CCVR invariance checks", flush=True)


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=SCRIPT_ROOT.parents[3])
    parser.add_argument("--output-dir", type=Path,
                        help="New external workspace for checkpoint capture; never overwrite published assets")
    parser.add_argument("--stage", choices=("preflight", "capture", "summarize", "all"), required=True)
    parser.add_argument("--model", choices=("DEIM", "GCCS"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    release = args.release_root.resolve()
    if args.output_dir is not None:
        ROOT = args.output_dir.expanduser().resolve()
        if ROOT == release or release in ROOT.parents or ROOT == SCRIPT_ROOT:
            parser.error("--output-dir must be outside the published release")
        if not ROOT.exists():
            if args.stage not in ("preflight", "all"):
                parser.error("Initialize a new workspace with --stage preflight or all")
            ROOT.mkdir(parents=True)
            shutil.copyfile(SCRIPT_ROOT / "protocol.json", ROOT / "protocol.json")
        elif not (ROOT / "protocol.json").is_file():
            parser.error("Existing output directory is not a geometry workspace")
    elif args.stage != "summarize":
        parser.error("New capture requires --output-dir outside the release; use summarize for saved captures")
    elif not (ROOT / "results/SUMMARY.json").is_file():
        parser.error("Published results are missing; do not write into the release")
    if args.stage in ("preflight", "all"):
        bindings = preflight(release)
        package_manifest()
    else:
        bindings = relocated_bindings(read_json(ROOT / "bindings.json"), release)
        if args.stage == "capture":
            check_bindings(bindings)
    if args.stage == "capture":
        if args.model is None or args.seed is None:
            parser.error("capture requires --model and --seed")
        capture(bindings, args.model, args.seed)
        check_bindings(bindings)
        package_manifest()
    elif args.stage == "summarize":
        aggregate(bindings)
    elif args.stage == "all":
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for seed in (0, 1, 2):
            for name in ("DEIM", "GCCS"):
                print(f"Starting {name} seed {seed}", flush=True)
                subprocess.run([sys.executable, "-B", "-u", str(SCRIPT_ROOT / "run_analysis.py"), "--release-root", str(release),
                                "--output-dir", str(ROOT),
                                "--stage", "capture", "--model", name, "--seed", str(seed)], env=env, check=True)
        check_bindings(bindings)
        package_manifest()
        aggregate(bindings)
        package_manifest()


if __name__ == "__main__":
    main()
