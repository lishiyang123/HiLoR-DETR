#!/usr/bin/env python3
"""Build a conservative, reproducible CDW1630 characteristics package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


TARGET_NAME = "dataset_analysis"
DATA_RELATIVE = Path("datasets/CDW1630")
SPLITS = ("Train", "Val", "Test")
SPLIT_ASSETS = {
    "Train": ("train2017", "instances_train2017.json", 1144),
    "Val": ("val2017", "instances_val2017.json", 249),
    "Test": ("test2017", "instances_test2017.json", 237),
}
SIZE_BINS = ("small", "medium", "large")
DENSITY_BINS = ("0", "1", "2", "3-5", "6-10", ">10")


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


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def project_paths(project: Path) -> dict:
    main = project / "main_host"
    data = main / DATA_RELATIVE
    target = project / TARGET_NAME
    result = {
        "project": project,
        "main": main,
        "data": data,
        "target": target,
        "main_manifest": main / "manifests/FILE_MANIFEST.sha256.json",
        "protocol": target / "protocol/DATASET_CHARACTERISTICS_PROTOCOL_LOCK.json",
        "bindings": target / "manifests/INPUT_BINDINGS.json",
        "summary": target / "results/DATASET_CHARACTERISTICS_SUMMARY.json",
        "status": target / "PACKAGE_STATUS.json",
    }
    for split, (image_dir, annotation_name, _expected) in SPLIT_ASSETS.items():
        key = split.lower()
        result[f"{key}_images"] = data / image_dir
        result[f"{key}_annotation"] = data / "annotations" / annotation_name
        result[f"local_{key}_annotation"] = target / "inputs/annotations" / annotation_name
    return result


def package_file_rows(root: Path) -> list[dict]:
    manifest = root / "manifests/PACKAGE_MANIFEST.sha256.json"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return rows


def rebuild_package_manifest(root: Path) -> None:
    write_json(root / "manifests/PACKAGE_MANIFEST.sha256.json", {
        "algorithm": "sha256",
        "format_version": 1,
        "package": root.name,
        "files": package_file_rows(root),
    })


def verify_package_manifest(root: Path) -> int:
    manifest = read_json(root / "manifests/PACKAGE_MANIFEST.sha256.json")
    expected = {row["path"]: row for row in manifest.get("files", [])}
    observed = {row["path"]: row for row in package_file_rows(root)}
    if set(expected) != set(observed):
        fail(
            "Package manifest path-set mismatch; missing=%s extra=%s"
            % (sorted(set(expected) - set(observed))[:5], sorted(set(observed) - set(expected))[:5])
        )
    for relative, row in expected.items():
        current = observed[relative]
        if current["sha256"] != row.get("sha256") or current["bytes"] != int(row.get("bytes", -1)):
            fail(f"Package manifest mismatch: {relative}")
    return len(expected)


def manifest_index(path: Path) -> dict[str, dict]:
    payload = read_json(path)
    rows = payload.get("files")
    if not isinstance(rows, list):
        fail(f"Unsupported main-package manifest: {path}")
    return {str(row["path"]): row for row in rows}


def validate_annotation(payload: dict, split: str, expected_images: int) -> dict:
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    if len(images) != expected_images:
        fail(f"{split} image count mismatch: {len(images)} != {expected_images}")
    image_ids = [int(row["id"]) for row in images]
    category_ids = [int(row["id"]) for row in categories]
    if len(set(image_ids)) != len(image_ids):
        fail(f"{split} image IDs are not unique")
    if len(categories) != 5 or len(set(category_ids)) != 5:
        fail(f"{split} must contain five uniquely identified categories")
    image_set = set(image_ids)
    category_set = set(category_ids)
    annotation_ids = []
    for row in annotations:
        annotation_ids.append(int(row["id"]))
        bbox = row.get("bbox", [])
        if int(row.get("image_id", -1)) not in image_set:
            fail(f"{split} annotation references an unknown image: {row.get('id')}")
        if int(row.get("category_id", -1)) not in category_set:
            fail(f"{split} annotation references an unknown category: {row.get('id')}")
        if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
            fail(f"{split} annotation has an invalid bbox: {row.get('id')}")
        if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            fail(f"{split} annotation has a non-positive bbox: {row.get('id')}")
    if len(set(annotation_ids)) != len(annotation_ids):
        fail(f"{split} annotation IDs are not unique")
    return {
        "images": {int(row["id"]): row for row in images},
        "annotations": annotations,
        "categories": {int(row["id"]): str(row["name"]) for row in categories},
    }


def validate_main(project: Path) -> tuple[dict, dict[str, dict]]:
    p = project_paths(project)
    if not p["main_manifest"].is_file():
        fail(f"Missing final main-package manifest: {p['main_manifest']}")
    manifest = manifest_index(p["main_manifest"])
    parsed = {}
    category_reference = None
    for split in SPLITS:
        key = split.lower()
        image_root = p[f"{key}_images"]
        annotation_path = p[f"{key}_annotation"]
        if not image_root.is_dir():
            fail(f"Missing {split} image directory: {image_root}")
        checked = validate_annotation(read_json(annotation_path), split, SPLIT_ASSETS[split][2])
        missing = [
            str(image_root / row["file_name"])
            for row in checked["images"].values()
            if not (image_root / row["file_name"]).is_file()
        ]
        if missing:
            fail(f"{split} images are incomplete: {missing[:5]}")
        categories = checked["categories"]
        if category_reference is None:
            category_reference = categories
        elif categories != category_reference:
            fail(f"{split} category mapping differs from Train")
        relative = annotation_path.relative_to(p["main"]).as_posix()
        row = manifest.get(relative)
        if row is None or row.get("sha256") != sha256(annotation_path):
            fail(f"Main-package SHA binding failed: {relative}")
        parsed[split] = checked
    return p, parsed


def size_bin(area: float) -> str:
    if area < 32.0 ** 2:
        return "small"
    if area < 96.0 ** 2:
        return "medium"
    return "large"


def density_bin(count: int) -> str:
    if count <= 2:
        return str(count)
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    return ">10"


def protocol_payload() -> dict:
    return {
        "protocol_name": "CDW1630_DATASET_CHARACTERISTICS_CONSERVATIVE_v1",
        "protocol_version": "1.0",
        "status": "FROZEN_BEFORE_ANALYSIS",
        "analysis_profile": "CONSERVATIVE_CORE_ONLY",
        "dataset_path_in_main_package": DATA_RELATIVE.as_posix(),
        "splits": list(SPLITS),
        "expected_image_counts": {split: SPLIT_ASSETS[split][2] for split in SPLITS},
        "expected_category_count": 5,
        "included_statistics": [
            "split image and valid-instance counts",
            "class instance and image-presence distributions",
            "COCO small/medium/large bbox-area distributions",
            "annotated-object count per image",
        ],
        "explicitly_excluded": [
            "box-overlap or IoMin analysis",
            "class co-occurrence analysis",
            "pixel-derived scene analysis",
            "model predictions, checkpoints, training, or inference",
            "hypothesis testing or causal claims",
        ],
        "size_definition": "small < 32^2, medium >= 32^2 and < 96^2, large >= 96^2 bbox pixels",
        "pixel_policy": "image paths are existence-checked; image pixels are never opened",
        "test_policy": "frozen Test annotations are used for neutral dataset description only",
        "test_used_for_model_or_parameter_selection": False,
        "gpu_required": False,
    }


def protocol_markdown() -> str:
    return """# CDW1630 dataset-characteristics protocol

Status: **frozen before analysis**. Profile: **conservative core only**.

The analysis reads the final COCO annotation JSON files for the frozen Train,
Val, and Test splits and reports only four neutral characteristics: split
size, category distribution, COCO bounding-box scale, and annotated-object
count per image. Image paths are checked for completeness, but image pixels
are never opened.

The package intentionally excludes annotation-box overlap, IoMin, class
co-occurrence, model outputs, significance testing, and causal interpretation.
These exclusions prevent descriptive statistics from being presented as
post-hoc justification for the method design.

Small, medium, and large follow the standard COCO bounding-box area cutoffs of
`32^2` and `96^2` pixels. Frozen Test annotations are used only for dataset
description and never for model, checkpoint, or parameter selection.
"""


def package_readme() -> str:
    return """# CDW1630 Dataset Characteristics Reproduction Package v1

This is a conservative, annotation-only evidence package. It reproduces the
basic descriptive statistics of the final CDW1630 Train/Val/Test splits and
binds every input to `main_host` with SHA-256.

Contents:

- `inputs/annotations/`: local copies of the three final COCO JSON files;
- `tables/`: canonical CSV tables;
- `results/`: machine-readable and paper-facing summaries;
- `protocol/`: frozen scope and interpretation boundary;
- `manifests/`: input bindings and package checksum manifest;
- `source/`: the exact analysis program.

The original images are not duplicated. Their existence is verified against
the formal dataset tree. No pixels, models, checkpoints, predictions, GPU,
training, or inference are used. Numerical statistics are reproduced without figure generation.
"""


def preflight(project: Path) -> tuple[dict, dict[str, dict]]:
    p, parsed = validate_main(project)
    total_images = sum(len(parsed[split]["images"]) for split in SPLITS)
    total_instances = sum(len(parsed[split]["annotations"]) for split in SPLITS)
    if total_images != 1630:
        fail(f"CDW1630 total image count mismatch: {total_images}")
    print("===== CDW1630 DATASET CHARACTERISTICS PREFLIGHT =====")
    print("Main package :", p["main"])
    print("Dataset path :", p["data"])
    print("Splits       : Train=1144 / Val=249 / Test=237 / Total=1630")
    print("Instances    :", total_instances, "valid annotation boxes")
    print("Categories   : 5; ID/name mapping consistent across splits")
    print("Formal scope : split / class / COCO scale / objects per image")
    print("Excluded     : overlap / IoMin / co-occurrence / method claims")
    print("Pixels/models: NOT ACCESSED / NOT ACCESSED")
    print("GPU/Test use : NOT REQUIRED / DESCRIPTION ONLY")
    print("Writes       : NO")
    print("CDW1630 DATASET CHARACTERISTICS PREFLIGHT: PASS")
    return p, parsed


def verify_prepared(p: dict) -> None:
    protocol = read_json(p["protocol"])
    if protocol.get("status") != "FROZEN_BEFORE_ANALYSIS":
        fail("Dataset-characteristics protocol is not frozen")
    if protocol.get("analysis_profile") != "CONSERVATIVE_CORE_ONLY":
        fail("Dataset-characteristics protocol is not conservative-core")
    if protocol.get("test_used_for_model_or_parameter_selection") is not False:
        fail("Protocol declares Test-based selection")
    bindings = read_json(p["bindings"])
    if bindings.get("image_pixels_accessed") is not False:
        fail("Input binding does not prohibit pixel access")
    for split in SPLITS:
        row = bindings["annotations"][split]
        local = p[f"local_{split.lower()}_annotation"]
        if not local.is_file() or local.stat().st_size != int(row["bytes"]) or sha256(local) != row["sha256"]:
            fail(f"Prepared {split} annotation binding mismatch")
        validate_annotation(read_json(local), split, SPLIT_ASSETS[split][2])


def prepare(project: Path) -> None:
    p, _parsed = preflight(project)
    target = p["target"]
    if target.exists() and any(target.iterdir()):
        if p["protocol"].is_file() and p["bindings"].is_file():
            verify_prepared(p)
            installed = target / "source/analyze_dataset.py"
            current = Path(__file__).resolve()
            if not installed.is_file() or sha256(installed) != sha256(current):
                shutil.copy2(current, installed)
                rebuild_package_manifest(target)
                print("Existing verified preparation retained; analysis source refreshed")
            else:
                print("Existing verified preparation retained; writes performed: NO")
            return
        fail(f"Refusing to overwrite a partial target: {target}")
    for relative in (
        "inputs/annotations", "protocol", "source", "tables", "results",
        "environment", "manifests",
    ):
        (target / relative).mkdir(parents=True, exist_ok=True)
    annotation_rows = {}
    for split in SPLITS:
        key = split.lower()
        source = p[f"{key}_annotation"]
        local = p[f"local_{key}_annotation"]
        shutil.copy2(source, local)
        annotation_rows[split] = {
            "source_project_relative_path": source.relative_to(project).as_posix(),
            "local_path": local.relative_to(target).as_posix(),
            "bytes": local.stat().st_size,
            "sha256": sha256(local),
            "expected_images": SPLIT_ASSETS[split][2],
            "image_directory_project_relative_path": p[f"{key}_images"].relative_to(project).as_posix(),
        }
    shutil.copy2(Path(__file__).resolve(), target / "source/analyze_dataset.py")
    write_json(p["protocol"], protocol_payload())
    write_text(target / "protocol/DATASET_CHARACTERISTICS_PROTOCOL.md", protocol_markdown())
    write_text(target / "README.md", package_readme())
    write_json(p["bindings"], {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "main_package_manifest": {
            "project_relative_path": p["main_manifest"].relative_to(project).as_posix(),
            "sha256": sha256(p["main_manifest"]),
        },
        "annotations": annotation_rows,
        "image_pixels_accessed": False,
        "test_used_for_model_or_parameter_selection": False,
    })
    write_json(target / "environment/PREPARE_ENVIRONMENT.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "gpu_required": False,
        "image_pixels_accessed": False,
    })
    write_json(p["status"], {
        "status": "PREPARED__ANALYSIS_PENDING",
        "analysis_profile": "CONSERVATIVE_CORE_ONLY",
        "claimed_results": 0,
        "test_used_for_model_or_parameter_selection": False,
    })
    rebuild_package_manifest(target)
    verify_prepared(p)
    print("===== CDW1630 DATASET CHARACTERISTICS PREPARATION =====")
    print("Package      :", target)
    print("Protocol     : FROZEN / CONSERVATIVE CORE")
    print("Local inputs : three final annotation JSON files")
    print("Images       : referenced and completeness-checked; not duplicated")
    print("Analysis     : PENDING")
    print("CDW1630 DATASET CHARACTERISTICS PREPARATION: PASS")


def analyze_one_split(split: str, checked: dict) -> dict:
    images = checked["images"]
    annotations = checked["annotations"]
    categories = checked["categories"]
    by_image = defaultdict(list)
    for row in annotations:
        by_image[int(row["image_id"])].append(row)

    class_instances = Counter()
    class_images = Counter()
    size_counts = Counter()
    density_rows = []
    counts = []
    for image_id in sorted(images):
        image_rows = by_image.get(image_id, [])
        count = len(image_rows)
        counts.append(count)
        density_rows.append({
            "split": split,
            "image_id": image_id,
            "file_name": images[image_id]["file_name"],
            "annotated_objects": count,
            "density_bin": density_bin(count),
        })
        for category_id in {int(row["category_id"]) for row in image_rows}:
            class_images[category_id] += 1
        for row in image_rows:
            category_id = int(row["category_id"])
            width = float(row["bbox"][2])
            height = float(row["bbox"][3])
            class_instances[category_id] += 1
            size_counts[size_bin(width * height)] += 1

    image_count = len(images)
    instance_count = len(annotations)
    split_row = {
        "split": split,
        "images": image_count,
        "instances": instance_count,
        "objects_per_image_mean": instance_count / image_count,
        "objects_per_image_median": statistics.median(counts),
        "objects_per_image_min": min(counts),
        "objects_per_image_max": max(counts),
    }
    class_rows = []
    for category_id in sorted(categories):
        instances = class_instances[category_id]
        containing = class_images[category_id]
        class_rows.append({
            "split": split,
            "category_id": category_id,
            "category": categories[category_id],
            "instances": instances,
            "instance_percent": 100.0 * instances / instance_count if instance_count else 0.0,
            "images_containing_class": containing,
            "image_percent": 100.0 * containing / image_count if image_count else 0.0,
        })
    size_rows = []
    for label in SIZE_BINS:
        count = size_counts[label]
        size_rows.append({
            "split": split,
            "size_bin": label,
            "instances": count,
            "percent": 100.0 * count / instance_count if instance_count else 0.0,
        })
    density_counts = Counter(row["density_bin"] for row in density_rows)
    density_summary = []
    for label in DENSITY_BINS:
        count = density_counts[label]
        density_summary.append({
            "split": split,
            "density_bin": label,
            "images": count,
            "percent": 100.0 * count / image_count if image_count else 0.0,
        })
    return {
        "split": split_row,
        "class": class_rows,
        "size": size_rows,
        "density_images": density_rows,
        "density_summary": density_summary,
    }


def aggregate_results(per_split: dict[str, dict], categories: dict[int, str]) -> dict:
    split_rows = [per_split[split]["split"] for split in SPLITS]
    total_images = sum(row["images"] for row in split_rows)
    total_instances = sum(row["instances"] for row in split_rows)
    all_counts = [
        row["annotated_objects"]
        for split in SPLITS
        for row in per_split[split]["density_images"]
    ]
    split_row = {
        "split": "All",
        "images": total_images,
        "instances": total_instances,
        "objects_per_image_mean": total_instances / total_images,
        "objects_per_image_median": statistics.median(all_counts),
        "objects_per_image_min": min(all_counts),
        "objects_per_image_max": max(all_counts),
    }
    class_rows = []
    for category_id in sorted(categories):
        source = [
            next(row for row in per_split[split]["class"] if row["category_id"] == category_id)
            for split in SPLITS
        ]
        instances = sum(row["instances"] for row in source)
        containing = sum(row["images_containing_class"] for row in source)
        class_rows.append({
            "split": "All",
            "category_id": category_id,
            "category": categories[category_id],
            "instances": instances,
            "instance_percent": 100.0 * instances / total_instances,
            "images_containing_class": containing,
            "image_percent": 100.0 * containing / total_images,
        })
    size_rows = []
    for label in SIZE_BINS:
        count = sum(
            next(row["instances"] for row in per_split[split]["size"] if row["size_bin"] == label)
            for split in SPLITS
        )
        size_rows.append({
            "split": "All", "size_bin": label, "instances": count,
            "percent": 100.0 * count / total_instances,
        })
    density_counts = Counter(density_bin(value) for value in all_counts)
    density_summary = [
        {
            "split": "All", "density_bin": label, "images": density_counts[label],
            "percent": 100.0 * density_counts[label] / total_images,
        }
        for label in DENSITY_BINS
    ]
    return {
        "split": split_row,
        "class": class_rows,
        "size": size_rows,
        "density_summary": density_summary,
    }




def paper_summary(summary: dict) -> str:
    total = next(row for row in summary["split_summary"] if row["split"] == "All")
    class_text = ", ".join(
        f"{row['category']} ({row['instances']})" for row in summary["class_distribution_all"]
    )
    size_text = ", ".join(
        f"{row['size_bin']} {row['percent']:.2f}%" for row in summary["size_distribution_all"]
    )
    return f"""# Paper-facing neutral dataset summary

CDW1630 contains {total['images']} images and {total['instances']} valid
bounding-box instances. The frozen Train, Val, and Test partitions contain
1144, 249, and 237 images, respectively. The five category totals are:
{class_text}.

Using standard COCO bounding-box area thresholds, the scale distribution is
{size_text}. Images contain a mean of {total['objects_per_image_mean']:.3f}
annotated objects (median {total['objects_per_image_median']:.1f}, range
{total['objects_per_image_min']}--{total['objects_per_image_max']}).

These values are descriptive dataset statistics only. They are not used as
post-hoc evidence for architecture design, model selection, or parameter
selection.
"""




def verify_external_bindings(p: dict) -> None:
    bindings = read_json(p["bindings"])
    manifest_row = bindings["main_package_manifest"]
    main_manifest = p["project"] / manifest_row["project_relative_path"]
    if not main_manifest.is_file() or sha256(main_manifest) != manifest_row["sha256"]:
        fail("Main-package manifest changed after preparation")
    for split in SPLITS:
        row = bindings["annotations"][split]
        source = p["project"] / row["source_project_relative_path"]
        if not source.is_file() or source.stat().st_size != int(row["bytes"]) or sha256(source) != row["sha256"]:
            fail(f"External {split} annotation changed after preparation")


def run(project: Path) -> None:
    p = project_paths(project)
    verify_prepared(p)
    verify_external_bindings(p)
    if p["summary"].is_file():
        verify(project)
        print("Existing verified dataset-characteristics results retained; writes performed: NO")
        return
    parsed = {
        split: validate_annotation(
            read_json(p[f"local_{split.lower()}_annotation"]), split, SPLIT_ASSETS[split][2]
        )
        for split in SPLITS
    }
    per_split = {split: analyze_one_split(split, parsed[split]) for split in SPLITS}
    categories = parsed["Train"]["categories"]
    aggregate = aggregate_results(per_split, categories)
    split_rows = [per_split[split]["split"] for split in SPLITS] + [aggregate["split"]]
    class_rows = [row for split in SPLITS for row in per_split[split]["class"]] + aggregate["class"]
    size_rows = [row for split in SPLITS for row in per_split[split]["size"]] + aggregate["size"]
    density_image_rows = [row for split in SPLITS for row in per_split[split]["density_images"]]
    density_summary_rows = [
        row for split in SPLITS for row in per_split[split]["density_summary"]
    ] + aggregate["density_summary"]

    write_csv(p["target"] / "tables/SPLIT_SUMMARY.csv", split_rows, list(split_rows[0]))
    write_csv(p["target"] / "tables/CLASS_DISTRIBUTION.csv", class_rows, list(class_rows[0]))
    write_csv(p["target"] / "tables/SIZE_DISTRIBUTION.csv", size_rows, list(size_rows[0]))
    write_csv(p["target"] / "tables/IMAGE_OBJECT_COUNTS.csv", density_image_rows, list(density_image_rows[0]))
    write_csv(p["target"] / "tables/OBJECT_DENSITY_DISTRIBUTION.csv", density_summary_rows, list(density_summary_rows[0]))

    summary = {
        "status": "COMPLETE__CONSERVATIVE_CORE_ONLY",
        "dataset": "CDW1630",
        "dataset_path_in_main_package": DATA_RELATIVE.as_posix(),
        "protocol_sha256": sha256(p["protocol"]),
        "input_bindings_sha256": sha256(p["bindings"]),
        "categories": categories,
        "split_summary": split_rows,
        "class_distribution_all": aggregate["class"],
        "size_distribution_all": aggregate["size"],
        "density_distribution_all": aggregate["density_summary"],
        "excluded_analyses": ["box overlap", "IoMin", "class co-occurrence", "pixel-derived analysis"],
        "image_pixels_accessed": False,
        "model_checkpoint_prediction_accessed": False,
        "gpu_used": False,
        "test_annotation_used_for_neutral_description": True,
        "test_used_for_model_or_parameter_selection": False,
    }
    write_json(p["summary"], summary)
    write_text(p["target"] / "results/PAPER_SUMMARY.md", paper_summary(summary))
    write_json(p["status"], {
        "status": "COMPLETE__VERIFIED",
        "analysis_profile": "CONSERVATIVE_CORE_ONLY",
        "claimed_results": 1,
        "gpu_used": False,
        "test_used_for_model_or_parameter_selection": False,
    })
    rebuild_package_manifest(p["target"])
    verify(project)
    print("CDW1630 DATASET CHARACTERISTICS ANALYSIS: PASS")


def check_export(path: Path, signature: bytes | None = None) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"Missing or empty export: {path}")
    if signature is not None:
        with path.open("rb") as handle:
            if handle.read(len(signature)) != signature:
                fail(f"Invalid export signature: {path}")


def verify(project: Path) -> None:
    p = project_paths(project)
    _validated_paths, _parsed = validate_main(project)
    verify_prepared(p)
    verify_external_bindings(p)
    summary = read_json(p["summary"])
    if summary.get("status") != "COMPLETE__CONSERVATIVE_CORE_ONLY":
        fail("Dataset-characteristics result status mismatch")
    if summary.get("image_pixels_accessed") is not False:
        fail("Result does not preserve the no-pixel-access policy")
    if summary.get("model_checkpoint_prediction_accessed") is not False:
        fail("Result claims model/checkpoint/prediction access")
    if summary.get("test_used_for_model_or_parameter_selection") is not False:
        fail("Result claims Test-based model or parameter selection")
    if set(summary.get("excluded_analyses", [])) != {
        "box overlap", "IoMin", "class co-occurrence", "pixel-derived analysis"
    }:
        fail("Conservative-scope exclusions changed")
    split_rows = {row["split"]: row for row in summary.get("split_summary", [])}
    if [split_rows[split]["images"] for split in SPLITS] != [1144, 249, 237]:
        fail("Frozen split counts changed")
    if split_rows["All"]["images"] != 1630:
        fail("Total image count changed")
    if len(summary.get("class_distribution_all", [])) != 5:
        fail("Class distribution is incomplete")
    if len(summary.get("size_distribution_all", [])) != 3:
        fail("Size distribution is incomplete")
    for name in (
        "SPLIT_SUMMARY.csv", "CLASS_DISTRIBUTION.csv", "SIZE_DISTRIBUTION.csv",
        "IMAGE_OBJECT_COUNTS.csv", "OBJECT_DENSITY_DISTRIBUTION.csv",
    ):
        check_export(p["target"] / "tables" / name)
    count = verify_package_manifest(p["target"])
    print("===== CDW1630 DATASET CHARACTERISTICS VERIFICATION =====")
    print("Dataset path      : PASS / datasets/CDW1630")
    print("Splits/images     : PASS (1144/249/237; total 1630)")
    print("Annotation SHA    : PASS / bound to final main package")
    print("Categories        : PASS (5 consistent mappings)")
    print("Class/scale/density: PASS / conservative core only")
    print("Overlap/co-occur  : EXCLUDED")
    print("Pixels/GPU/models : NOT ACCESSED / NOT USED / NOT ACCESSED")
    print("Test model/param  : NO")
    print("Package manifest  : PASS (%d files)" % count)
    print("CDW1630 DATASET CHARACTERISTICS VERIFICATION: PASS")


def summary_command(project: Path) -> None:
    p = project_paths(project)
    summary = read_json(p["summary"])
    total = next(row for row in summary["split_summary"] if row["split"] == "All")
    print("===== CDW1630 DATASET CHARACTERISTICS SUMMARY =====")
    print("Images/instances : %d / %d" % (total["images"], total["instances"]))
    print("Objects per image: mean=%.3f median=%.1f range=%d..%d" % (
        total["objects_per_image_mean"], total["objects_per_image_median"],
        total["objects_per_image_min"], total["objects_per_image_max"],
    ))
    print("Class instances  :")
    for row in summary["class_distribution_all"]:
        print("  %-20s %5d (%6.2f%%)" % (row["category"], row["instances"], row["instance_percent"]))
    print("COCO size bins   :", ", ".join(
        "%s=%.2f%%" % (row["size_bin"], row["percent"])
        for row in summary["size_distribution_all"]
    ))
    print("Scope            : conservative descriptive core only")
    print("Test selection   : NO")


def status(project: Path) -> None:
    p = project_paths(project)
    print("===== CDW1630 DATASET CHARACTERISTICS STATUS =====")
    print("Prepared:", p["protocol"].is_file() and p["bindings"].is_file())
    print("Complete:", p["summary"].is_file())
    print("Package :", p["target"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "prepare", "run", "summary", "verify", "status"))
    parser.add_argument("--project", type=Path, default=Path("/root/autodl-tmp/BTC_DEIM_project"))
    args = parser.parse_args()
    project = args.project.resolve()
    actions = {
        "preflight": lambda: preflight(project),
        "prepare": lambda: prepare(project),
        "run": lambda: run(project),
        "summary": lambda: summary_command(project),
        "verify": lambda: verify(project),
        "status": lambda: status(project),
    }
    actions[args.action]()


if __name__ == "__main__":
    main()
