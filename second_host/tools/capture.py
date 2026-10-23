"""Train/Val capture routines used by reproduce_pipeline.py.

No standalone run registry or Test fitting entry is provided here.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import torch
import yaml

TOPK_BOUNDARY_TIE_POLICY = "record_native_torch_topk_exact_boundary_ties"
ADAPTER_ROOT = Path(__file__).resolve().parents[1]


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(f"{value}\n".encode("utf-8"))
    return digest.hexdigest()


def image_tree_sha256(image_dir: Path, file_names: list[str], *, split: str) -> str:
    """Hash every and only COCO-listed image, rejecting tree drift first."""

    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise RuntimeError(f"{split} image directory is missing")
    if len(file_names) != len(set(file_names)):
        raise RuntimeError(f"{split} annotation contains duplicate file names")
    expected_names = sorted(file_names)
    actual_names = sorted(
        path.relative_to(image_dir).as_posix()
        for path in image_dir.rglob("*")
        if path.is_file()
    )
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        extras = sorted(set(actual_names) - set(expected_names))
        raise RuntimeError(
            f"{split} image tree differs from annotation: "
            f"missing={missing[:5]}, extras={extras[:5]}"
        )
    digest = hashlib.sha256()
    for file_name in expected_names:
        image_path = (image_dir / file_name).resolve()
        if not _inside(image_path, image_dir) or not image_path.is_file():
            raise RuntimeError(f"missing or escaping {split} image: {file_name}")
        digest.update(f"{sha256(image_path)}  ./{file_name}\n".encode("utf-8"))
    return digest.hexdigest()


def _canonical_config_value(value):
    """Return a JSON-safe, deterministic representation of a resolved config."""
    if isinstance(value, dict):
        return {
            str(key): _canonical_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("resolved config contains a non-finite float")
        return value
    raise RuntimeError(
        f"resolved config contains unsupported value type: {type(value).__name__}"
    )


def canonical_json_sha256(value) -> str:
    canonical = json.dumps(
        _canonical_config_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def config_provenance(config_path: Path, resolved_config: dict) -> dict:
    """Bind the runtime-resolved config and the complete ordered include graph."""
    root = config_path.resolve()
    graph = {}
    active = []

    def visit(path: Path) -> None:
        path = path.resolve()
        key = str(path)
        if path in active:
            cycle = " -> ".join(str(item) for item in [*active, path])
            raise RuntimeError(f"config include cycle detected: {cycle}")
        if key in graph:
            return
        if not path.is_file():
            raise RuntimeError(f"config include is missing: {path}")
        if path.suffix.lower() not in {".yml", ".yaml"}:
            raise RuntimeError(f"config include is not YAML: {path}")
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.Loader)
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise RuntimeError(f"config root must be a mapping: {path}")
        raw_includes = payload.get("__include__", [])
        if isinstance(raw_includes, (str, bytes)) or not isinstance(
            raw_includes, (list, tuple)
        ):
            raise RuntimeError(f"config __include__ must be a sequence: {path}")
        includes = []
        for raw_include in raw_includes:
            if not isinstance(raw_include, str) or not raw_include:
                raise RuntimeError(f"config include must be a non-empty string: {path}")
            expanded = Path(os.path.expanduser(raw_include))
            include_path = (
                expanded if expanded.is_absolute() else path.parent / expanded
            ).resolve()
            includes.append(str(include_path))

        active.append(path)
        try:
            for include in includes:
                visit(Path(include))
        finally:
            active.pop()
        graph[key] = {
            "path": key,
            "sha256": sha256(path),
            "includes_in_declared_order": includes,
        }

    visit(root)
    closure = {
        "root": str(root),
        "files": [graph[key] for key in sorted(graph)],
    }
    return {
        "resolved_config_sha256": canonical_json_sha256(resolved_config),
        "include_closure_sha256": canonical_json_sha256(closure),
        "include_closure": closure,
    }


def audit_topk_boundary(scores: torch.Tensor, topk: int) -> list[dict]:
    """Audit whether an exact post-sigmoid score tie straddles Top-k."""
    if scores.ndim < 2:
        raise RuntimeError("Top-k boundary audit expects a batched score tensor")
    flattened = scores.flatten(1)
    candidate_count = int(flattened.shape[1])
    if not torch.isfinite(flattened).all():
        raise RuntimeError("Top-k boundary audit found non-finite post-sigmoid scores")
    if topk <= 0 or topk > candidate_count:
        raise RuntimeError(
            f"invalid Top-k boundary audit k={topk} for {candidate_count} candidates"
        )
    audit_k = min(topk + 1, candidate_count)
    selected_scores = torch.topk(flattened, audit_k, dim=-1, sorted=True).values
    boundary_scores = selected_scores[:, topk - 1]
    reports = []
    for batch_index in range(int(flattened.shape[0])):
        row = flattened[batch_index]
        boundary = boundary_scores[batch_index]
        strictly_above = int((row > boundary).sum().item())
        equal_boundary = int((row == boundary).sum().item())
        selected_equal = topk - strictly_above
        boundary_tie = equal_boundary > selected_equal
        next_score = (
            float(selected_scores[batch_index, topk].item())
            if topk < candidate_count
            else None
        )
        boundary_score = float(boundary.item())
        reports.append(
            {
                "batch_index": batch_index,
                "candidate_count": candidate_count,
                "topk": int(topk),
                "boundary_score": boundary_score,
                "first_excluded_score": next_score,
                "boundary_margin": (
                    boundary_score - next_score if next_score is not None else None
                ),
                "strictly_above_count": strictly_above,
                "equal_boundary_count": equal_boundary,
                "selected_equal_boundary_count": selected_equal,
                "excluded_equal_boundary_count": equal_boundary - selected_equal,
                "boundary_tie": boundary_tie,
                "policy": TOPK_BOUNDARY_TIE_POLICY,
            }
        )
    return reports


def enforce_topk_boundary_policy(report: dict, *, split: str, image_id: int) -> None:
    if report.get("policy") != TOPK_BOUNDARY_TIE_POLICY:
        raise RuntimeError("unknown Top-k boundary tie policy in audit report")
    if not isinstance(report.get("boundary_tie"), bool):
        raise RuntimeError(f"{split} image {image_id}: invalid Top-k tie record")


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=-1,
    )


def checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload.get("ema"), dict) and isinstance(
        payload["ema"].get("module"), dict
    ):
        return payload["ema"]["module"], "ema.module"
    if isinstance(payload.get("model"), dict):
        return payload["model"], "model"
    raise RuntimeError("checkpoint has neither ema.module nor model state")


def load_annotation(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in payload["images"]}
    annotations = {image_id: [] for image_id in images}
    for annotation in payload["annotations"]:
        annotations.setdefault(int(annotation["image_id"]), []).append(annotation)
    return payload, images, annotations


def validate_split_binding(
    split, annotation_path, image_dir, payload, data_lock, *, verify_image_tree=True
):
    if data_lock.get("schema") != "cdw1630-rtdetr-r18-hilor-data-split-lock-v1":
        raise RuntimeError("invalid data split lock schema")
    eligible = data_lock.get("selection_eligible_splits")
    if split not in eligible or split not in {"train", "val"}:
        raise RuntimeError(f"split is not eligible for fitting capture: {split}")
    binding = data_lock["splits"][split]
    test_binding = data_lock["splits"]["test"]
    actual_annotation = str(annotation_path.resolve())
    actual_image_dir = str(image_dir.resolve())
    if actual_annotation != str(Path(binding["annotation"]).resolve()):
        raise RuntimeError(f"{split} annotation path is not protocol-bound")
    if actual_image_dir != str(Path(binding["images_dir"]).resolve()):
        raise RuntimeError(f"{split} image directory is not protocol-bound")
    annotation_sha = sha256(annotation_path)
    if annotation_sha != binding["annotation_sha256"]:
        raise RuntimeError(f"{split} annotation SHA-256 mismatch")
    if annotation_sha == test_binding["annotation_sha256"]:
        raise RuntimeError("Test annotation is forbidden in Train/Val capture")

    images = payload.get("images", [])
    image_ids = [int(image["id"]) for image in images]
    file_names = [str(image["file_name"]) for image in images]
    if len(image_ids) != len(set(image_ids)):
        raise RuntimeError(f"{split} annotation contains duplicate image ids")
    image_id_sha = sha256_lines(sorted(image_ids))
    file_name_sha = sha256_lines(sorted(file_names))
    if image_id_sha != binding["image_id_set_sha256"]:
        raise RuntimeError(f"{split} image-id-set SHA-256 mismatch")
    if file_name_sha != binding["file_name_set_sha256"]:
        raise RuntimeError(f"{split} file-name-set SHA-256 mismatch")
    if image_id_sha == test_binding["image_id_set_sha256"]:
        raise RuntimeError("Test image-id set is forbidden in Train/Val capture")
    if len(images) != int(binding["image_count"]):
        raise RuntimeError(f"{split} image count mismatch")
    if len(payload.get("annotations", [])) != int(binding["instance_count"]):
        raise RuntimeError(f"{split} instance count mismatch")
    category_ids = sorted(int(item["id"]) for item in payload.get("categories", []))
    if category_ids != list(data_lock["category_ids"]):
        raise RuntimeError(f"{split} category ids mismatch")
    image_tree_sha = binding.get("image_tree_sha256")
    if verify_image_tree:
        image_tree_sha = image_tree_sha256(image_dir, file_names, split=split)
        if image_tree_sha != binding.get("image_tree_sha256"):
            raise RuntimeError(f"{split} image-tree SHA-256 mismatch")
    return {
        "annotation_sha256": annotation_sha,
        "image_id_set_sha256": image_id_sha,
        "file_name_set_sha256": file_name_sha,
        "image_tree_sha256": image_tree_sha,
        "image_count": len(images),
        "instance_count": len(payload.get("annotations", [])),
    }


def atomic_torch_save(value, path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite evidence file: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"temporary output already exists: {temporary}")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ground_truth(image_id, images, annotations):
    image = images[image_id]
    width, height = int(image["width"]), int(image["height"])
    boxes, labels, areas, crowds = [], [], [], []
    for annotation in annotations.get(image_id, []):
        x, y, width_box, height_box = map(float, annotation["bbox"])
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(width), x + width_box), min(float(height), y + height_box)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
        labels.append(int(annotation["category_id"]))
        areas.append(float(annotation.get("area", (x2 - x1) * (y2 - y1))))
        crowds.append(int(annotation.get("iscrowd", 0)))
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "area": torch.tensor(areas, dtype=torch.float32),
        "iscrowd": torch.tensor(crowds, dtype=torch.int64),
        "height": height,
        "width": width,
    }


def manual_native_post(logits, boxes, sizes_wh, num_classes, topk):
    absolute_boxes = cxcywh_to_xyxy(boxes) * sizes_wh.repeat(1, 2).unsqueeze(1)
    score, flat_index = torch.topk(torch.sigmoid(logits).flatten(1), topk, dim=-1)
    label = flat_index % num_classes
    query = flat_index // num_classes
    selected_box = absolute_boxes.gather(
        1, query.unsqueeze(-1).expand(-1, -1, absolute_boxes.shape[-1])
    )
    return query, label, score, selected_box


def capture_split(
    *, split, config_path, model, postprocessor, device, checkpoint_path,
    checkpoint_sha, checkpoint_state_source, training_run_evidence, output_dir,
    prefix, data_lock, data_lock_sha,
):
    from src.core import YAMLConfig
    from src.data import CocoEvaluator, get_coco_api_from_dataset

    cfg = YAMLConfig(str(config_path))
    config_binding = config_provenance(config_path, cfg.yaml_cfg)
    loader = cfg.val_dataloader
    # This pinned RT-DETR revision does not expose an evaluator through
    # YAMLConfig.  Reproduce det_engine.evaluate's native construction
    # explicitly so the frozen baseline evidence uses the exact host COCO
    # evaluator rather than an adapter-specific implementation.
    base_dataset = get_coco_api_from_dataset(loader.dataset)
    evaluator = CocoEvaluator(base_dataset, postprocessor.iou_types)
    if not bool(getattr(postprocessor, "use_focal_loss", False)):
        raise RuntimeError(
            "frozen Q x C capture requires native focal-loss sigmoid Top-k"
        )
    if loader.shuffle:
        raise RuntimeError(f"{split} capture loader must not shuffle")
    dataset_config = cfg.yaml_cfg["val_dataloader"]["dataset"]
    annotation_path = Path(dataset_config["ann_file"]).resolve()
    image_dir = Path(dataset_config["img_folder"]).resolve()
    payload, images, annotations = load_annotation(annotation_path)
    split_binding = validate_split_binding(
        split, annotation_path, image_dir, payload, data_lock,
        verify_image_tree=False,
    )
    records, query_counts = [], set()
    boundary_audits = []
    maximum_score_error = 0.0
    maximum_box_error = 0.0

    with torch.inference_mode():
        for batch_index, (samples, targets) in enumerate(loader):
            samples = samples.to(device)
            outputs = model(samples)
            if "hilor_decoder_features" not in outputs:
                raise RuntimeError("capture config did not expose final decoder features")
            image_ground_truth, sizes_wh = [], []
            for target in targets:
                image_id = int(target["image_id"].item())
                gt = ground_truth(image_id, images, annotations)
                image_ground_truth.append((image_id, gt))
                sizes_wh.append((gt["width"], gt["height"]))
            sizes_wh = torch.tensor(sizes_wh, dtype=torch.float32, device=device)
            native = postprocessor(outputs, sizes_wh)
            batch_boundary_audits = audit_topk_boundary(
                torch.sigmoid(outputs["pred_logits"]),
                int(postprocessor.num_top_queries),
            )
            query, label, score, selected_box = manual_native_post(
                outputs["pred_logits"], outputs["pred_boxes"], sizes_wh,
                int(postprocessor.num_classes), int(postprocessor.num_top_queries),
            )

            result_by_image = {}

            for local_index, (image_id, gt) in enumerate(image_ground_truth):
                boundary_audit = dict(batch_boundary_audits[local_index])
                boundary_audit.pop("batch_index")
                boundary_audit["image_id"] = image_id
                enforce_topk_boundary_policy(
                    boundary_audit, split=split, image_id=image_id
                )
                boundary_audits.append(boundary_audit)
                if not torch.equal(native[local_index]["labels"], label[local_index]):
                    raise RuntimeError(f"{split} image {image_id}: native labels mismatch")
                score_error = float(
                    (native[local_index]["scores"] - score[local_index]).abs().max().item()
                )
                box_error = float(
                    (native[local_index]["boxes"] - selected_box[local_index]).abs().max().item()
                )
                maximum_score_error = max(maximum_score_error, score_error)
                maximum_box_error = max(maximum_box_error, box_error)
                if score_error != 0.0 or box_error != 0.0:
                    raise RuntimeError(
                        f"{split} image {image_id}: native replay is not bit-exact "
                        f"(score={score_error}, box={box_error})"
                    )
                logits = outputs["pred_logits"][local_index].detach().cpu().float()
                boxes = outputs["pred_boxes"][local_index].detach().cpu().float()
                hidden = outputs["hilor_decoder_features"][local_index].detach().cpu().float()
                query_counts.add(int(logits.shape[0]))
                records.append(
                    {
                        "image_id": image_id,
                        "pred_logits": logits,
                        "pred_boxes": boxes,
                        "hL": hidden,
                        "orig_size": torch.tensor(
                            [gt["height"], gt["width"]], dtype=torch.int64
                        ),
                        "gt_boxes": gt["boxes"],
                        "gt_labels": gt["labels"],
                        "gt_area": gt["area"],
                        "gt_iscrowd": gt["iscrowd"],
                        "target_boxes": gt["boxes"],
                        "target_labels": gt["labels"],
                        "target_area": gt["area"],
                        "target_iscrowd": gt["iscrowd"],
                        "post_query_idx": query[local_index].cpu().long(),
                        "post_labels": label[local_index].cpu().long(),
                        "post_scores": score[local_index].cpu().float(),
                        "post_boxes": selected_box[local_index].cpu().float(),
                        "topk_boundary_audit": boundary_audit,
                    }
                )
                result_by_image[image_id] = {
                    key: value.detach().cpu() for key, value in native[local_index].items()
                }
            evaluator.update(result_by_image)
            if batch_index % 20 == 0:
                print(f"capture {split}: {batch_index}/{len(loader)}")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    if len(query_counts) != 1:
        raise RuntimeError(f"non-constant query count: {sorted(query_counts)}")
    if len(records) != split_binding["image_count"]:
        raise RuntimeError(f"{split} capture did not cover every bound image exactly once")
    captured_ids = [int(record["image_id"]) for record in records]
    if len(captured_ids) != len(set(captured_ids)):
        raise RuntimeError(f"{split} capture contains duplicate image ids")
    if sha256_lines(sorted(captured_ids)) != split_binding["image_id_set_sha256"]:
        raise RuntimeError(f"{split} captured image-id set mismatch")
    capture = {
        "format_version": 4,
        "schema": "rtdetr-hilor-frozen-query-class-capture-v1",
        "method": "frozen_native_detector",
        "host": "RT-DETR-R18",
        "split": split,
        "test_accessed": False,
        "selection_eligible": split in {"train", "val"},
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_state_source": checkpoint_state_source,
        "training_run": training_run_evidence,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "resolved_config_sha256": config_binding["resolved_config_sha256"],
        "config_include_closure_sha256": config_binding["include_closure_sha256"],
        "config_include_closure": config_binding["include_closure"],
        "annotation": str(annotation_path),
        "annotation_sha256": split_binding["annotation_sha256"],
        "images_dir": str(image_dir),
        "image_id_set_sha256": split_binding["image_id_set_sha256"],
        "file_name_set_sha256": split_binding["file_name_set_sha256"],
        "image_tree_sha256": split_binding["image_tree_sha256"],
        "image_count": split_binding["image_count"],
        "data_split_lock_sha256": data_lock_sha,
        "postprocessor_num_classes": int(postprocessor.num_classes),
        "postprocessor_num_top_queries": int(postprocessor.num_top_queries),
        "records": records,
        "summary": {
            "image_count": len(records),
            "query_counts": sorted(query_counts),
            "resolved_config_sha256": config_binding["resolved_config_sha256"],
            "config_include_closure_sha256": config_binding[
                "include_closure_sha256"
            ],
            "image_tree_sha256": split_binding["image_tree_sha256"],
            "native_manual_score_max_error": maximum_score_error,
            "native_manual_box_max_error": maximum_box_error,
            "topk_boundary_tie_policy": TOPK_BOUNDARY_TIE_POLICY,
            "topk_boundary_audit_count": len(boundary_audits),
            "topk_boundary_audit_sha256": canonical_json_sha256(boundary_audits),
            "topk_boundary_tie_count": sum(
                int(item["boundary_tie"]) for item in boundary_audits
            ),
            "topk_boundary_min_margin": min(
                (
                    float(item["boundary_margin"])
                    for item in boundary_audits
                    if item["boundary_margin"] is not None
                ),
                default=None,
            ),
        },
    }
    path = output_dir / f"{prefix}_{split}_capture.pt"
    eval_path = output_dir / f"{prefix}_{split}_baseline_eval.pth"
    atomic_torch_save(capture, path)
    atomic_torch_save(evaluator.coco_eval["bbox"].eval, eval_path)
    return path, eval_path, capture["summary"]
