"""Evaluate the value-residual grid on saved Train/Val detector outputs."""

import argparse
from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

TAUS = (0.0, 0.1, 0.2, 0.3, 0.4)
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)
HEAD_SEEDS = (0, 1, 2)


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    study = Path(__file__).resolve().parent
    parser.add_argument("--release", type=Path, default=study.parents[1])
    parser.add_argument("--study", type=Path, default=study)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refit-heads", action="store_true",
                        help="Refit all three value heads from the saved Train capture")
    args = parser.parse_args()
    release = args.release.resolve(strict=True)
    study = args.study.resolve(strict=True)
    inputs = study / "inputs"
    output = args.output.resolve()
    if output == release or release in output.parents or output.exists() or output in study.parents:
        raise RuntimeError("Use a new output directory outside the release")
    required = [study / "settings.json", inputs / "mapping.json", inputs / "val_capture.pt"]
    if args.refit_heads:
        required.append(inputs / "train_capture.pt")
    else:
        required += [study / f"heads/seed{seed}.pt" for seed in HEAD_SEEDS]
        required.append(study / "heads/training.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing inputs:\n" + "\n".join(missing))
    torch.set_num_threads(1)
    source = release / "main_host/source/FINAL_CODE"
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(release))
    from engine.deim.competition_mapping import apply_qccr, compute_competition_risk
    from engine.deim.candidate_value import (
        UMR, UMRHead, box_cxcywh_to_xyxy, build_image_features,
    )
    from coco_test_evaluation import evaluator_classes, evaluate_bbox, named_metrics

    pipeline = module_from(release / "reproduce_pipeline.py", "_tau_beta_pipeline")
    trainer = module_from(
        source / "tools/calibration/common/train_umr_select_val.py", "_tau_beta_trainer"
    )
    trainer.NUM_CLASSES = 5
    settings = json.loads((study / "settings.json").read_text())
    if settings["detector_delta"] != 0.02 or settings["detector_seed"] != 0:
        raise RuntimeError("This grid requires the delta=0.02 detector seed 0")
    if settings["tau_grid"] != list(TAUS) or settings["beta_grid"] != list(BETAS):
        raise RuntimeError("Unexpected sensitivity grid")
    mapping = json.loads((inputs / "mapping.json").read_text())
    parameters = mapping["parameters"]
    if parameters != settings["mapping_parameters"]:
        raise RuntimeError("Mapping parameters differ from study settings")
    captures, annotations = {}, {}
    for split in (("train", "val") if args.refit_heads else ("val",)):
        path = inputs / f"{split}_capture.pt"
        capture = torch.load(path, map_location="cpu", weights_only=False)
        annotation = release / f"main_host/datasets/CDW1630/annotations/instances_{split}2017.json"
        expected = {int(row["id"]) for row in json.loads(annotation.read_text())["images"]}
        ids = [int(row["image_id"]) for row in capture["records"]]
        if capture["split"] != split or set(ids) != expected or len(ids) != len(expected):
            raise RuntimeError(f"Incomplete or incorrect {split} capture")
        if capture["checkpoint_sha256"] != mapping["source_checkpoint_sha256"]:
            raise RuntimeError(f"Wrong detector in {split} capture")
        for row in capture["records"]:
            for key, shape in (("pred_logits", (300, 5)), ("pred_boxes", (300, 4)), ("hL", (300, 256))):
                if tuple(row[key].shape) != shape or row[key].dtype != torch.float32:
                    raise RuntimeError(f"Unexpected {key} in {split}")
        captures[split], annotations[split] = capture, annotation

    output.mkdir(parents=True)
    heads_dir = output / "heads"
    heads_dir.mkdir()
    write_json(output / "settings.json", settings)
    if args.refit_heads:
        histories = {}
        print("Building Train features with the existing fitting implementation", flush=True)
        train_data, train_ids = pipeline.main_train_data(trainer, captures["train"], parameters)
        for seed in HEAD_SEEDS:
            model, history, epoch, loss, nfit, nhold = trainer.train_one(
                UMRHead, train_data, train_ids, seed, 4
            )
            torch.save(model.state_dict(), heads_dir / f"seed{seed}.pt")
            fit_idx, holdout_idx = trainer.make_split(train_ids, seed)
            histories[str(seed)] = {
                "best_epoch": epoch, "best_holdout_qfl": loss, "history": history,
                "fit_image_count": nfit, "holdout_image_count": nhold,
                "fit_image_ids": [train_ids[int(i)] for i in fit_idx],
                "holdout_image_ids": [train_ids[int(i)] for i in holdout_idx],
            }
            write_json(heads_dir / "training.json", histories)
            print(f"Head seed {seed} ready, best epoch={epoch}", flush=True)
        del train_data
        del captures["train"]
    else:
        for seed in HEAD_SEEDS:
            shutil.copy2(study / f"heads/seed{seed}.pt", heads_dir / f"seed{seed}.pt")
        histories = json.loads((study / "heads/training.json").read_text())
        write_json(heads_dir / "training.json", histories)

    prepared = []
    for record in captures["val"]["records"]:
        raw, boxes = record["pred_logits"], record["pred_boxes"]
        height, width = [int(x) for x in record["orig_size"].tolist()]
        absolute = box_cxcywh_to_xyxy(boxes) * torch.tensor([width, height, width, height])
        risk = compute_competition_risk(raw.unsqueeze(0), boxes.unsqueeze(0))[0]
        mapped = apply_qccr(raw, risk, **parameters)
        indices, features, classes, hidden = build_image_features(
            raw, boxes, absolute, record["hL"], mapped, mapped.sigmoid(), risk, K=300
        )
        prepared.append({
            "image_id": int(record["image_id"]), "raw": raw, "boxes": boxes,
            "absolute": absolute, "indices": indices, "features": features,
            "classes": classes, "hidden": hidden, "hL": record["hL"],
        })
    COCO, _ = evaluator_classes()
    gt = COCO(str(annotations["val"]))
    image_ids = sorted(row["image_id"] for row in prepared)

    def final_logits(record, values, tau, beta):
        if values is None or beta == 0:
            updated = record["raw"]
        else:
            bonus = torch.zeros(1500, dtype=torch.float32)
            bonus[record["indices"]] = torch.relu(values - tau)
            updated = record["raw"] + beta * bonus.reshape(300, 5)
        risk = compute_competition_risk(updated.unsqueeze(0), record["boxes"].unsqueeze(0))[0]
        return apply_qccr(updated, risk, **parameters)

    def evaluate(values, tau, beta):
        detections = []
        for i, record in enumerate(prepared):
            logits = final_logits(record, None if values is None else values[i], tau, beta)
            scores, indices = torch.topk(logits.sigmoid().flatten(), k=300)
            boxes = record["absolute"][indices // 5].clone()
            boxes[:, 2:] -= boxes[:, :2]
            for label, score, box in zip((indices % 5).tolist(), scores.tolist(), boxes.tolist()):
                detections.append({
                    "image_id": record["image_id"], "category_id": label,
                    "score": score, "bbox": box,
                })
        with redirect_stdout(io.StringIO()):
            evaluation = evaluate_bbox(gt, detections, image_ids)
        return named_metrics(evaluation.eval)

    baseline = evaluate(None, 0, 0)
    write_json(output / "baseline.json", {"method": "mapping_only", "metrics": baseline})
    print(f"Mapping-only Val AP={100 * baseline['AP']:.6f}", flush=True)
    rows = []
    with torch.inference_mode():
        for seed in HEAD_SEEDS:
            model = UMRHead(num_classes=5)
            model.load_state_dict(torch.load(heads_dir / f"seed{seed}.pt", weights_only=True))
            model.eval()
            values = [
                model(r["hidden"].unsqueeze(0), r["classes"].unsqueeze(0), r["features"].unsqueeze(0)).sigmoid()[0]
                for r in prepared
            ]
            for tau, beta in ((0.0, 2.0), (0.3, 1.0), (0.4, 0.0)):
                direct = UMR(num_classes=5, tau=tau, beta=beta,
                             checkpoint=heads_dir / f"seed{seed}.pt", **parameters).eval()
                for i in (0, len(prepared) - 1):
                    r = prepared[i]
                    expected = direct(r["raw"][None], r["boxes"][None], r["absolute"][None], r["hL"][None])[0]
                    actual = final_logits(r, values[i], tau, beta)
                    if not torch.equal(expected, actual):
                        raise RuntimeError("Cached grid arithmetic differs from the formal value module")
            for tau in TAUS:
                for beta in BETAS:
                    metrics = baseline if beta == 0 else evaluate(values, tau, beta)
                    rows.append({"head_seed": seed, "tau": tau, "beta": beta, **metrics})
                    write_csv(output / "per_head.csv", rows)
                    print(f"seed={seed} tau={tau:g} beta={beta:g} Val AP={100 * metrics['AP']:.6f}", flush=True)

    means = []
    for tau in TAUS:
        for beta in BETAS:
            group = [r for r in rows if r["tau"] == tau and r["beta"] == beta]
            row = {"tau": tau, "beta": beta}
            for metric in baseline:
                values = [r[metric] for r in group]
                row[metric] = float(np.mean(values))
                row[metric + "_sd"] = float(np.std(values, ddof=1))
                row["delta_" + metric + "_pp"] = 100 * (row[metric] - baseline[metric])
            row["score"] = trainer.score(row, baseline)
            row["guarded"], failures = trainer.guards(row, baseline, group)
            row["failed_guards"] = ";".join(failures)
            means.append(row)
    write_csv(output / "mean.csv", means)
    best_ap = max(r["AP"] for r in means)
    best = [r for r in means if abs(r["AP"] - best_ap) < 1e-12]
    current = next(r for r in means if r["tau"] == 0 and r["beta"] == 2)
    eligible = [r for r in means if r["guarded"]]
    write_json(output / "summary.json", {
        "complete": True, "metric_unit": "fraction; delta columns are percentage points",
        "combinations": len(means), "per_head_results": len(rows),
        "baseline": baseline, "current": current, "best_by_mean_ap": best,
        "best_guarded_by_mean_ap": max(eligible, key=lambda r: r["AP"]) if eligible else None,
        "current_is_best": abs(current["AP"] - best_ap) < 1e-12,
        "best_minus_current_ap_pp": 100 * (best_ap - current["AP"]),
        "formal_parameters_changed": False,
    })
    print("Complete: 25 settings, three head-fitting seeds. Formal settings unchanged.", flush=True)
    print(json.dumps({"best": [(r["tau"], r["beta"], r["AP"] * 100) for r in best],
                      "current_ap": current["AP"] * 100}, indent=2), flush=True)


if __name__ == "__main__":
    main()
