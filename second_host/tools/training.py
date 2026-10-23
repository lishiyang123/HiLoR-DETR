"""RT training loop with Val-only checkpoint selection.

Shared by fresh Native and GCA runs; optimizer, EMA, scheduler and epoch order
are the recorded training implementation. Output namespaces are run-specific.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

BEST_SCHEMA = "cdw1630-rtdetr-r18-hilor-best-val-selection-v1"
BEST_VAL_SELECTION_RULE = "maximum Val COCO bbox AP; earliest exact tie"
SOURCE_RELATIVE = Path("source/RT-DETR_HiLoR/rtdetr_pytorch")


def finalize_training_run(root, output_dir, run_plan, selected, elapsed):
    if sha256_file(output_dir / "best_val.pth") != selected["checkpoint_sha256"]:
        raise TrainingGuardError("Selected checkpoint changed before completion")
    completed = dict(selected, state="complete", training_seconds=float(elapsed))
    _atomic_json(output_dir / "BEST_VAL.json", completed)
    _atomic_json(output_dir / "RUN_COMPLETED.json", dict(
        state="complete", seed=run_plan["seed"], test_used=False,
        best_val_checkpoint_sha256=selected["checkpoint_sha256"]))
    return completed


class TrainingGuardError(RuntimeError):
    """Raised when a formal-training invariant is not satisfied."""


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingGuardError(f"required JSON file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingGuardError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingGuardError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    incoming = path.with_name(path.name + ".incoming")
    incoming.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(incoming, path)


@dataclass
class BestValTracker:
    """Val-only maximum tracker with deterministic earliest-tie semantics."""

    ap: float = float("-inf")
    epoch: int = -1

    def consider(self, epoch: int, ap: float) -> bool:
        epoch_value = int(epoch)
        ap_value = float(ap)
        if epoch_value < 0:
            raise TrainingGuardError("validation epoch must be non-negative")
        if not math.isfinite(ap_value):
            raise TrainingGuardError(f"Val COCO bbox AP must be finite, got {ap_value}")
        if ap_value > self.ap:
            self.ap = ap_value
            self.epoch = epoch_value
            return True
        return False


def _bbox_metrics(test_stats: Mapping[str, Any]) -> tuple[float, float, float]:
    values = test_stats.get("coco_eval_bbox")
    if values is None:
        raise TrainingGuardError("validation did not return coco_eval_bbox")
    try:
        if len(values) < 3:
            raise TrainingGuardError("validation did not return AP/AP50/AP75")
        metrics = tuple(float(values[index]) for index in range(3))
    except (TypeError, ValueError) as exc:
        raise TrainingGuardError("invalid COCO bbox validation metrics") from exc
    if not all(math.isfinite(value) for value in metrics):
        raise TrainingGuardError(f"COCO bbox validation metrics must be finite: {metrics}")
    return metrics


def install_best_val_training(source_root: Path, run_plan: Mapping[str, Any]) -> None:
    """Install the audited Val-only fit loop into the pinned host process."""

    sys.path.insert(0, str(source_root))
    import torch
    from src.data import get_coco_api_from_dataset
    from src.misc import dist
    from src.solver.det_engine import evaluate, train_one_epoch
    from src.solver.det_solver import DetSolver

    adapter_root = source_root.resolve()
    for _part in SOURCE_RELATIVE.parts:
        adapter_root = adapter_root.parent
    if (adapter_root / SOURCE_RELATIVE).resolve() != source_root.resolve():
        raise TrainingGuardError("cannot derive adapter root from formal source root")

    plan_binding = {
        "seed": run_plan["seed"],
        "config_sha256": run_plan["config"]["sha256"],
        "coco_initialization_sha256": run_plan["coco_initialization"]["sha256"],
        "protocol_lock_sha256": run_plan["bindings"]["protocol_lock_sha256"],
        "implementation_lock_sha256": run_plan["bindings"][
            "implementation_lock_sha256"
        ],
        "training_run_manifest": "TRAIN_RUN_MANIFEST.json",
    }

    def fit_with_audited_best_val(self) -> None:
        print("Start RT-DETR training / Val AP checkpoint selection")
        self.train()
        args = self.cfg
        n_parameters = sum(parameter.numel() for parameter in self.model.parameters()
                           if parameter.requires_grad)
        print("number of params:", n_parameters)
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        tracker = BestValTracker()
        selected: Optional[dict[str, Any]] = None
        if isinstance(run_plan.get("resume"), Mapping):
            selected = read_json(self.output_dir / "BEST_VAL_RUNNING.json")
            tracker.ap = float(selected["val_coco_bbox_ap"])
            tracker.epoch = int(selected["epoch_zero_based"])
            selected = {**selected, **plan_binding}
            print(
                "Resuming best-Val tracker:",
                {"epoch": tracker.epoch, "coco_eval_bbox": tracker.ap},
            )
        start_time = time.time()

        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                args.clip_max_norm,
                print_freq=args.log_step,
                ema=self.ema,
                scaler=self.scaler,
            )
            self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / "checkpoint.pth"]
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            val_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                base_ds,
                self.device,
                self.output_dir,
            )
            val_ap, val_ap50, val_ap75 = _bbox_metrics(val_stats)
            if tracker.consider(epoch, val_ap) and self.output_dir and dist.is_main_process():
                best_path = self.output_dir / "best_val.pth"
                incoming = best_path.with_name(best_path.name + ".incoming")
                torch.save(self.state_dict(epoch), incoming)
                os.replace(incoming, best_path)
                selected = {
                    "schema": BEST_SCHEMA,
                    "state": "running",
                    "selection_split": "Val",
                    "test_used": False,
                    "selection_rule": BEST_VAL_SELECTION_RULE,
                    "epoch_zero_based": epoch,
                    "epoch_one_based": epoch + 1,
                    "val_coco_bbox_ap": val_ap,
                    "val_coco_bbox_ap50": val_ap50,
                    "val_coco_bbox_ap75": val_ap75,
                    "checkpoint": "best_val.pth",
                    "checkpoint_sha256": sha256_file(best_path),
                    **plan_binding,
                }
                _atomic_json(self.output_dir / "BEST_VAL_RUNNING.json", selected)

            print("best_val_stat:", {"epoch": tracker.epoch, "coco_eval_bbox": tracker.ap})
            log_stats = {
                **{f"train_{key}": value for key, value in train_stats.items()},
                # Preserve the upstream log key for compatibility.  Selection
                # provenance unambiguously records that this is Val.
                **{f"test_{key}": value for key, value in val_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(log_stats) + "\n")
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        elapsed = time.time() - start_time
        if selected is None:
            raise TrainingGuardError("training completed without a selectable Val checkpoint")
        if self.output_dir and dist.is_main_process():
            selected = finalize_training_run(
                adapter_root,
                self.output_dir,
                run_plan,
                selected,
                elapsed,
            )
        print("Training time {}".format(
            str(datetime.timedelta(seconds=int(elapsed)))
        ))

    DetSolver.fit = fit_with_audited_best_val
