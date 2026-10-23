#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import json
import os
import platform
import socket
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from common import fps_cv, percentile, require, sha256, write_json


def enforce_thread_protocol() -> dict[str, int]:
    """Reassert the protocol after third-party source imports.

    Some detector repositories rewrite OMP_NUM_THREADS while importing their
    utility modules.  The launcher already supplies the frozen values, but the
    timing process must reassert them immediately before profiling/timing and
    bind the effective PyTorch thread-pool sizes as evidence.
    """
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    require(torch.get_num_threads() == 1,
            "PyTorch intra-op thread protocol could not be enforced")
    require(torch.get_num_interop_threads() == 1,
            "PyTorch inter-op thread protocol could not be enforced")
    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def load_test_index(annotation: Path, image_root: Path,
                    expected_count: int = 237) -> list[dict[str, Any]]:
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    rows = []
    for image in sorted(payload["images"], key=lambda row: int(row["id"])):
        path = image_root / image["file_name"]
        require(path.is_file(), f"Missing indexed image: {path}")
        rows.append({
            "image_id": int(image["id"]),
            "file_name": str(image["file_name"]),
            "path": path,
            "height": int(image["height"]),
            "width": int(image["width"]),
        })
    require(len(rows) == expected_count,
            f"Expected {expected_count} images, found {len(rows)}")
    return rows


def prepare_image(row: dict[str, Any]) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(row["path"]) as handle:
        image = handle.convert("RGB")
        require(image.size == (row["width"], row["height"]),
                f"Annotation/image size mismatch: {row['file_name']}")
        require(image.size == (640, 640),
                f"Protocol requires already-resized 640x640 images: {row['file_name']}")
        array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)
    return tensor.unsqueeze(0), (row["height"], row["width"])


def profile_flops(model) -> dict[str, Any]:
    import importlib.metadata
    from calflops import calculate_flops

    flops, macs, profiler_parameters = calculate_flops(
        model=model,
        input_shape=(1, 3, 640, 640),
        output_as_string=False,
        output_precision=6,
        print_detailed=False,
    )
    try:
        version = importlib.metadata.version("calflops")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "scope": "accepted deployed detector forward only",
        "input_shape": [1, 3, 640, 640],
        "mac_to_flop_factor": 2,
        "profiler": "calflops",
        "profiler_version": version,
        "forward_MACs": float(macs),
        "forward_FLOPs": float(flops),
        "forward_GMACs": float(macs) / 1.0e9,
        "forward_GFLOPs": float(flops) / 1.0e9,
        "profiler_parameters": int(profiler_parameters),
        "postprocessor_included": False,
    }


def benchmark(
    *,
    name: str,
    detector,
    postprocessor,
    forward_fn: Callable[[torch.Tensor], Any],
    postprocess_fn: Callable[[Any, torch.Tensor, tuple[int, int]], Any],
    count_outputs_fn: Callable[[Any], int],
    annotation: Path,
    image_root: Path,
    protocol: Path,
    checkpoint: Path,
    checkpoint_role: str,
    source_record: dict[str, Any],
    config_record: dict[str, Any] | None,
    predeploy_parameters: int,
    postdeploy_parameters: int,
    postprocessor_parameters: int,
    fitted_scalars: int,
    deployment: dict[str, Any],
    output: Path,
    mode: str,
    warmup: int,
    repeats: int,
) -> None:
    require(not output.exists(), f"Output already exists: {output}")
    effective_threads = enforce_thread_protocol()
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require("4090" in torch.cuda.get_device_name(0), "Protocol requires RTX 4090")
    rows = load_test_index(annotation, image_root)
    require(0 < warmup <= len(rows), "Invalid warmup image count")
    require(repeats >= 1, "Invalid repeat count")

    flops = profile_flops(detector.eval())
    profiler_parameters = int(flops["profiler_parameters"])
    require(profiler_parameters >= 0 and postdeploy_parameters > 0,
            "Invalid profiler or deployed parameter count")
    # calflops labels this value as "Total Training Params" and can omit
    # frozen/non-trainable registered parameters.  It is therefore retained as
    # a diagnostic only.  The paper-facing Params value is the complete
    # registered parameter count of the exact deployed detector object below.
    flops["parameter_diagnostic"] = {
        "role": "diagnostic only; not the reported Params value",
        "profiler_parameters": profiler_parameters,
        "deployed_registered_parameters": int(postdeploy_parameters),
        "profiler_minus_deployed": profiler_parameters - int(postdeploy_parameters),
        "exact_agreement": profiler_parameters == int(postdeploy_parameters),
        "formal_parameter_source": "complete registered parameters of the timed deployed detector",
    }
    detector = detector.cuda().eval()
    if postprocessor is not None:
        postprocessor = postprocessor.cuda().eval()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def one(row: dict[str, Any], timed: bool) -> tuple[float, float, float, int]:
        cpu, original_hw = prepare_image(row)
        image = cpu.cuda(non_blocking=False)
        original_sizes = torch.tensor([list(original_hw)], device="cuda", dtype=torch.int64)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        middle = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            raw = forward_fn(image)
        middle.record()
        with torch.cuda.amp.autocast(enabled=False):
            result = postprocess_fn(raw, original_sizes, original_hw)
        end.record()
        end.synchronize()
        if not timed:
            return 0.0, 0.0, 0.0, count_outputs_fn(result)
        forward_ms = float(start.elapsed_time(middle))
        post_ms = float(middle.elapsed_time(end))
        total_ms = float(start.elapsed_time(end))
        require(abs(total_ms - (forward_ms + post_ms)) <= max(0.05, total_ms * 0.01),
                "CUDA-event timing partition is inconsistent")
        return forward_ms, post_ms, total_ms, count_outputs_fn(result)

    with torch.inference_mode():
        for row in rows[:warmup]:
            one(row, False)
        torch.cuda.synchronize()
        samples: list[dict[str, Any]] = []
        repeat_rows = []
        for repeat in range(repeats):
            repeat_total = []
            for row in rows:
                forward_ms, post_ms, total_ms, outputs = one(row, True)
                repeat_total.append(total_ms)
                samples.append({
                    "repeat": repeat + 1,
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "forward_ms": forward_ms,
                    "postprocess_ms": post_ms,
                    "total_ms": total_ms,
                    "output_detections": outputs,
                })
            mean = statistics.mean(repeat_total)
            repeat_rows.append({
                "repeat": repeat + 1,
                "images": len(repeat_total),
                "mean_total_ms": mean,
                "fps": 1000.0 / mean,
            })
    torch.cuda.synchronize()

    forward_values = [row["forward_ms"] for row in samples]
    post_values = [row["postprocess_ms"] for row in samples]
    total_values = [row["total_ms"] for row in samples]
    coefficient = fps_cv([row["mean_total_ms"] for row in repeat_rows])
    total_parameters = postdeploy_parameters + postprocessor_parameters + fitted_scalars
    result = {
        "schema": "cdw1630-nine-model-real-test-efficiency-result-v2",
        "status": "PASS" if coefficient <= 5.0 else "UNSTABLE",
        "mode": mode,
        "model": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(protocol), "sha256": sha256(protocol)},
        "dataset": {
            "split": "CDW1630 Test",
            "annotation": str(annotation),
            "annotation_sha256": sha256(annotation),
            "image_root": str(image_root),
            "images": len(rows),
            "image_order": "ascending COCO image id",
            "real_test_images_used_for_timing": True,
            "test_used_for_selection": False,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "role": checkpoint_role,
        },
        "source": source_record,
        "config": config_record,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
            **effective_threads,
        },
        "deployment": deployment,
        "parameters": {
            "scope": "actual timed accepted deployment inference graph",
            "predeploy_detector": predeploy_parameters,
            "deployed_detector": postdeploy_parameters,
            "fusion_delta": postdeploy_parameters - predeploy_parameters,
            "postprocessor_registered_nonduplicated": postprocessor_parameters,
            "fitted_scalars": fitted_scalars,
            "total_inference": total_parameters,
            "total_M": total_parameters / 1.0e6,
        },
        "flops": flops,
        "latency": {
            "scope": "accepted deployment detector forward plus complete native postprocessing",
            "warmup_real_images": warmup,
            "timed_repeats": repeats,
            "images_per_repeat": len(rows),
            "sample_count": len(samples),
            "forward_mean_ms": statistics.mean(forward_values),
            "forward_median_ms": statistics.median(forward_values),
            "postprocess_mean_ms": statistics.mean(post_values),
            "postprocess_median_ms": statistics.median(post_values),
            "total_mean_ms": statistics.mean(total_values),
            "total_median_ms": statistics.median(total_values),
            "total_p90_ms": percentile(total_values, 0.90),
            "total_p95_ms": percentile(total_values, 0.95),
            "fps": 1000.0 / statistics.mean(total_values),
            "repeat_fps_cv_percent": coefficient,
            "repeat_results": repeat_rows,
        },
        "peak_cuda_memory_MiB": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "training_executed": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "EFFICIENCY.json", result)
    with (output / "LATENCY_SAMPLES.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    print(f"{name}: Params={total_parameters/1e6:.6f}M GFLOPs={flops['forward_GFLOPs']:.6f} "
          f"Forward={result['latency']['forward_mean_ms']:.4f}ms "
          f"Post={result['latency']['postprocess_mean_ms']:.4f}ms "
          f"Total={result['latency']['total_mean_ms']:.4f}ms FPS={result['latency']['fps']:.3f} "
          f"CV={coefficient:.3f}%")
    require(result["status"] == "PASS", "Repeat-level FPS CV exceeds 5%")
