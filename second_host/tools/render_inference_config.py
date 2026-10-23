#!/usr/bin/env python3
"""Render a bound QCCR or QCCR–UMR YAML after assets have been frozen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from fit_common import (
        adapter_root, canonical_implementation_lock, require_output_path,
        require_sha,
    )
except ImportError:  # imported as tools.render_inference_config in unit tests
    from tools.fit_common import (
        adapter_root, canonical_implementation_lock, require_output_path,
        require_sha,
    )


SCHEMA = "cdw1630-rtdetr-r18-qccr-umr-inference-binding-v1"


def quoted(value: str) -> str:
    return json.dumps(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--expected-binding-sha", required=True)
    parser.add_argument("--mode", choices=("qccr", "qccr-umr"), required=True)
    parser.add_argument("--detector-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--adapter-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = adapter_root(args.adapter_root)
    _implementation, implementation_sha = canonical_implementation_lock(root)
    binding = require_output_path(
        args.binding, namespace="inference", detector_seed=args.detector_seed,
        adapter_root_path=root, label="inference binding input",
    )
    binding_sha = require_sha(
        binding, args.expected_binding_sha, "inference binding"
    )
    payload = json.loads(binding.read_text(encoding="utf-8"))
    if payload.get("implementation_lock_sha256") != implementation_sha:
        raise RuntimeError("inference binding belongs to another implementation lock")
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("invalid inference binding schema")
    if payload.get("host") != "RT-DETR-R18" or payload.get("dataset") != "CDW1630":
        raise RuntimeError("inference binding belongs to another host or dataset")
    if payload.get("test_accessed_for_fitting_or_selection") is not False:
        raise RuntimeError("binding does not prove Test exclusion")
    if payload.get("mode") != args.mode:
        raise RuntimeError("requested inference mode does not match the binding")
    if payload.get("detector_seed") != args.detector_seed:
        raise RuntimeError("inference binding belongs to another detector seed")
    detector_sha = payload["detector_checkpoint_sha256"]
    output = require_output_path(
        args.output, namespace="generated_config", detector_seed=args.detector_seed,
        adapter_root_path=root, label="generated inference config",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite inference config: {output}")

    base = root / "configs" / "hilor" / "cdw1630_rtdetr_r18_gca.yml"
    include = os.path.relpath(base, output.parent).replace("\\", "/")
    use_umr = args.mode == "qccr-umr"
    binding_relative = binding.relative_to(root).as_posix()
    manifest_spec = {
        "path": binding_relative,
        "sha256": binding_sha,
        "path_base": "adapter_root",
    }
    content = f'''# Generated only after RT-DETR-specific fitting assets were frozen.
__include__: [{quoted(include)}]

postprocessor: HiLoRRTDETRPostProcessor

RTDETRTransformer:
  hilor_return_decoder_features: {str(use_umr)}

HiLoRRTDETRPostProcessor:
  num_classes: 5
  use_focal_loss: True
  num_top_queries: 300
  remap_mscoco_category: False
  qccr_enabled: True
  umr_enabled: {str(use_umr)}
  num_queries: 300
  artifact_manifest: {json.dumps(manifest_spec, sort_keys=True)}
  expected_detector_checkpoint_sha256: {quoted(detector_sha)}
  allow_unbound_artifacts: False
'''
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output)
    print("BOUND INFERENCE CONFIG: PASS")
    print("Mode:", args.mode)
    print("Detector SHA:", detector_sha)
    print("Binding SHA:", binding_sha)


if __name__ == "__main__":
    main()
