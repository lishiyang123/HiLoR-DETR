"""Read-only replay of the formal QCCR and QCCR--UMR implementations.

The nearest-neighbour study must compare against the exact frozen HiLoR
post-processing path. This adapter loads only the competition-mapping and
candidate-value files from an explicitly supplied formal source tree. It
does not import or modify the detector package as a whole.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any, Mapping

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _formal_deim_modules(source_root: Path):
    """Load the two formal modules without executing ``engine.__init__``."""

    source_root = Path(source_root).resolve()
    deim_dir = source_root / "engine" / "deim"
    qccr_path = deim_dir / "competition_mapping.py"
    umr_path = deim_dir / "candidate_value.py"
    for path in (qccr_path, umr_path):
        if not path.is_file():
            raise FileNotFoundError(f"formal implementation is missing: {path}")

    namespace = "_cdw_formal_deim_" + hashlib.sha256(
        str(deim_dir).encode("utf-8")
    ).hexdigest()[:16]
    qccr_name = namespace + ".qccr"
    umr_name = namespace + ".umr"
    if qccr_name in sys.modules and umr_name in sys.modules:
        return sys.modules[qccr_name], sys.modules[umr_name]

    package = types.ModuleType(namespace)
    package.__path__ = [str(deim_dir)]
    package.__package__ = namespace
    sys.modules[namespace] = package

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load formal module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    qccr_module = load(qccr_name, qccr_path)
    umr_module = load(umr_name, umr_path)
    return qccr_module, umr_module


def _qccr_parameters(payload: Mapping[str, Any]) -> dict[str, float]:
    raw = payload.get("parameters", payload)
    required = ("beta0", "beta_positive", "beta_negative", "duplicate_penalty")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"QCCR artifact is missing parameters: {missing}")
    parameters = {name: float(raw[name]) for name in required}
    if parameters["beta_positive"] <= 0 or parameters["beta_negative"] <= 0:
        raise ValueError("QCCR beta parameters must be positive")
    if parameters["duplicate_penalty"] < 0:
        raise ValueError("QCCR duplicate penalty must be non-negative")
    return parameters


def _record_tensor(value, *, dtype, device):
    import torch

    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _absolute_xyxy(record: Mapping[str, Any], boxes):
    """Match the formal postprocessor's FP32 cxcywh-to-pixel conversion."""

    import torch

    cx, cy, width, height = boxes.unbind(dim=-1)
    xyxy = torch.stack(
        (cx - 0.5 * width, cy - 0.5 * height,
         cx + 0.5 * width, cy + 0.5 * height),
        dim=-1,
    )
    size = _record_tensor(record["orig_size"], dtype=torch.float32, device=boxes.device)
    image_height, image_width = size.unbind(dim=-1)
    scale = torch.stack((image_width, image_height, image_width, image_height))
    return xyxy * scale


@dataclass(frozen=True)
class FormalAssetBinding:
    qccr_sha256: str
    umr_checkpoint_sha256: str | None
    umr_metadata_sha256: str | None
    source_checkpoint_sha256: str | None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "qccr_sha256": self.qccr_sha256,
            "umr_checkpoint_sha256": self.umr_checkpoint_sha256,
            "umr_metadata_sha256": self.umr_metadata_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
        }


class FormalPostprocessorReplay:
    """Reusable exact QCCR or QCCR--UMR capture replayer."""

    def __init__(
        self,
        *,
        method: str,
        formal_source: Path,
        qccr_params: Path,
        capture_checkpoint_sha256: str | None,
        umr_checkpoint: Path | None = None,
        umr_metadata: Path | None = None,
        device: str = "cpu",
    ) -> None:
        import torch

        if method not in {"qccr", "qccr_umr"}:
            raise ValueError("method must be 'qccr' or 'qccr_umr'")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA replay was requested but unavailable: {device}")

        self.method = method
        self.device = torch.device(device)
        self.formal_source = Path(formal_source).resolve()
        self.qccr_path = Path(qccr_params).resolve()
        self.qccr_payload = _load_json(self.qccr_path)
        self.parameters = _qccr_parameters(self.qccr_payload)
        qccr_module, umr_module = _formal_deim_modules(self.formal_source)

        artifact_checkpoint = self.qccr_payload.get("source_checkpoint_sha256")
        if (
            capture_checkpoint_sha256
            and artifact_checkpoint
            and capture_checkpoint_sha256 != artifact_checkpoint
        ):
            raise RuntimeError(
                "QCCR artifact and frozen capture belong to different checkpoints: "
                f"{artifact_checkpoint} != {capture_checkpoint_sha256}"
            )

        self.qccr = qccr_module.QueryCompetitionConditionalRecalibration(
            **self.parameters
        ).to(self.device).eval()
        self.umr = None
        umr_checkpoint_sha = None
        umr_metadata_sha = None

        if method == "qccr_umr":
            if umr_checkpoint is None or umr_metadata is None:
                raise RuntimeError(
                    "--umr-checkpoint and --umr-metadata are required for qccr_umr"
                )
            umr_checkpoint = Path(umr_checkpoint).resolve()
            umr_metadata = Path(umr_metadata).resolve()
            metadata = _load_json(umr_metadata)
            umr_checkpoint_sha = _sha256(umr_checkpoint)
            umr_metadata_sha = _sha256(umr_metadata)
            if metadata.get("weight_sha256") not in (None, umr_checkpoint_sha):
                raise RuntimeError("UMR metadata/weight SHA-256 mismatch")
            if metadata.get("qccr_sha256") not in (None, _sha256(self.qccr_path)):
                raise RuntimeError("UMR metadata/QCCR SHA-256 mismatch")
            metadata_checkpoint = metadata.get("source_checkpoint_sha256")
            if (
                capture_checkpoint_sha256
                and metadata_checkpoint
                and capture_checkpoint_sha256 != metadata_checkpoint
            ):
                raise RuntimeError(
                    "UMR metadata and frozen capture belong to different checkpoints"
                )
            fusion = metadata.get("fusion", {})
            self.umr = umr_module.UMR(
                num_classes=int(metadata.get("num_classes", 5)),
                num_queries=300,
                topk=int(fusion.get("topk", metadata.get("umr_topk", 300))),
                tau=float(fusion.get("tau", metadata.get("umr_tau", 0.0))),
                beta=float(fusion.get("beta", metadata.get("umr_beta", 2.0))),
                checkpoint=umr_checkpoint,
                **self.parameters,
            ).to(self.device).eval()

        self.binding = FormalAssetBinding(
            qccr_sha256=_sha256(self.qccr_path),
            umr_checkpoint_sha256=umr_checkpoint_sha,
            umr_metadata_sha256=umr_metadata_sha,
            source_checkpoint_sha256=artifact_checkpoint,
        )

    def synchronize(self) -> None:
        import torch

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def scores(self, record: Mapping[str, Any]) -> np.ndarray:
        import torch

        logits = _record_tensor(
            record["pred_logits"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        boxes = _record_tensor(
            record["pred_boxes"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            if self.umr is None:
                final_logits = self.qccr(logits, boxes)
            else:
                if "hL" not in record:
                    raise ValueError("QCCR--UMR replay requires hL in every record")
                boxes_abs = _absolute_xyxy(record, boxes[0]).unsqueeze(0)
                h_l = _record_tensor(
                    record["hL"], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                final_logits = self.umr(logits, boxes, boxes_abs, h_l)
            scores = torch.sigmoid(final_logits).reshape(-1)
        return scores.detach().cpu().numpy().astype(np.float64, copy=False)

