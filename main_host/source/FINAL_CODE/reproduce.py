"""Run DEIM/HiLoR main-host training or frozen-checkpoint evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expand(value, release, source=None):
    source = source or Path(__file__).resolve().parent
    if isinstance(value, dict):
        return {k: expand(v, release, source) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, release, source) for v in value]
    if isinstance(value, str):
        return value.replace("${RELEASE_ROOT}", str(release)).replace("${SOURCE_ROOT}", str(source))
    return value


def external_ccvr(binding_path, checkpoint, profile):
    """Validate checkpoint-specific newly fitted inputs without touching paper assets."""
    binding = json.loads(binding_path.read_text())
    if (binding.get("host") != "deim" or binding.get("dataset") != profile["dataset"]
            or binding.get("detector_seed") != profile["seed"]
            or binding.get("checkpoint_sha256") != sha256(checkpoint)):
        raise RuntimeError("New CCVR binding belongs to another detector/dataset/seed")
    if binding.get("test_used_for_fitting_or_selection") is not False:
        raise RuntimeError("New fit binding must certify Train/Val-only fitting")
    paths = {}
    required = ["qccr"] + (["umr"] if profile["method"] in ("full", "deim_ccvr") else [])
    for name in required:
        row = binding[name]
        path = Path(row["path"])
        if not path.is_absolute():
            path = binding_path.parent / path
        if sha256(path) != row["sha256"]:
            raise RuntimeError("New fitted asset SHA mismatch: " + name)
        paths[name] = path.resolve()
    qccr = json.loads(paths["qccr"].read_text())
    if qccr.get("source_checkpoint_sha256") != binding["checkpoint_sha256"]:
        raise RuntimeError("New mapping belongs to another checkpoint")
    expected = {"lambda_reg": 0.001, "head_seed": 0, "positive_weight": 4,
                "tau": 0.0, "beta": 2.0, "topk": 300}
    if binding.get("settings") != expected:
        raise RuntimeError("New fitting must retain the published fixed settings")
    return qccr["parameters"], paths


def metrics(evaluation):
    import numpy as np
    params = evaluation["params"]
    precision, recall = evaluation["precision"], evaluation["recall"]
    result = {}
    def mean_valid(x):
        valid = x[x > -1]
        return float(np.mean(valid)) if valid.size else None
    for name, area, threshold in (
        ("AP", "all", None), ("AP50", "all", .5), ("AP75", "all", .75),
        ("APs", "small", None), ("APm", "medium", None), ("APl", "large", None),
        ("AP80", "all", .8), ("AP85", "all", .85),
        ("AP90", "all", .9), ("AP95", "all", .95),
    ):
        a, m = params.areaRngLbl.index(area), params.maxDets.index(100)
        tensor = precision[:, :, :, a, m]
        if threshold is not None:
            tensor = tensor[np.isclose(params.iouThrs, threshold)]
        result[name] = mean_valid(tensor)
    for name, area, budget in (
        ("AR1", "all", 1), ("AR10", "all", 10), ("AR100", "all", 100),
        ("ARs", "small", 100), ("ARm", "medium", 100), ("ARl", "large", 100),
    ):
        a, m = params.areaRngLbl.index(area), params.maxDets.index(budget)
        result[name] = mean_valid(recall[:, :, a, m])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Training seed; evaluation seed is profile-bound")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a newly trained checkpoint; paper replay remains the default")
    parser.add_argument("--ccvr-binding", type=Path, help="FIT_BINDING.json produced for this new checkpoint")
    parser.add_argument("--compare-reference", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Check frozen inputs and paths without writing outputs or using a GPU")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    release, output = args.release_root.resolve(), args.output.resolve()
    if release == output or release in output.parents:
        raise RuntimeError("Use a new output directory outside frozen release assets")
    if output.exists():
        raise RuntimeError("Output exists; use a new directory")
    profiles = json.loads((source / "PROFILES.json").read_text())
    profile = profiles[args.profile]
    training = profile["split"] == "train"
    seed = args.seed if training else profile["seed"]
    weight_key = "pretrained" if training else "checkpoint"
    if training and (args.checkpoint or args.ccvr_binding):
        parser.error("New checkpoint/CCVR overrides are evaluation-only")
    if args.ccvr_binding and not args.checkpoint:
        parser.error("--ccvr-binding requires --checkpoint")
    if args.ccvr_binding and profile["method"] not in ("full", "qccr", "deim_ccvr"):
        parser.error("--ccvr-binding is only used by rescored evaluation")
    weight = args.checkpoint.resolve() if args.checkpoint else release / profile[weight_key]
    weight_sha = sha256(weight)
    if not args.checkpoint and weight_sha != profile[weight_key + "_sha256"]:
        raise RuntimeError("Weight does not match the profile")
    if args.checkpoint and args.compare_reference:
        parser.error("New checkpoint evaluation is separate from exact paper replay")
    if args.checkpoint and profile["method"] in ("full", "qccr", "deim_ccvr") and not args.ccvr_binding:
        parser.error("New rescored evaluation requires its own --ccvr-binding")
    for asset in ([] if args.checkpoint else profile.get("inference_assets", [])):
        if sha256(release / asset["path"]) != asset["sha256"]:
            raise RuntimeError("Frozen inference asset changed: " + asset["path"])
    import yaml
    config = expand(yaml.safe_load((source / profile["config"]).read_text()), release)
    if args.ccvr_binding:
        parameters, paths = external_ccvr(args.ccvr_binding.resolve(), weight, profile)
        post = config["PostProcessor"]
        for name, value in parameters.items():
            post["qccr_" + name] = float(value)
        if "umr" in paths:
            post["umr_checkpoint"] = str(paths["umr"])
    config.update(output_dir=str(output), use_amp=False, seed=seed, device=args.device)
    config["tuning" if training else "resume"] = str(weight)
    config["HGNetv2"]["pretrained"] = False
    for loader_name in ("train_dataloader", "val_dataloader"):
        if loader_name not in config:
            continue
        dataset = config[loader_name]["dataset"]
        for key in ("img_folder", "ann_file"):
            if not Path(dataset[key]).exists():
                raise RuntimeError("Dataset path missing: " + dataset[key])
        if training and "test" in Path(dataset["ann_file"]).name.lower():
            raise RuntimeError("Training profiles must not load Test annotations")
    if config["DEIMCriterion"].get("ufd_enabled"):
        cdf = config["DEIMCriterion"].get("ufd_cdf_path")
        if cdf and not Path(cdf).is_file():
            raise RuntimeError("Difficulty CDF is missing")
    if args.compare_reference and not training:
        if sha256(release / profile["reference_eval"]) != profile["reference_eval_sha256"]:
            raise RuntimeError("Reference evaluation changed")
    if args.preflight_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", "profile": args.profile,
                          "training": training, "seed": seed,
                          "files_written": False, "gpu_used": False}), flush=True)
        return
    output.mkdir(parents=True)
    runtime_config = output / "config.yml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False))
    sys.path.insert(0, str(source))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    import torch
    import numpy as np
    from engine.core import YAMLConfig
    from engine.misc import dist_utils
    from engine.solver import TASKS
    from engine.solver.det_engine import evaluate
    dist_utils.setup_distributed(seed=seed)
    cfg = YAMLConfig(str(runtime_config))
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    if training:
        solver.fit()
        dist_utils.cleanup()
        return
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    _, evaluator = evaluate(
        model, solver.criterion, solver.postprocessor, solver.val_dataloader,
        solver.evaluator, solver.device,
    )
    evaluation = evaluator.coco_eval["bbox"].eval
    torch.save(evaluation, output / "eval.pth")
    report = {"profile": args.profile, "seed": seed, "weight_sha256": weight_sha,
              "evaluation_scope": "new_checkpoint" if args.checkpoint else "paper_checkpoint",
              "metrics": metrics(evaluation), "precision": "FP32", "weight_source": "ema.module"}
    if args.compare_reference:
        reference_path = release / profile["reference_eval"]
        if sha256(reference_path) != profile["reference_eval_sha256"]:
            raise RuntimeError("Reference evaluation changed")
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        exact = {key: bool(np.array_equal(evaluation[key], reference[key]))
                 for key in ("precision", "recall", "scores")}
        report["reference_metrics"] = metrics(reference)
        report["exact_arrays"] = exact
        report["metrics_equal"] = report["metrics"] == report["reference_metrics"]
        report["status"] = "PASS" if all(exact.values()) and report["metrics_equal"] else "MISMATCH"
    else:
        report["status"] = "EVALUATED"
    (output / "RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    dist_utils.cleanup()
    if report["status"] == "MISMATCH":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
