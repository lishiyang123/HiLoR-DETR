"""Paired image-bootstrap helpers from the reproduction package."""

from pathlib import Path
import contextlib
import copy
import io

CONFIDENCE = 0.95


def fail(message):
    raise RuntimeError(message)


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
    """Use the same faster-coco-eval backend as the reproduction package."""
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
    """Repeat cached image records and recompute global COCO accumulation."""
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
