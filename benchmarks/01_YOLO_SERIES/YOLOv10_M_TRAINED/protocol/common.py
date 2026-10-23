"""Class names and COCO bbox conversion."""
import math


def normalize_names(value):
    if isinstance(value, dict):
        return [str(value[key]) for key in sorted(value, key=lambda item: int(item))]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise RuntimeError(f"Unsupported class-name schema: {type(value).__name__}")


def coco_row(image_id, category_id, coordinates, score, class_count):
    if not 0 <= category_id < class_count:
        raise RuntimeError("Predicted class id is outside the frozen class space")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError("Invalid detection score")
    if len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
        raise RuntimeError("Invalid detection coordinates")
    x1, y1, x2, y2 = (float(value) for value in coordinates)
    if x2 < x1 or y2 < y1:
        raise RuntimeError("Negative prediction box extent")
    return {"image_id": int(image_id), "category_id": int(category_id),
            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)}
