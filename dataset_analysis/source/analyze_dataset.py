"""Describe CDW1630 and prepare fixed-density and Train-overlap Test groups."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import shutil
import statistics

SPLITS = ("Train", "Val", "Test")
SIZE_BINS = ("small", "medium", "large")
DENSITY_BINS = ("0", "1", "2", "3-5", "6-10", ">10")
DENSITY_THRESHOLD = 5


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_annotations(project):
    data = project / "main_host/datasets/CDW1630"
    parsed = {}
    categories = None
    for split in SPLITS:
        path = data / f"annotations/instances_{split.lower()}2017.json"
        payload = json.loads(path.read_text())
        images = {int(row["id"]): row for row in payload["images"]}
        mapping = {int(row["id"]): row["name"] for row in payload["categories"]}
        assert len(images) == len(payload["images"]) and len(mapping) == 5
        assert len({int(row["id"]) for row in payload["annotations"]}) == len(payload["annotations"])
        if categories is None:
            categories = mapping
        assert mapping == categories
        for image in images.values():
            assert (data / f"{split.lower()}2017" / image["file_name"]).is_file()
        for row in payload["annotations"]:
            assert int(row["image_id"]) in images and int(row["category_id"]) in mapping
            assert len(row["bbox"]) == 4 and all(math.isfinite(float(v)) for v in row["bbox"])
            assert float(row["bbox"][2]) > 0 and float(row["bbox"][3]) > 0
        parsed[split] = dict(images=images, annotations=payload["annotations"], categories=mapping)
    return parsed


def size_bin(area):
    if area < 32.0 ** 2:
        return "small"
    if area < 96.0 ** 2:
        return "medium"
    return "large"


def density_bin(count):
    if count <= 2:
        return str(count)
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    return ">10"


def describe(split, images, annotations, categories):
    grouped = defaultdict(list)
    for row in annotations:
        grouped[int(row["image_id"])].append(row)
    image_rows, class_counts, class_images, size_counts = [], Counter(), Counter(), Counter()
    for image_id in sorted(images):
        rows = grouped[image_id]
        image_rows.append(dict(split=split, image_id=image_id, file_name=images[image_id]["file_name"],
                               annotated_objects=len(rows), density_bin=density_bin(len(rows))))
        class_images.update({int(row["category_id"]) for row in rows})
        for row in rows:
            class_counts[int(row["category_id"])] += 1
            size_counts[size_bin(float(row["bbox"][2]) * float(row["bbox"][3]))] += 1
    return summarize(split, image_rows, class_counts, class_images, size_counts, categories)


def summarize(split, image_rows, class_counts, class_images, size_counts, categories):
    counts = [row["annotated_objects"] for row in image_rows]
    n, total = len(counts), sum(counts)
    summary = dict(split=split, images=n, instances=total, objects_per_image_mean=total / n,
                   objects_per_image_median=statistics.median(counts), objects_per_image_min=min(counts),
                   objects_per_image_max=max(counts))
    class_rows = [dict(split=split, category_id=k, category=categories[k], instances=class_counts[k],
                       instance_percent=100.0 * class_counts[k] / total if total else 0.0,
                       images_containing_class=class_images[k], image_percent=100.0 * class_images[k] / n)
                  for k in sorted(categories)]
    size_rows = [dict(split=split, size_bin=k, instances=size_counts[k],
                      percent=100.0 * size_counts[k] / total if total else 0.0) for k in SIZE_BINS]
    density_counts = Counter(row["density_bin"] for row in image_rows)
    density_rows = [dict(split=split, density_bin=k, images=density_counts[k],
                         percent=100.0 * density_counts[k] / n) for k in DENSITY_BINS]
    assert sum(row["instances"] for row in class_rows) == total
    assert sum(row["instances"] for row in size_rows) == total
    assert sum(row["images"] for row in density_rows) == n
    return dict(split=summary, **{"class": class_rows}, size=size_rows,
                density_images=image_rows, density_summary=density_rows)


def descriptive_tables(parsed):
    per_split = {split: describe(split, **parsed[split]) for split in SPLITS}
    all_images = [row for split in SPLITS for row in per_split[split]["density_images"]]
    class_counts, class_images, size_counts = Counter(), Counter(), Counter()
    for split in SPLITS:
        class_counts.update({row["category_id"]: row["instances"] for row in per_split[split]["class"]})
        class_images.update({row["category_id"]: row["images_containing_class"] for row in per_split[split]["class"]})
        size_counts.update({row["size_bin"]: row["instances"] for row in per_split[split]["size"]})
    per_split["All"] = summarize("All", all_images, class_counts, class_images, size_counts,
                                 parsed["Train"]["categories"])
    fields = {"SPLIT_SUMMARY.csv": "split", "CLASS_DISTRIBUTION.csv": "class",
              "SIZE_DISTRIBUTION.csv": "size", "OBJECT_DENSITY_DISTRIBUTION.csv": "density_summary"}
    tables = {name: [row for split in (*SPLITS, "All")
                     for row in ([per_split[split][key]] if key == "split" else per_split[split][key])]
              for name, key in fields.items()}
    tables["IMAGE_OBJECT_COUNTS.csv"] = all_images
    return per_split, tables


def intersection_over_own(box, other):
    x1, y1, w1, h1 = map(float, box)
    x2, y2, w2, h2 = map(float, other)
    width = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    height = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return width * height / (w1 * h1)


def image_scores(checked):
    grouped = defaultdict(list)
    for row in checked["annotations"]:
        grouped[int(row["image_id"])].append(row)
    scores = {}
    for image_id in sorted(checked["images"]):
        rows = grouped[image_id]
        values = [max((intersection_over_own(row["bbox"], other["bbox"])
                       for j, other in enumerate(rows) if i != j), default=0.0)
                  for i, row in enumerate(rows)]
        scores[image_id] = dict(gt_count=len(rows), mean_max_ioa=sum(values) / len(values) if values else 0.0)
    return scores


def group_thresholds(scores):
    values = [row["mean_max_ioa"] for row in scores.values()]
    median = float(statistics.median(values))
    q75 = float(statistics.quantiles(values, n=4, method="inclusive")[2])
    threshold = median if median > 0 else q75 if q75 > 0 else 0.0
    strategy = "positive_median" if median > 0 else "positive_q75_after_zero_median" if q75 > 0 else "zero_versus_positive"
    return dict(format_version=1,
                density=dict(threshold=DENSITY_THRESHOLD, strategy="fixed",
                             low_rule="N_i <= 5", high_rule="N_i > 5"),
                overlap=dict(threshold=threshold, strategy=strategy, source_split="Train",
                             train_image_count=len(scores), train_median=median, train_q75=q75,
                             zero_score_images=sum(v == 0 for v in values), positive_score_images=sum(v > 0 for v in values),
                             low_rule="O_i <= threshold", high_rule="O_i > threshold"))


def grouping_tables(parsed):
    scores = {split: image_scores(parsed[split]) for split in SPLITS}
    thresholds = group_thresholds(scores["Train"])
    groups = {name: [] for name in ("density_low", "density_high", "overlap_low", "overlap_high")}
    for image_id, row in scores["Test"].items():
        groups["density_low" if row["gt_count"] <= thresholds["density"]["threshold"] else "density_high"].append(image_id)
        groups["overlap_low" if row["mean_max_ioa"] <= thresholds["overlap"]["threshold"] else "overlap_high"].append(image_id)
    groups = {name: dict(image_ids=ids, image_count=len(ids),
                         gt_count=sum(scores["Test"][i]["gt_count"] for i in ids)) for name, ids in groups.items()}
    for low, high in (("density_low", "density_high"), ("overlap_low", "overlap_high")):
        assert set(groups[low]["image_ids"]).isdisjoint(groups[high]["image_ids"])
        assert set(groups[low]["image_ids"]) | set(groups[high]["image_ids"]) == set(scores["Test"])
    per_image = [dict(split=split, image_id=i, file_name=parsed[split]["images"][i]["file_name"], **row)
                 for split in SPLITS for i, row in scores[split].items()]
    overlap_rows = []
    for split in (*SPLITS, "All"):
        selected = [r for r in per_image if split == "All" or r["split"] == split]
        positive = sum(row["mean_max_ioa"] > 0 for row in selected)
        overlap_rows.append(dict(split=split, images=len(selected), images_with_box_overlap=positive,
                                 image_percent=100.0 * positive / len(selected),
                                 mean_image_max_ioa=statistics.mean(row["mean_max_ioa"] for row in selected),
                                 median_image_max_ioa=statistics.median(row["mean_max_ioa"] for row in selected)))
    test_groups = dict(format_version=1, test_image_count=len(scores["Test"]), thresholds=thresholds, groups=groups,
                       image_scores={str(i): row for i, row in scores["Test"].items()},
                       selection_policy="Fixed density cutoff 5; Train-derived overlap threshold; whole Test images and all their GT objects")
    return thresholds, test_groups, per_image, overlap_rows


def paper_summary(per_split, thresholds, groups, overlap):
    lines = ["# CDW1630 dataset statistics", "", "| Split | Images | GT objects | Mean objects/image | Median | Range |",
             "|---|---:|---:|---:|---:|---|" ]
    for split in (*SPLITS, "All"):
        row = per_split[split]["split"]
        lines.append(f"| {split} | {row['images']} | {row['instances']} | {row['objects_per_image_mean']:.6f} | {row['objects_per_image_median']:g} | {row['objects_per_image_min']}–{row['objects_per_image_max']} |")
    lines += ["", "| Category | Train | Val | Test | All | All (%) |", "|---|---:|---:|---:|---:|---:|"]
    for i, row in enumerate(per_split["All"]["class"]):
        counts = [per_split[split]["class"][i]["instances"] for split in (*SPLITS, "All")]
        lines.append(f"| {row['category']} | " + " | ".join(map(str, counts)) + f" | {row['instance_percent']:.4f} |")
    lines += ["", "| Bbox size | Instances | Percent |", "|---|---:|---:|"]
    for row in per_split["All"]["size"]:
        lines.append(f"| {row['size_bin']} | {row['instances']} | {row['percent']:.6f} |")
    lines += ["", f"Fixed density threshold: {thresholds['density']['threshold']}; Train overlap threshold: {thresholds['overlap']['threshold']!r}.",
              "", "| Test group | Images | GT objects |", "|---|---:|---:|"]
    for name, row in groups["groups"].items():
        lines.append(f"| {name} | {row['image_count']} | {row['gt_count']} |")
    row = next(r for r in overlap if r["split"] == "All")
    lines += ["", f"{row['images_with_box_overlap']} of {row['images']} images ({row['image_percent']:.6f}%) contain boxes with positive-area overlap.",
              "", "Percentages use the image or GT-object denominator stated by each table. These are annotation statistics; no model performance is calculated."]
    return "\n".join(lines) + "\n"


README = """# CDW1630 dataset statistics

From the release root, run with Python 3.8 or newer and a new output directory:

```bash
python -B dataset_analysis/source/analyze_dataset.py --project . --output /path/to/new/data_stats
```

The only inputs are the three COCO annotations and image paths under
`main_host/datasets/CDW1630/`. Image paths are checked; pixels, model outputs and
the GPU are not accessed. The script uses only the Python standard library.

- `tables/`: split image/GT counts, class instance and image-presence counts,
  bbox scales, per-image object counts and density bins. CSV names retain the
  paper's existing table interfaces. Empty images are included in image statistics.
- `results/`: full-precision JSON and a paper-facing summary with tables.
- `grouping/`: fixed density cutoff, Train overlap threshold, Test image IDs, image scores and subset sizes;
  no subgroup AP results are included.

Bbox scale uses width times height in annotation pixels: small <32²,
medium ≥32² and <96², large ≥96². Density bins are 0, 1, 2, 3–5, 6–10 and >10.
Each class's image count counts an image once regardless of instance count.

For grouping, density is the number of GT boxes. Each object's overlap score is
its maximum intersection-over-own-area with another GT box, across all classes;
the image score is the mean over its objects, or zero for an empty/single-object
image. The density threshold is fixed at 5 (low ≤5, high >5). The overlap threshold is the
Train median when positive, otherwise its inclusive 75th percentile when positive,
otherwise zero. Low uses ≤ and high uses >. Groups contain whole Test images and
all their GT objects.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, output = args.project.resolve(strict=True), args.output.resolve()
    if output.exists() or output == project or project in output.parents:
        parser.error("Use a new output directory outside the release")
    parsed = load_annotations(project)
    per_split, tables = descriptive_tables(parsed)
    thresholds, groups, image_rows, overlap = grouping_tables(parsed)
    for name in ("source", "tables", "results", "grouping"):
        (output / name).mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__).resolve(), output / "source/analyze_dataset.py")
    (output / "README.md").write_text(README, encoding="utf-8")
    for name, rows in tables.items():
        write_csv(output / "tables" / name, rows)
    summary = dict(dataset="CDW1630", categories=parsed["Train"]["categories"],
                   split_summary=tables["SPLIT_SUMMARY.csv"],
                   class_distribution_all=per_split["All"]["class"], size_distribution_all=per_split["All"]["size"],
                   density_distribution_all=per_split["All"]["density_summary"])
    write_json(output / "results/DATASET_CHARACTERISTICS_SUMMARY.json", summary)
    (output / "results/PAPER_SUMMARY.md").write_text(paper_summary(per_split, thresholds, groups, overlap), encoding="utf-8")
    write_json(output / "grouping/thresholds.json", thresholds)
    write_json(output / "grouping/test_image_groups.json", groups)
    write_csv(output / "grouping/image_scores.csv", image_rows)
    write_csv(output / "grouping/overlap_summary.csv", overlap)
    write_csv(output / "grouping/group_counts.csv", [dict(group=name, **{k: row[k] for k in ("image_count", "gt_count")})
                                                   for name, row in groups["groups"].items()])
    print(json.dumps(summary["split_summary"], indent=2))
    print("Fixed density / Train overlap thresholds:", thresholds["density"]["threshold"], thresholds["overlap"]["threshold"])
    print("Saved annotation statistics:", output)


if __name__ == "__main__":
    main()
