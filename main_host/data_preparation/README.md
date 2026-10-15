# Raw-data preparation for CDW1630 and BTC

Run from the aggregate release root. Python 3.10+ and Pillow are sufficient;
the existing model environments already provide them. No GPU or model weights
are required. Supplier download instructions, attribution and licenses are
centralized in `../../DATA_AVAILABILITY.md`; citations are in `DATASET_CITATIONS.bib`.

The supplied COCO annotations remain the default for direct reproduction. This
entry independently reconstructs them from source labels and fixed image lists.
`source_manifest.json` contains file hashes, split membership, output hashes and
descriptive metadata, not the converted bounding boxes. Raw files are checked
before conversion. Different supplier versions are refused rather than silently
producing a different experimental dataset.

## CDW1630: YOLO to COCO

Download the **version-5 YOLOv11** archive and extract it. Supply the directory
containing `train/images`, `train/labels`, `valid/` and `test/`:

```bash
python -B main_host/data_preparation/prepare_dataset.py \
  --dataset cdw1630 --raw-root '/path/to/AI waste Sorting_2.v5i.yolov11' \
  --output /path/to/new/cdw1630_coco
```

Bundled raw source for an offline check:

```bash
python -B main_host/data_preparation/prepare_dataset.py \
  --dataset cdw1630 \
  --raw-root 'benchmarks/01_YOLO_SERIES/OFFICIAL_ASSETS_INBOX/AI waste Sorting_2.v5i.yolov11' \
  --verify-only
```

The original 1,144/249/237 image split is unchanged. Class IDs are 0: Brick,
1: Concrete, 2: Plastic Bottle, 3: Reinforcement, 4: timber. Files are ordered
lexicographically; image and annotation IDs start at 1 independently per split.
YOLO normalized center/size coordinates are converted using actual image width
and height; COCO coordinates and area are rounded to four decimal places.
No image is resized, reencoded or augmented by this script. The supplier lists
640x640 preprocessing, but the preserved download contains one 588x441 Train
image; its actual dimensions are retained. Output images are under
`train2017`, `val2017`, `test2017`; annotations use the corresponding
`instances_*2017.json` names. The original COCO `licenses` field is left unchanged
for byte-identical reconstruction; the dataset license is documented centrally.

## BTC: Pascal VOC XML to COCO

Download Mendeley **version 1**, DOI 10.17632/24d45pf8wm.1. After extraction,
the input root must contain `training/`, `testing_set_1/`, `testing_set_2/`,
with each JPG next to its same-stem XML. Do not use an independently resplit or
relabelled copy.

```bash
python -B main_host/data_preparation/prepare_dataset.py \
  --dataset btc --raw-root /path/to/extracted/BTC \
  --output /path/to/new/btc_coco
```

Bundled raw-source check:

```bash
python -B main_host/data_preparation/prepare_dataset.py \
  --dataset btc --raw-root generalization/btc/data/raw_official \
  --verify-only
```

Class IDs are 0: brick, 1: concrete, 2: tile. The official 350 training images
are divided into the same fixed 280 Train/70 Val images used by every detector
seed. Rank original training stems by SHA256 of
`BTC_INTERNAL_280_70_V1|seed=0|stem=<stem>`; the first 70 are Val. Within each
subset, stems use numeric order. The script verifies this rule against the
published membership, also recorded in
`generalization/btc/protocol/BTC_INTERNAL_SPLIT_SEED0.json`.
Official Test1 (150 images) and Test2 (50 images) are untouched.

The original coordinate convention is **x=xmin-1, y=ymin-1,
w=xmax-xmin, h=ymax-ymin**, without an additional inclusive +1. Objects retain
XML order; only entirely empty `<object/>` nodes are ignored (945 total).
Incomplete nonempty objects, unknown classes, invalid boxes and size mismatches
are errors. No confidence filtering or model outputs are involved.

Image-ID bases are 0/500000/1000000/2000000 and annotation-ID bases are
0/50000000/100000000/200000000 for Train/Val/Test1/Test2. The combined Test JSON
concatenates the two official tests with those IDs intact; it is reporting-only,
not a source for training or selection. The output uses `train2017`, `val2017`, `test1`,
`test2`, `test_combined` image directories and `annotations/instances_*.json`.

## Validation and use

Every generated annotation must be **byte-identical** to its published COCO
reference before any output is written: three CDW1630 files and five BTC files.
`--verify-only` performs the reconstruction in memory without writing outputs.
Normal preparation writes only to a new directory outside the release and raw
input; it never overwrites supplied data. Image copies preserve the original
bytes. Use the new annotation/image paths as the dataset inputs of your chosen
training or evaluation configuration; the original published paths are unchanged.

```bash
python -B -m unittest discover \
  -s main_host/data_preparation -p 'test_prepare_dataset.py'
```
