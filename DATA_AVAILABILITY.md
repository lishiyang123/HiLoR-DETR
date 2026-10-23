# Dataset access, attribution and preparation

This is the central data-access guide. Dataset licenses are independent of model
code and checkpoint licenses. Cite the original dataset providers; this project
does not claim authorship of their images or annotations. Bibliographic entries
are in `main_host/data_preparation/DATASET_CITATIONS.bib`.

## CDW1630

- Original provider: umesh, Roboflow Universe **AI waste Sorting_2, version 5**.
- Version-specific source and export page:
  <https://universe.roboflow.com/umesh-5ausw/ai-waste-sorting_2-bnqge/dataset/5>.
  Choose YOLOv11 export, not another version or a platform-generated new split.
- Archived original-provider release: Umesh Samarasekara, Zenodo **v1**,
  DOI <https://doi.org/10.5281/zenodo.17447590>.
  Download `AI waste Sorting_2.v5i.yolov11.zip` from
  <https://zenodo.org/records/17447590>. The extracted source is bundled under
  `benchmarks/01_YOLO_SERIES/OFFICIAL_ASSETS_INBOX/AI waste Sorting_2.v5i.yolov11/`.
- The Roboflow version page and original archive README identify **CC BY 4.0**;
  retain those attribution/license notices when using or redistributing the data.
- Source Train/Val/Test split: **1,144/249/237** images. The experimental split
  retains **1,042/237/237** images (**5,336/958/1,071** objects). Source-name and
  C1 cross-split groups retain Test before Val before Train; 102 Train and 12 Val
  images are excluded, without moving images between splits. Exact membership
  and thresholds are in `main_host/datasets/CDW1630/split_summary.json`. This
  rule does not remove all near-duplicates. Categories are Brick, Concrete, Plastic Bottle,
  Reinforcement and timber, with COCO IDs 0--4 in that order.
- The source page reports Auto-Orient and 640x640 stretch preprocessing with no
  augmentation. These are supplier preprocessing operations, not operations
  added by this project's conversion script. One preserved Train image has
  actual size 588x441; conversion uses its actual size and leaves pixels intact.
- Project modifications: YOLO labels are converted to COCO; box coordinates and
  area are rounded to four decimal places; image pixels and retained IDs are
  unchanged. The conversion script applies the declared exclusions. The supplied converted data are under
  `main_host/datasets/CDW1630/`.

## BTC (brick/concrete/tile)

- Official dataset title: **Construction and Demolition Waste Object Detection
  Dataset**; "BTC" is this project's shorthand for its three classes.
- Original contributors: Demetris Demetriou, Pavlos Mavromatidis, Robert
  Mwombeki, Harris Papadopoulos, Michael Petrou and Demetris Nicolaides.
- Version: **1**, published **19 December 2022**; DOI
  <https://doi.org/10.17632/24d45pf8wm.1>.
- Obtain the original JPG/XML files using **Download All** on the official
  Mendeley page: <https://data.mendeley.com/datasets/24d45pf8wm/1>.
  After extraction use the root containing `training/`, `testing_set_1/` and
  `testing_set_2/`. The same original files are bundled under
  `generalization/btc/data/raw_official/`.
- The official version page identifies **CC BY 4.0**. Retain contributor,
  dataset-title, source, DOI, version and license attribution.
- Original data: **350 training images, 150 Test1 images, 50 Test2 images**;
  original annotations contain 4,230/1,727/596 valid boxes respectively.
- Project modifications: the original training images are divided into a fixed
  **280 Train/70 Val** subset; official Test1 and Test2 are unchanged. The split
  is identical for detector seeds 0/1/2 and is explicitly listed in
  `generalization/btc/protocol/BTC_INTERNAL_SPLIT_SEED0.json`.
  XML-to-COCO conversion uses x=xmin-1, y=ymin-1, w=xmax-xmin, h=ymax-ymin.
  Entirely empty XML object nodes are ignored; images are not resized or
  reencoded. Category IDs are 0: brick, 1: concrete, 2: tile.
- Converted COCO data: `generalization/btc/data/coco_deim/`.
  The combined Test annotation is reporting-only. BTC detector training and
  calibration use BTC Train/Val; these results are not zero-shot CDW evaluation.

## Attribution and license notices

Both datasets are identified as CC BY 4.0 by their respective official source
pages. The license requires appropriate attribution, a license link and an
indication of changes; see the provider notices and
<https://creativecommons.org/licenses/by/4.0/>. The full license text is bundled
at `LICENSE-CC-BY-4.0.txt`; `THIRD_PARTY_LICENSES.md` is the central license guide.
Cite the original datasets even
when obtaining the bundled copies. The project modifications are described
above and in the conversion README. These notices do not confer additional
rights or override third-party privacy or other applicable rights.

## Rebuild or use the supplied COCO data

Direct reproduction can use the supplied converted annotations without further
data preparation. To reconstruct them from the original YOLO/VOC labels, use
`main_host/data_preparation/prepare_dataset.py`; its README provides
download/extraction layouts and commands for both datasets. The file checks
raw-source identities, preserves the declared splits and requires identical
published annotation bytes. It needs Python and Pillow, not a GPU. Original
data, checkpoints, calibration artifacts and published results are not overwritten.

Train is used for detector training and training-only statistics/features. Val
is used only for the declared checkpoint/calibration choices. Test is used for
frozen evaluation and disclosed reporting analyses, not selection or fitting.
External source pages and licensing notices were checked on **2026-10-15**.
