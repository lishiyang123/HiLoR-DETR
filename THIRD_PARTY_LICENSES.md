# Third-party licenses and attribution

This is the central guide to the licenses and original sources of the materials
in this release. The root `LICENSE` applies to author-owned code where the
applicable upstream license permits; it does not relicense third-party code,
data or weights. GPL/AGPL-covered components and modifications remain subject to
their respective licenses. Preserve upstream copyright and attribution notices.

## Source code

| Component | License | Bundled full text | Official source |
|---|---|---|---|
| DEIM / shared CDW1630 and BTC host | Apache-2.0 | `main_host/source/FINAL_CODE/LICENSE` | [DEIM](https://github.com/Intellindust-AI-Lab/DEIM) |
| RT-DETR / RT-DETR host adaptation | Apache-2.0 | `benchmarks/02_RT_DETR_R18/source/RT-DETR_official/LICENSE`; `second_host/source/RT-DETR_HiLoR/LICENSE` | [RT-DETR](https://github.com/lyuwenyu/RT-DETR) |
| D-FINE | Apache-2.0 | `benchmarks/03_D_FINE_M/source/D-FINE_official/LICENSE` | [D-FINE](https://github.com/Peterande/D-FINE) |
| Ultralytics runtime for YOLOv8 / YOLO11 (8.3.221) | AGPL-3.0 | `benchmarks/01_YOLO_SERIES/YOLOv8_L_TRAINED/source/Ultralytics_official/LICENSE`; `LICENSE-AGPL-3.0.txt` | [Pinned license](https://github.com/ultralytics/ultralytics/blob/v8.3.221/LICENSE) |
| YOLOv9 | GPL-3.0 | `LICENSE-GPL-3.0.txt` | [YOLOv9](https://github.com/WongKinYiu/yolov9), [license](https://github.com/WongKinYiu/yolov9/blob/main/LICENSE.md) |
| YOLOv10 | AGPL-3.0 | `benchmarks/01_YOLO_SERIES/YOLOv10_M_TRAINED/source/YOLOv10_official/LICENSE`; `LICENSE-AGPL-3.0.txt` | [YOLOv10](https://github.com/THU-MIG/yolov10) |
| YOLOv12 | AGPL-3.0 | `benchmarks/01_YOLO_SERIES/YOLOv12_M_TRAINED/source/YOLOv12_official/LICENSE`; `LICENSE-AGPL-3.0.txt` | [YOLOv12](https://github.com/sunsmarterjie/yolov12) |
| calflops 0.3.2 | MIT | `efficiency/vendor/calflops-0.3.2.dist-info/licenses/LICENSE` | [calflops](https://github.com/MrYxJ/calculate-flops.pytorch) |

The full GPL text at the release root accompanies the bundled YOLOv9 source at
`benchmarks/10_YOLOV9_C/source/YOLOv9_official/`.
Keep this guide and the referenced license texts when redistributing the code
archive. For a component-only redistribution, also include its applicable full
license, copyright notices and corresponding source as required by that license.
Making a repository public does not replace these obligations. Merely collecting
independent programs in one archive does not relicense them under Apache-2.0;
the relevant copyleft requirements still apply to covered combined or modified
works.

The RT adapter's donor references retain their original notices. Gilg conditional
Beta calibration and Soft-NMS are independent project implementations with
reference attribution in their source/protocol, not wholesale imports of the
authors' repositories. Python/CUDA environments are installed separately; their
packages retain their own licenses.

## Datasets

| Data | Provider and version | License | Project changes |
|---|---|---|---|
| CDW1630 | umesh / Umesh Samarasekara; [Roboflow AI waste Sorting_2 v5](https://universe.roboflow.com/umesh-5ausw/ai-waste-sorting_2-bnqge/dataset/5), [Zenodo v1](https://doi.org/10.5281/zenodo.17447590) | CC BY 4.0 | YOLO-to-COCO annotation conversion; supplied image pixels and Train/Val/Test membership preserved |
| BTC | Demetris Demetriou, Pavlos Mavromatidis, Robert Mwombeki, Harris Papadopoulos, Michael Petrou and Demetris Nicolaides; [Mendeley v1](https://doi.org/10.17632/24d45pf8wm.1) | CC BY 4.0 | VOC-to-COCO conversion; fixed internal Train/Val split of the original training set; original Test1/Test2 preserved |

The full data license is `LICENSE-CC-BY-4.0.txt`; the official license page is
<https://creativecommons.org/licenses/by/4.0/>. Retain provider attribution,
source/version, the license notice and an indication of changes when sharing
these data. `DATA_AVAILABILITY.md` contains the exact splits, conversions and
download/rebuild instructions. The code license does not replace the data
license or grant privacy, publicity or other rights outside its scope.

## Checkpoints and fitted assets

Source-code licenses and dataset licenses must not be treated as a blanket
relicensing of every checkpoint. Initializers and public comparison checkpoints
retain their upstream model/data terms. Project-trained detectors retain the
applicable upstream and training-data terms; fitted mapping/value assets are
distributed with their corresponding detector bindings, not as new permission
for an upstream model.

| Model assets | Original release/source | Applicable notice |
|---|---|---|
| DEIM initializers and derived main-host detectors | [Official DEIM repository and model links](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 project; preserve upstream notices and check the terms attached to the specific model release |
| RT-DETR-R18 initializer and derived detectors | [Official RT-DETR PyTorch models](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch) | Apache-2.0 project; preserve the original model source and any model-specific terms |
| D-FINE-M initializer and derived detectors | [Official D-FINE models](https://github.com/Peterande/D-FINE), [author's model repository](https://huggingface.co/Peterande/D-FINE) | Apache-2.0 project/model repository; model-specific training-data terms still apply |
| YOLOv9-C | [Official release](https://github.com/WongKinYiu/yolov9/releases/tag/v0.1) | GPL-3.0 project; preserve the specific model source and applicable weight terms |
| YOLOv10-M | [Official releases](https://github.com/THU-MIG/yolov10/releases) | AGPL-3.0 project; preserve the specific model source and applicable weight terms |
| YOLOv12-M (turbo) | [Official turbo release](https://github.com/sunsmarterjie/yolov12/releases/tag/turbo) | AGPL-3.0 project; preserve the turbo variant and applicable weight terms |
| YOLOv8-L COCO initializer and derived three-seed detectors | [Official COCO initializer](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt), [corresponding source](https://github.com/ultralytics/ultralytics/tree/v8.3.221) | Ultralytics AGPL-3.0 terms apply; retain the corresponding source and notices. |

Exact model filenames, source revisions, checkpoint identities and training
settings are in the experiment manifests/configurations. This notice preserves
those sources; it does not claim an additional grant from any model owner.

Provider CDW-trained YOLOv8-L / YOLO11-L checkpoints are not release inputs.
YOLOv8-L and YOLO11-L use bundled official COCO initializers, corresponding
source and derived three-seed detectors.

Official sources and license texts were checked on 2026-10-15. This inventory
documents their stated terms; it is not an independent legal authorization.
