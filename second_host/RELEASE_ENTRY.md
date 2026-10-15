# Second-host entry

Use `README.md` for the complete workflow and `CURRENT_RESULT_MANIFEST.json`
for the selected checkpoints and Test outputs. The publication tables are:

- `paper_tables/RTDETR_COCO_AND_COMPLEXITY.csv`: twelve COCO metrics and Params/GFLOPs.
- `paper_tables/RTDETR_PER_SEED.csv`: three seeds for Native, GCCS and full HiLoR-DETR.
- `paper_tables/RTDETR_MEAN_STD_AND_DELTA.csv`: mean, sample SD and stage increments,
  including AP80/AP85/AP90/AP95.

Native assets are shared with `../benchmarks/02_RT_DETR_R18/`.
The method source is `source/RT-DETR_HiLoR/rtdetr_pytorch/`. Stored predictions
support CPU evaluation replay; checkpoint inference needs GPU. New training,
capture, fitting and inference outputs must use a fresh directory outside the release.
