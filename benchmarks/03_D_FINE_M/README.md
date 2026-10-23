# D-FINE-M on CDW1630

See `../CONFIGURATION_GUIDE.md` for the project-specific training command,
initialization, environment and evaluation entry. The authoritative training
configuration is `configs/cdw1630_dfine_m_formal.yml`, including its upstream
closure: 102 epochs, batch 8, FP32, EMA. Official initialization is
`source/dfine_m_coco.pth`; no Objects365 initialization is substituted.

Selected paper checkpoints are identified by `runs/seedN/VAL_SELECTION.json`.
Each run contains one selected checkpoint, its expanded configuration and compact
training metrics. Test predictions and unrounded metrics are under
`results/test_reeval_v1_1/seedN/`.

From the release root, in the D-FINE environment:

```bash
python -B benchmarks/reproduce_comparison.py --model dfine --seed 0 --output /your/new/dfine_test --evaluate
```

Repeat for seeds 1 and 2. This uses the selected EMA checkpoint and the released
FP32, batch-8 Test path; it does not retrain or select checkpoints.
Saved Test predictions and evaluation arrays are under `results/`. Use the release-root
`evaluate_test_predictions.py --model dfine-m --seed 0 --output ...` for CPU
metric replay. New training and inference outputs must not overwrite these
published assets.
