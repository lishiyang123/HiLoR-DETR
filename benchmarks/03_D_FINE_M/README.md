# D-FINE-M on CDW1630

See `../CONFIGURATION_GUIDE.md` for the project-specific training command,
initialization, environment and evaluation entry. The authoritative training
configuration is `configs/cdw1630_dfine_m_formal.yml`, including its upstream
closure: 102 epochs, batch 8, FP32, EMA. Official initialization is
`source/dfine_m_coco.pth`; no Objects365 initialization is substituted.

Selected paper checkpoints are identified by `runs/seedN/VAL_SELECTION.json`.
Seed 0/2 select stage1; seed 1 selects stage2. Saved Test predictions and
evaluation arrays are under `results/`. Use the release-root
`evaluate_test_predictions.py --model dfine-m --seed 0 --output ...` for CPU
metric replay. New training and inference outputs must not overwrite these
published assets.
