# BTC generalization reproduction

BTC uses the same engine as CDW1630:
`../../main_host/source/FINAL_CODE/`.
Dataset-specific profiles preserve BTC's three categories, difficulty CDF,
detector checkpoints, fitted mappings and value heads.

From the aggregate release root:

```bash
python -u -B main_host/source/FINAL_CODE/reproduce.py \
  --release-root . --profile btc_full_seed0_test1 \
  --output /path/to/new/btc_full_seed0_test1 --compare-reference
```

Use `btc_deim_seedN_test1`, `btc_gca_seedN_test1` or `btc_full_seedN_test1`;
Test2 profiles end in `test2`. N = 0, 1 or 2. Training profiles are
`btc_deim_train` and `btc_gca_train`; supply `--seed` and a new output directory.
Full HiLoR uses the GCA-trained detector with separately fitted BTC CCVR assets.
The local `scripts/manager.py` and `scripts/run_server.sh` accept the same profile
arguments. Test sets are evaluation-only. Original run records retain their
original paths; current profiles resolve assets inside the aggregate release.

## Original data and preparation

The official dataset is Mendeley Data version 1, DOI
`10.17632/24d45pf8wm.1`, under CC BY 4.0. Download instructions, original
contributors, citations and modification notices are in the aggregate
`DATA_AVAILABILITY.md`. `main_host/data_preparation/README.md`
documents the executable XML-to-COCO rebuild. It preserves the fixed
280/70 internal Train/Val split, untouched official 150/50 Test1/Test2
and published annotation bytes. The original and converted data already
bundled here remain unchanged.
