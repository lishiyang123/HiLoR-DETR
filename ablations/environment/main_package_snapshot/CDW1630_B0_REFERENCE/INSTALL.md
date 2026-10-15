# Environment installation

The files in this directory serve two different purposes:

- `runtime.json`, `runtime_environment.txt`, and `pip_freeze.txt` record the
  exact environment used for the canonical seed0 run.
- `conda_from_history.yml` and `conda_full_no_builds.yml` are provenance
  snapshots. They contain no dependency list and are not sufficient by
  themselves to recreate the environment.

For a practical recreation, create Python 3.11.9, install the recorded CUDA
12.4 PyTorch wheels, and then install the application dependencies:

```bash
conda create -n deim_cvpr2025 python=3.11.9 -y
conda activate deim_cvpr2025
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r environment/requirements_runtime.txt
```

`pip_freeze.txt` remains the authoritative full package inventory. Exact
kernel-level numerical reproduction can still depend on the recorded GPU,
driver, CUDA runtime, cuDNN, and platform values in `runtime.json`.
