# RT-DETR environment installation

Use Linux with Python 3.11 and the recorded RT-DETR PyTorch/CUDA stack.
This environment is separate from the main-host Torch 2.5.1 environment.
Run the following commands from the aggregate release root:

```bash
conda create -n rtdetr_sota python=3.11 -y
conda activate rtdetr_sota
python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r second_host/requirements_runtime.txt
python -m pip check
```

`requirements_runtime.txt` pins the application dependencies, including
`faster-coco-eval==1.7.2`. NumPy remains at 1.26.4 for this Torch 2.0.1 stack;
do not install the main-host requirements into this environment. The original
environment inventory remains available at
`../benchmarks/02_RT_DETR_R18/environment/rtdetr_sota_pip_freeze.txt`.

Check the installed stack without loading weights or using a GPU:

```bash
python -c "import torch, torchvision; from importlib.metadata import version; print(torch.__version__, torchvision.__version__, version('faster-coco-eval'))"
```

Expected versions are `2.0.1+cu118`, `0.15.2+cu118` and `1.7.2`.
Saved-prediction replay, table extraction and CPU tests do not require a GPU.
Checkpoint inference and detector training require a compatible NVIDIA GPU and
driver. Installing Python dependencies does not install checkpoints, binding
JSON files or dataset images.

Use the [task-to-download table](../DISTRIBUTION.md#downloads-by-task) before
running a command. In particular, full RT inference needs the results archive
for `second_host/artifacts/inference/seedN/qccr_umr_binding.json` and its fitted
metadata, as well as the selected detector/value-head weights. Running all RT
CPU tests also requires the main-host COCO initializer; see the same table.
Train/Val feature archives are optional when using already fitted CCVR assets.
