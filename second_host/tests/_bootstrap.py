"""Make the pinned RT-DETR source and bundled test dependencies importable."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
for path in (
    WORKSPACE / ".local_test_deps",
    ROOT / "source" / "RT-DETR_HiLoR" / "rtdetr_pytorch",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

# Run in the documented RT environment (including torchvision 0.15.2).
# Import the real data stack: process-wide stubs here would contaminate tests
# of the evaluator when unittest discovers the whole suite in one process.
