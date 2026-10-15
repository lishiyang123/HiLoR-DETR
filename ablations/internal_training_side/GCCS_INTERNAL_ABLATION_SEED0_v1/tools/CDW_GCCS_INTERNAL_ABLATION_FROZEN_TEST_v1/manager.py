"""Run a release-bound main-host profile in a new output directory."""
import os
from pathlib import Path
import subprocess
import sys

def main():
    here = Path(__file__).resolve()
    root = next((p for p in here.parents if (p / "MAIN_HOST_SOURCE.json").is_file()), None)
    if root is None:
        raise RuntimeError("Aggregate release root not found")
    root = Path(os.environ.get("FINAL_PAPER_RELEASE_ROOT", str(root))).resolve()
    executable = root / "main_host/source/FINAL_CODE/reproduce.py"
    raise SystemExit(subprocess.call([sys.executable, "-u", "-B", str(executable), "--release-root", str(root), *sys.argv[1:]]))

if __name__ == "__main__":
    main()
