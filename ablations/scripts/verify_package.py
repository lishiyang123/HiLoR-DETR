"""Verify the aggregate release's source, asset and result bindings."""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", nargs="?", default="verify", choices=["verify", "preflight"])
parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
args = parser.parse_args()
raise SystemExit(subprocess.call([sys.executable, "-B", str(args.project / "verify_release.py"), "verify", "--release", str(args.project)]))
