"""Run the shared CPU per-class analysis."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    root=args.package_root.resolve().parent
    subprocess.run([sys.executable,'-B',str(root/'dataset_analysis/run.py'),'--stage','classes',
                    '--output',str(args.output_dir.resolve())],check=True)


if __name__=='__main__':
    main()
