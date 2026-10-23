#!/usr/bin/env python3
"""Train RT-DETR Native or GCCS with the shared faster-coco-eval Val backend."""
import argparse
from pathlib import Path
import sys

RELEASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RELEASE))
import reproduce_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('native', 'gccs'), required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    sys.argv = [str(RELEASE / 'reproduce_pipeline.py'), '--host', 'rtdetr',
                '--dataset', 'cdw', '--method', args.method, '--seed', str(args.seed),
                '--stage', 'train', '--workdir', str(args.workdir)]
    if args.preflight_only:
        sys.argv.append('--preflight-only')
    reproduce_pipeline.main()


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
