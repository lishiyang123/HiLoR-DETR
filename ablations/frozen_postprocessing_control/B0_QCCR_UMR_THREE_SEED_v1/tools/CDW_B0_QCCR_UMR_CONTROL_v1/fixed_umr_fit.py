#!/usr/bin/env python3
"""Run the released UMR fitter at the single predeclared formal tuple."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--official-script", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    script = known.official_script.resolve()
    if not script.is_file():
        raise RuntimeError("Missing released UMR fitter: %s" % script)
    spec = importlib.util.spec_from_file_location("released_fixed_umr_fit", str(script))
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load released UMR fitter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SEEDS = (0,)
    module.POS_WEIGHTS = (4,)
    module.TAU_GRID = (0.0,)
    module.BETA_GRID = (2.0,)
    sys.argv = [str(script)] + remaining
    module.main()


if __name__ == "__main__":
    main()
