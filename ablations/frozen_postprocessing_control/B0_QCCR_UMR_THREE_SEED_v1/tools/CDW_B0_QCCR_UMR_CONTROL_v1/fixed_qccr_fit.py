#!/usr/bin/env python3
"""Run the released QCCR fitter at the single predeclared regularization."""

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
        raise RuntimeError("Missing released QCCR fitter: %s" % script)
    spec = importlib.util.spec_from_file_location("released_fixed_qccr_fit", str(script))
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load released QCCR fitter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LAMBDA_REG_VALUES = (1.0e-3,)
    sys.argv = [str(script)] + remaining
    module.main()


if __name__ == "__main__":
    main()
