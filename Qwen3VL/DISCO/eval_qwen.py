#!/usr/bin/env python3
"""Qwen3-VL DISCO UCIT evaluation entry point."""

from pathlib import Path
import sys


METHOD_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(METHOD_ROOT.parent / "common"))

from qwen3vl_continual import eval_main  # noqa: E402


if __name__ == "__main__":
    eval_main(METHOD_ROOT)

