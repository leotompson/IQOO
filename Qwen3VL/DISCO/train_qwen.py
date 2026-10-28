#!/usr/bin/env python3
"""Qwen3-VL DISCO continual instruction tuning entry point."""

from pathlib import Path
import sys


METHOD_ROOT = Path(__file__).resolve().parent
COMMON_ROOT = METHOD_ROOT.parent / "common"
sys.path.insert(0, str(COMMON_ROOT))

from qwen3vl_continual import run  # noqa: E402


if __name__ == "__main__":
    run("disco", METHOD_ROOT)

