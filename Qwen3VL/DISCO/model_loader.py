"""Evaluation loader for a trained Qwen3-VL DISCO checkpoint."""

from pathlib import Path
import sys


METHOD_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(METHOD_ROOT.parent / "common"))

from qwen3vl_continual import load_continual_model  # noqa: E402


def load_model(checkpoint: str, device: str = "cuda"):
    return load_continual_model(checkpoint, METHOD_ROOT, device=device)

