"""Memory-efficient DISCO launcher for Qwen3-VL.

Qwen3-VL selects Flash Attention through ``attn_implementation`` when the
model is loaded, so the LLaMA monkey patch used by the original launcher must
not be applied here.
"""

import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[2]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from llava.train.train_MOE import train


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
