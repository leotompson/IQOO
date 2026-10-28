#!/usr/bin/env python3
"""Compatibility entry point; the DISCO main function lives in train_MOE.py."""

from llava.train.train_MOE import train


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
