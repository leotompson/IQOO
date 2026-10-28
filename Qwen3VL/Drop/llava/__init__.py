"""Drop package namespace for the Qwen3-VL port.

Legacy LLaVA evaluators remain available, but importing ``llava`` must not
eagerly load the old LLaMA multimodal model.
"""
