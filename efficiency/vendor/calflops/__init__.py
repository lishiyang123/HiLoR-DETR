"""Ordinary PyTorch FLOP calculator used by the v2 benchmark."""
from .flops_counter import calculate_flops
__all__ = ["calculate_flops"]
