"""Fusion layer and confidence estimation utilities."""
import numpy as np
from typing import Tuple
from scipy.stats import entropy


def fuse_probabilities(probs_list: list, weights: np.ndarray) -> np.ndarray:
    # probs_list: list of (N,C) arrays
    stacked = np.stack(probs_list, axis=0)
    # weights shape (k,)
    w = weights.reshape(-1,1,1)
    fused = np.sum(w * stacked, axis=0)
    # renormalize
    fused = fused / (fused.sum(axis=1, keepdims=True) + 1e-12)
    return fused


def entropy_confidence(probs: np.ndarray) -> np.ndarray:
    # probs shape (N,C)
    probs = np.asarray(probs, dtype=float)
    ent = entropy(probs.T)
    # Normalize to [0,1] by max entropy log(C)
    max_ent = np.log(probs.shape[1])
    conf = 1.0 - (ent / (max_ent + 1e-12))
    return conf


def confidence_level(conf: float, advisory_thresh: float = 0.6, warning_thresh: float = 0.4) -> str:
    # conf in [0,1]
    if conf >= advisory_thresh:
        return "NORMAL"
    if conf >= warning_thresh:
        return "ADVISORY"
    return "WARNING"
