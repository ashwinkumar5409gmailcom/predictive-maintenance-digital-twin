import numpy as np
import time
import logging
from typing import Tuple, List

logger = logging.getLogger("predmaint")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def timestamp() -> float:
    return time.time()


def ensure_rng(seed: int):
    np.random.seed(seed)


def sliding_windows(data: np.ndarray, window_size: int, step: int) -> np.ndarray:
    n = data.shape[-1]
    if n < window_size:
        return np.empty((0, data.shape[0], window_size))
    count = 1 + (n - window_size) // step
    out = np.empty((count, data.shape[0], window_size))
    for i in range(count):
        s = i * step
        out[i] = data[:, s:s+window_size]
    return out
