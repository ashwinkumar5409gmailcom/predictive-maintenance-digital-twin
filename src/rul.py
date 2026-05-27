"""Remaining Useful Life (RUL) estimation using robust trend extraction and bootstrap confidence bands."""
import numpy as np
from scipy.stats import theilslopes
from typing import Optional, Tuple


class RULTracker:
    def __init__(self, history_len: int = 100):
        self.history_len = history_len
        self.history = []

    def push(self, metric: float):
        self.history.append(float(metric))
        if len(self.history) > self.history_len:
            self.history.pop(0)

    @staticmethod
    def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
        if len(values) < window:
            return values
        kernel = np.ones(window) / window
        padded = np.pad(values, (window - 1, 0), mode='edge')
        smooth = np.convolve(padded, kernel, mode='valid')
        return smooth[: len(values)]

    def estimate_rul(self, safety_factor: float = 1.15, min_slope: float = 1e-3, bootstrap_iters: int = 200) -> Tuple[float, float, float]:
        if len(self.history) < 8:
            return float('inf'), 0.0, 1.0
        y = np.asarray(self.history, dtype=float)
        y = self._smooth(y, window=min(7, len(y)))
        x = np.arange(len(y), dtype=float)

        try:
            slope, intercept, lo, hi = theilslopes(y, x)
        except Exception:
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs[0], coeffs[1]
            lo, hi = slope, slope

        if slope <= min_slope:
            return float('inf'), float(lo), float(hi)

        threshold = y[-1] * safety_factor + 1e-9
        t_to_threshold = (threshold - intercept) / slope
        rul = max(0.0, float(t_to_threshold - (len(y) - 1)))

        slopes = []
        for _ in range(bootstrap_iters):
            indices = np.random.choice(len(y), size=len(y), replace=True)
            x_b = x[indices]
            y_b = y[indices]
            try:
                result = np.polyfit(x_b, y_b, 1)
                slopes.append(float(result[0]))
            except Exception:
                continue
        if len(slopes) == 0:
            return rul, float(lo), float(hi)

        low = float(np.percentile(slopes, 2.5))
        high = float(np.percentile(slopes, 97.5))
        return rul, low, high


if __name__ == "__main__":
    tracker = RULTracker(history_len=150)
    for i in range(120):
        tracker.push(1.0 + 0.02 * i + 0.1 * np.random.randn())
    print(tracker.estimate_rul())
