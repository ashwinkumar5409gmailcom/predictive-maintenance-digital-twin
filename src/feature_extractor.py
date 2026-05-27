"""Sliding-window feature extraction engine.
Window: 256 samples; step: 32 (87.5% overlap).
Produces a richer feature vector per window with time, frequency, and cross-channel descriptors.
Includes training-set normalization support.
"""
from typing import Tuple, List
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq

class FeatureExtractor:
    def __init__(self, window_size: int = 256, step: int = 32, sample_rate: int = 2048):
        self.window_size = window_size
        self.step = step
        self.sample_rate = sample_rate
        self.mean_ = None
        self.std_ = None
        self.feature_dim = self._estimate_feature_dim()

    def _estimate_feature_dim(self) -> int:
        dummy = np.zeros(self.window_size, dtype=float)
        time_len = len(self._time_features(dummy))
        freq_len = len(self._freq_features(dummy))
        pairwise_len = 2
        channel_pairs = 6
        return 4 * (time_len + freq_len) + channel_pairs * pairwise_len

    def _time_features(self, x: np.ndarray) -> List[float]:
        mean_v = np.mean(x)
        var_v = np.var(x)
        std_v = np.std(x)
        skew_v = float(skew(x))
        kurt_v = float(kurtosis(x))
        rms_v = np.sqrt(np.mean(x**2))
        peak_v = np.max(np.abs(x))
        crest = peak_v / (rms_v + 1e-9)
        mean_abs = np.mean(np.abs(x))
        median_v = float(np.median(x))
        q75, q25 = np.percentile(x, [75, 25])
        iqr = float(q75 - q25)
        impulse = peak_v / (mean_abs + 1e-9)
        clearance = peak_v / (np.mean(np.sqrt(np.abs(x))) + 1e-9)
        shape = rms_v / (mean_abs + 1e-9)
        return [mean_v, var_v, std_v, skew_v, kurt_v, rms_v, peak_v, crest,
                mean_abs, median_v, iqr, impulse, clearance, shape]

    def _freq_features(self, x: np.ndarray) -> List[float]:
        X = np.abs(rfft(x))
        freqs = rfftfreq(len(x), 1.0 / self.sample_rate)
        ps = X**2
        ps_sum = np.sum(ps) + 1e-12
        centroid = float(np.sum(freqs * ps) / ps_sum)
        dom_idx = np.argmax(ps)
        dom_freq = float(freqs[dom_idx])
        ps_norm = ps / ps_sum
        spec_entropy = -float(np.sum(ps_norm * np.log(ps_norm + 1e-12)))
        bandwidth = float(np.sqrt(np.sum(ps * (freqs - centroid) ** 2) / ps_sum))
        cumulative = np.cumsum(ps)
        rolloff_idx = np.searchsorted(cumulative, 0.85 * ps_sum)
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
        flatness = float(np.exp(np.mean(np.log(ps + 1e-12))) / (np.mean(ps) + 1e-12))
        top_two = np.sort(ps)[-2:]
        top2_ratio = float(np.sum(top_two) / ps_sum)
        return [centroid, dom_freq, spec_entropy, bandwidth, rolloff, flatness, top2_ratio]

    def _pairwise_features(self, a: np.ndarray, b: np.ndarray) -> List[float]:
        if np.std(a) == 0 or np.std(b) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
        rms_a = np.sqrt(np.mean(a**2))
        rms_b = np.sqrt(np.mean(b**2))
        rms_ratio = float(rms_a / (rms_b + 1e-9))
        return [corr, rms_ratio]

    def extract_from_window(self, window: np.ndarray) -> np.ndarray:
        feats = []
        channels = window.shape[0]
        for c in range(channels):
            x = window[c]
            feats.extend(self._time_features(x))
            feats.extend(self._freq_features(x))

        # cross-channel pairwise descriptors for all unique pairs
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for i, j in pairs:
            feats.extend(self._pairwise_features(window[i], window[j]))

        return np.array(feats, dtype=float)

    def sliding_extract(self, channels: np.ndarray) -> np.ndarray:
        n = channels.shape[1]
        if n < self.window_size:
            return np.empty((0, self.feature_dim))
        count = 1 + (n - self.window_size) // self.step
        out = np.zeros((count, self.feature_dim), dtype=float)
        for i in range(count):
            s = i * self.step
            win = channels[:, s:s + self.window_size]
            out[i] = self.extract_from_window(win)
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-9
        return (X - self.mean_) / self.std_

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted. Call fit_transform on training data first.")
        return (X - self.mean_) / self.std_


if __name__ == "__main__":
    fe = FeatureExtractor()
    data = np.random.randn(4, 2048)
    feats = fe.sliding_extract(data)
    print(feats.shape)
