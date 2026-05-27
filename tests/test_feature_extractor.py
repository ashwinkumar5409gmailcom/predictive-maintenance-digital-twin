import numpy as np
from src.feature_extractor import FeatureExtractor

def test_extract_shape():
    fe = FeatureExtractor()
    data = np.random.randn(4, 1024)
    feats = fe.sliding_extract(data)
    assert feats.shape[1] == 48
