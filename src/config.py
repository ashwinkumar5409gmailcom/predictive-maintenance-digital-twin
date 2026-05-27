from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Config:
    sample_rate: int = 2048
    window_size: int = 256
    window_step: int = 32  # 87.5% overlap -> step = 256*(1-0.875)=32
    sensors: int = 4
    rng_seed: int = 42
    fault_modes: List[str] = None
    class_names: List[str] = None
    severity_levels: Dict[str, int] = None

cfg = Config()
cfg.fault_modes = [
    "healthy",
    "bearing_wear",
    "misalignment",
    "overload",
    "imbalance",
    "looseness",
    "multi_fault"
]
cfg.class_names = [
    "Healthy",
    "Bearing Wear",
    "Misalignment",
    "Overload",
    "Imbalance",
    "Looseness",
    "Multi-Fault"
]
cfg.classes = {name: idx for idx, name in enumerate(cfg.fault_modes)}
cfg.severity_levels = {"mild": 1, "moderate": 2, "severe": 3}
