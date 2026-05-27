"""Experiment tracking utilities for reproducibility, configuration logging, and run directories."""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def create_run_directory(base_dir: str = None) -> str:
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'experiments')
    os.makedirs(base_dir, exist_ok=True)
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.abspath(os.path.join(base_dir, run_id))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_config(config: Dict[str, Any], run_dir: str, filename: str = 'config.json') -> None:
    path = os.path.join(run_dir, filename)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def save_run_metrics(metrics: Dict[str, Any], run_dir: str, filename: str = 'metrics.json') -> None:
    path = os.path.join(run_dir, filename)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


def get_latest_run(base_dir: str = None) -> Optional[str]:
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'experiments')
    if not os.path.exists(base_dir):
        return None
    entries = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
    runs = [d for d in entries if os.path.isdir(d)]
    if not runs:
        return None
    return max(runs)
