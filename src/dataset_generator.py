"""Generate a synthetic dataset using the digital twin simulator.
Produces balanced runs across fault types, severities, speeds and loads.
Writes raw telemetry files, dataset index, distribution plots, and statistics.
"""
from typing import Optional
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from .digital_twin import simulate
from .config import cfg


def generate_dataset(target_samples: int = 50000, out_dir: str = None, stats_dir: Optional[str] = None, rng_seed: int = 42, force_rebuild: bool = False):
    rng = np.random.default_rng(rng_seed)
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if stats_dir is None:
        stats_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    if force_rebuild and os.path.exists(out_dir):
        for fname in os.listdir(out_dir):
            if fname.startswith('run_') and fname.endswith('.npz'):
                os.remove(os.path.join(out_dir, fname))
        index_path = os.path.join(out_dir, 'dataset_index.csv')
        if os.path.exists(index_path):
            os.remove(index_path)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    class_names = cfg.fault_modes
    severities = ['mild', 'moderate', 'severe']
    speeds = [1000, 1400, 1800, 2200]
    loads = [0.2, 0.45, 0.65, 0.85]
    duration = 5.0
    samples_per_run = int(duration * cfg.sample_rate)
    target_runs = max(len(class_names) * 6, int(np.ceil(target_samples / samples_per_run)))
    runs_per_class = int(np.ceil(target_runs / len(class_names)))
    if runs_per_class % len(severities) != 0:
        runs_per_class += len(severities) - (runs_per_class % len(severities))

    total_runs = runs_per_class * len(class_names)
    speed_schedule = np.tile(speeds, int(np.ceil(total_runs / len(speeds))))[:total_runs]
    load_schedule = np.tile(loads, int(np.ceil(total_runs / len(loads))))[:total_runs]
    rng.shuffle(speed_schedule)
    rng.shuffle(load_schedule)

    records = []
    file_idx = 0
    for class_idx, fault in enumerate(class_names):
        for class_run_idx in range(runs_per_class):
            if fault == 'healthy':
                severity = 'mild'
            else:
                severity = severities[class_run_idx % len(severities)]
            speed = float(speed_schedule[file_idx] * (1.0 + rng.uniform(-0.08, 0.08)))
            load = float(np.clip(load_schedule[file_idx] + rng.uniform(-0.05, 0.05), 0.05, 0.95))
            ambient_temp = float(20.0 + 10.0 * rng.random())
            channels, meta = simulate(duration,
                                      cfg.sample_rate,
                                      fault=fault,
                                      severity=severity,
                                      speed_rpm=speed,
                                      load_fraction=load,
                                      ambient_temp=ambient_temp,
                                      rng_seed=int(rng.integers(0, 1_000_000)))
            fname = os.path.join(out_dir, f"run_{file_idx}.npz")
            np.savez_compressed(fname, channels=channels, meta=meta)
            records.append({
                'run_id': f'run_{file_idx}',
                'file': fname,
                'samples': channels.shape[1],
                **meta
            })
            file_idx += 1

    if sum(r['samples'] for r in records) < target_samples:
        while sum(r['samples'] for r in records) < target_samples:
            fault = rng.choice(class_names)
            severity = 'mild' if fault == 'healthy' else rng.choice(severities, p=[0.35, 0.35, 0.30])
            speed = float(rng.choice(speeds) * (1.0 + rng.uniform(-0.08, 0.08)))
            load = float(np.clip(rng.choice(loads) + rng.uniform(-0.05, 0.05), 0.05, 0.95))
            ambient_temp = float(20.0 + 10.0 * rng.random())
            channels, meta = simulate(duration,
                                      cfg.sample_rate,
                                      fault=fault,
                                      severity=severity,
                                      speed_rpm=speed,
                                      load_fraction=load,
                                      ambient_temp=ambient_temp,
                                      rng_seed=int(rng.integers(0, 1_000_000)))
            fname = os.path.join(out_dir, f"run_{file_idx}.npz")
            np.savez_compressed(fname, channels=channels, meta=meta)
            records.append({
                'run_id': f'run_{file_idx}',
                'file': fname,
                'samples': channels.shape[1],
                **meta
            })
            file_idx += 1

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, 'dataset_index.csv'), index=False)

    stats = df.groupby('fault').agg(run_count=('file', 'count'), samples=('samples', 'sum')).reset_index()
    stats['proportion'] = stats['samples'] / stats['samples'].sum()
    stats.to_csv(os.path.join(stats_dir, 'dataset_statistics.csv'), index=False)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.bar(stats['fault'], stats['samples'], color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Dataset Sample Distribution by Fault Mode')
    plt.ylabel('Raw Sample Count')
    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, 'dataset_distribution.png'), dpi=300)
    plt.close()

    print(f"Generated {df['samples'].sum()} raw samples across {len(df)} runs. Index saved to {out_dir}")
    print(f"Dataset statistics saved to {stats_dir}")


if __name__ == '__main__':
    generate_dataset()
