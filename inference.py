"""Batch inference script: load saved model and run on dataset producing predictions.csv"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from src.config import cfg
from src.experiment import get_latest_run
from src.feature_extractor import FeatureExtractor
from src.models.hybrid_model import HybridEnsemble


def main(data_dir: str = 'data', model_dir: str = 'outputs', output_name: str = 'predictions.csv'):
    data_dir = os.path.abspath(data_dir)
    model_dir = os.path.abspath(model_dir)
    if model_dir.endswith(os.path.sep) or model_dir == os.path.abspath('outputs'):
        latest_run = get_latest_run()
        if latest_run is not None:
            model_dir = latest_run
    out_path = os.path.join(model_dir, output_name)
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    model = HybridEnsemble()
    model.load(os.path.join(model_dir, 'hybrid_model'))

    idx = pd.read_csv(os.path.join(data_dir, 'dataset_index.csv'))
    rows = []
    for _, row in idx.iterrows():
        npz = np.load(row['file'], allow_pickle=True)
        channels = npz['channels']
        meta = npz['meta'].item()
        fe = FeatureExtractor(window_size=cfg.window_size, step=cfg.window_step, sample_rate=cfg.sample_rate)
        feats = fe.sliding_extract(channels)
        if feats.shape[0] == 0:
            continue
        Xs = scaler.transform(feats)
        probs = model.predict_proba(Xs)
        preds = np.argmax(probs, axis=1)
        for i in range(len(preds)):
            rows.append({
                'file': row['file'],
                'window_idx': int(i),
                'prediction': int(preds[i]),
                'confidence': float(np.max(probs[i])),
                'fault_mode': cfg.fault_modes[int(preds[i])],
                'meta_fault': meta.get('fault', 'unknown'),
                'severity': meta.get('severity', 'unknown')
            })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    summary = {
        'dataset': data_dir,
        'model_dir': model_dir,
        'rows': len(rows)
    }
    with open(os.path.join(model_dir, 'predictions_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Saved predictions to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run batch inference on the saved hybrid ensemble.')
    parser.add_argument('--data_dir', default='data', help='Data directory containing dataset_index.csv')
    parser.add_argument('--model_dir', default='outputs', help='Directory containing saved model artifacts')
    parser.add_argument('--output_name', default='predictions.csv', help='Output CSV file name')
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.output_name)
