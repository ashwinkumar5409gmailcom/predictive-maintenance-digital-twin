"""Training pipeline: load generated dataset, extract features, train hybrid ensemble, save models and metrics."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
import joblib

from src.config import cfg
from src.dataset_generator import generate_dataset
from src.experiment import create_run_directory, save_run_config, save_run_metrics
from src.feature_extractor import FeatureExtractor
from src.models.hybrid_model import HybridEnsemble

BASELINE_MODELS = {
    'RandomForest': RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=80, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=500, solver='saga', penalty='l2', class_weight='balanced', random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=250, random_state=42)
}


def load_raw_runs(data_dir: str):
    idx = pd.read_csv(os.path.join(data_dir, 'dataset_index.csv'))
    runs = []
    fe = FeatureExtractor(window_size=cfg.window_size, step=cfg.window_step, sample_rate=cfg.sample_rate)
    sev_map = {'mild': 1.0, 'moderate': 2.0, 'severe': 3.0}
    for _, row in idx.iterrows():
        npz = np.load(row['file'], allow_pickle=True)
        channels = npz['channels']
        meta = npz['meta'].item()
        feats = fe.sliding_extract(channels)
        if feats.shape[0] == 0:
            continue
        runs.append({
            'run_id': row.get('run_id', f"run_{len(runs)}"),
            'fault': meta.get('fault', 'healthy'),
            'severity': meta.get('severity', 'mild'),
            'X': feats,
            'y': np.full((feats.shape[0],), cfg.classes.get(meta.get('fault', 'healthy'), 0), dtype=int),
            'sev': np.full((feats.shape[0],), sev_map.get(meta.get('severity', 'mild'), 1.0), dtype=float)
        })
    return idx, runs


def validate_dataset(index_path: str) -> bool:
    if not os.path.exists(index_path):
        return False
    idx = pd.read_csv(index_path)
    if 'run_id' not in idx.columns or 'fault' not in idx.columns:
        return False
    if len(idx) < len(cfg.fault_modes):
        return False
    present = set(idx['fault'].unique())
    required = set(cfg.fault_modes)
    return required.issubset(present)


def stack_runs(run_entries):
    X = np.vstack([entry['X'] for entry in run_entries])
    y = np.concatenate([entry['y'] for entry in run_entries])
    sev = np.concatenate([entry['sev'] for entry in run_entries])
    return X, y, sev


def metrics_for_model(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    metrics = {
        'accuracy': float(np.mean(y_true == y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'matthews_corrcoef': float(matthews_corrcoef(y_true, y_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
    }
    if y_prob is not None:
        try:
            metrics['log_loss'] = float(log_loss(y_true, y_prob))
        except Exception:
            metrics['log_loss'] = None
        if y_prob.shape[1] > 1:
            from sklearn.metrics import roc_auc_score
            try:
                metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
            except Exception:
                metrics['roc_auc_ovr'] = None
    else:
        metrics['log_loss'] = None
        metrics['roc_auc_ovr'] = None
    return metrics


def train_baselines(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, run_dir: str) -> dict:
    results = {}
    baseline_dir = os.path.join(run_dir, 'baseline_models')
    os.makedirs(baseline_dir, exist_ok=True)
    for name, model in BASELINE_MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)
        metrics = metrics_for_model(name, y_val, y_pred, y_prob)
        results[name] = metrics
        joblib.dump(model, os.path.join(baseline_dir, f'{name}.joblib'))
    return results


def save_distribution_plots(y: np.ndarray, sev: np.ndarray, run_dir: str):
    label_names = [name.title().replace('_', ' ') for name in cfg.fault_modes]
    counts = pd.Series(y).map(lambda v: cfg.fault_modes[int(v)]).value_counts().reindex(cfg.fault_modes, fill_value=0)
    df = pd.DataFrame({'fault': counts.index, 'count': counts.values})
    fig, ax = plt.subplots(figsize=(9, 4), dpi=120)
    df.plot(kind='bar', x='fault', y='count', ax=ax, color='#22d3ee', legend=False)
    ax.set_title('Validation Sample Distribution by Fault Mode')
    ax.set_ylabel('Count')
    ax.set_xlabel('Fault Mode')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    path = os.path.join(run_dir, 'assets', 'class_distribution.png')
    fig.savefig(path)
    plt.close(fig)

    sev_map = {1.0: 'mild', 2.0: 'moderate', 3.0: 'severe'}
    sev_names = [sev_map.get(float(v), 'mild') for v in sev]
    sv_counts = pd.Series(sev_names).value_counts().reindex(['mild', 'moderate', 'severe'], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    sv_counts.plot(kind='bar', ax=ax, color='#f97316', legend=False)
    ax.set_title('Severity Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Severity')
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(run_dir, 'assets', 'severity_distribution.png')
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Train hybrid ensemble and baseline models for predictive maintenance.')
    parser.add_argument('--data_dir', default='data', help='Directory containing dataset_index.csv and run files.')
    parser.add_argument('--out_dir', default='outputs', help='Base directory for experiment archives.')
    parser.add_argument('--target_samples', type=int, default=50000, help='Approximate number of raw sample points to generate when dataset is missing.')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs for the hybrid ensemble.')
    parser.add_argument('--rebuild', action='store_true', help='Force regeneration of the dataset before training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset generation and splitting.')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    run_dir = create_run_directory(os.path.join(os.path.abspath(args.out_dir), 'experiments'))
    assets_dir = os.path.join(run_dir, 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    config = {
        'data_dir': data_dir,
        'out_dir': args.out_dir,
        'run_dir': run_dir,
        'target_samples': args.target_samples,
        'epochs': args.epochs,
        'sample_rate': cfg.sample_rate,
        'window_size': cfg.window_size,
        'window_step': cfg.window_step,
        'fault_modes': cfg.fault_modes,
    }
    save_run_config(config, run_dir)

    index_path = os.path.join(data_dir, 'dataset_index.csv')
    dataset_ok = validate_dataset(index_path)
    if args.rebuild or not dataset_ok:
        if args.rebuild:
            print('Rebuilding dataset from scratch:', data_dir)
        else:
            print('Existing dataset is incomplete or missing required fault coverage. Regenerating:', data_dir)
        generate_dataset(target_samples=args.target_samples, out_dir=data_dir, rng_seed=args.seed, force_rebuild=True)
    else:
        print('Found existing dataset with full fault coverage. Using existing data. To force rebuild, use --rebuild.')

    print('Loading raw runs and preparing group-aware splits...')
    idx, runs = load_raw_runs(data_dir)
    if 'run_id' not in idx.columns:
        idx['run_id'] = [f'run_{i}' for i in range(len(idx))]

    run_ids = [entry['run_id'] for entry in runs]
    if len(run_ids) < 14:
        raise RuntimeError('Dataset must contain at least 14 independent runs to avoid leakage and produce meaningful group splits.')

    run_id_to_index = {entry['run_id']: idx for idx, entry in enumerate(runs)}
    run_ids_by_fault = {}
    for entry in runs:
        run_ids_by_fault.setdefault(entry['fault'], []).append(entry['run_id'])

    rng = np.random.default_rng(args.seed)
    train_run_ids = []
    val_run_ids = []
    test_run_ids = []
    for fault in cfg.fault_modes:
        class_run_ids = run_ids_by_fault.get(fault, [])
        if len(class_run_ids) < 3:
            raise RuntimeError(f'Not enough runs for fault class {fault} to create a held-out validation and test split.')
        rng.shuffle(class_run_ids)
        n = len(class_run_ids)
        train_n = max(3, int(np.round(n * 0.50)))
        val_n = max(1, int(np.round(n * 0.25)))
        test_n = max(1, n - train_n - val_n)
        if train_n + val_n + test_n != n:
            test_n = n - train_n - val_n
        train_run_ids.extend(class_run_ids[:train_n])
        val_run_ids.extend(class_run_ids[train_n:train_n + val_n])
        test_run_ids.extend(class_run_ids[train_n + val_n:])

    train_run_idx = [run_id_to_index[run_id] for run_id in train_run_ids]
    val_run_idx = [run_id_to_index[run_id] for run_id in val_run_ids]
    test_run_idx = [run_id_to_index[run_id] for run_id in test_run_ids]

    idx['subset'] = 'train'
    idx.loc[idx['run_id'].isin(val_run_ids), 'subset'] = 'val'
    idx.loc[idx['run_id'].isin(test_run_ids), 'subset'] = 'test'
    idx.to_csv(os.path.join(data_dir, 'dataset_index.csv'), index=False)

    print('Run split summary:')
    split_counts = idx.groupby(['subset', 'fault']).size().unstack(fill_value=0)
    print(split_counts)
    print('Sample totals by split:')
    print(idx.groupby('subset')['samples'].sum().to_dict())

    X_train, y_train, sev_train = stack_runs([runs[i] for i in train_run_idx])
    X_val, y_val, sev_val = stack_runs([runs[i] for i in val_run_idx])
    X_test, y_test, sev_test = stack_runs([runs[i] for i in test_run_idx])
    print(f'Prepared train {X_train.shape[0]} windows, val {X_val.shape[0]} windows, test {X_test.shape[0]} windows.')

    scaler = FeatureExtractor(window_size=cfg.window_size, step=cfg.window_step, sample_rate=cfg.sample_rate)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, os.path.join(run_dir, 'scaler.joblib'))

    print('Training baseline models...')
    baseline_metrics = train_baselines(X_train_scaled, y_train, X_val_scaled, y_val, run_dir)
    save_run_metrics({'baseline': baseline_metrics}, run_dir, filename='baseline_metrics.json')

    print('Training hybrid ensemble...')
    model = HybridEnsemble(input_dim=X_train_scaled.shape[1])
    model.fit(X_train_scaled, y_train, severity_train=sev_train, X_val=X_val_scaled, y_val=y_val, epochs=args.epochs)
    model.save(os.path.join(run_dir, 'hybrid_model'))

    print('Evaluating hybrid model on validation set...')
    preds = model.predict(X_val_scaled)
    probs = model.predict_proba(X_val_scaled)
    metrics = metrics_for_model('HybridEnsemble', y_val, preds, probs)
    report = classification_report(y_val, preds, output_dict=True, zero_division=0)
    save_run_metrics({'hybrid': metrics}, run_dir, filename='model_metrics.json')
    with open(os.path.join(run_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(6, 6), dpi=120)
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Hybrid Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'assets', 'confusion_matrix.png'))
    plt.close()

    save_distribution_plots(y_val, sev_val, run_dir)

    summary = {
        'run_dir': run_dir,
        'trained_samples': int(X_train.shape[0]),
        'validation_samples': int(X_val.shape[0]),
        'hybrid_metrics': metrics,
        'baseline_metrics': baseline_metrics,
    }
    save_run_metrics(summary, run_dir, filename='summary.json')
    print('Training complete. Experiment archived at:', run_dir)


if __name__ == '__main__':
    main()
