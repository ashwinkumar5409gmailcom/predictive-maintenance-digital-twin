"""Evaluation script to compute metrics on a dataset using saved model and scaler."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

from src.config import cfg
from src.experiment import get_latest_run
from src.feature_extractor import FeatureExtractor
from src.models.hybrid_model import HybridEnsemble


def load_features_from_index(data_dir: str, subset: str = 'test'):
    idx = pd.read_csv(os.path.join(data_dir, 'dataset_index.csv'))
    if subset != 'all' and 'subset' in idx.columns:
        subset_idx = idx[idx['subset'] == subset]
        if subset_idx.empty:
            print(f'Warning: requested subset "{subset}" is empty or not present. Falling back to full dataset.')
            subset_idx = idx
    else:
        subset_idx = idx

    X_list = []
    y_list = []
    for _, row in subset_idx.iterrows():
        npz = np.load(row['file'], allow_pickle=True)
        channels = npz['channels']
        meta = npz['meta'].item()
        fe = FeatureExtractor(window_size=cfg.window_size, step=cfg.window_step, sample_rate=cfg.sample_rate)
        feats = fe.sliding_extract(channels)
        if feats.shape[0] == 0:
            continue
        X_list.append(feats)
        y = np.full((feats.shape[0],), cfg.classes.get(meta['fault'], 0), dtype=int)
        y_list.append(y)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'matthews_corrcoef': float(matthews_corrcoef(y_true, y_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred))
    }
    if y_prob is not None:
        try:
            metrics['log_loss'] = float(log_loss(y_true, y_prob))
        except Exception:
            metrics['log_loss'] = None
        if y_prob.shape[1] > 1:
            try:
                metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
            except Exception:
                metrics['roc_auc_ovr'] = None
    return metrics


def plot_confusion_matrix(cm: np.ndarray, labels: list, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), fmt), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, labels: list, out_path: str):
    if y_prob is None or y_prob.shape[1] <= 1:
        return
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    classes = np.arange(y_prob.shape[1])
    y_binarized = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    for idx, label_text in enumerate(labels):
        if idx >= y_prob.shape[1]:
            break
        fpr, tpr, _ = roc_curve(y_binarized[:, idx], y_prob[:, idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{label_text} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multiclass ROC Curves')
    ax.legend(loc='lower right', fontsize='small')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Evaluate saved model and baseline comparisons.')
    parser.add_argument('--data_dir', default='data', help='Path to data directory containing dataset_index.csv.')
    parser.add_argument('--model_dir', default=None, help='Path to experiment or model directory to evaluate.')
    parser.add_argument('--results_dir', default=None, help='Optional directory for evaluation outputs.')
    parser.add_argument('--subset', default='test', choices=['train', 'val', 'test', 'all'], help='Subset to evaluate.')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if args.model_dir:
        model_dir = os.path.abspath(args.model_dir)
    else:
        model_dir = get_latest_run()
    if model_dir is None:
        raise RuntimeError('No experiment directory found. Run train.py first or specify --model_dir.')
    results_dir = os.path.abspath(args.results_dir) if args.results_dir else model_dir
    ensure_dir(results_dir)

    print('Loading data from', data_dir, 'subset=', args.subset)
    X, y = load_features_from_index(data_dir, subset=args.subset)
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    model = HybridEnsemble()
    model.load(os.path.join(model_dir, 'hybrid_model'))
    Xs = scaler.transform(X)

    probs = model.predict_proba(Xs)
    preds = np.argmax(probs, axis=1)
    metrics = evaluate(y, preds, probs)
    metrics['num_samples'] = int(len(y))
    metrics['model_dir'] = model_dir
    metrics['subset'] = args.subset
    with open(os.path.join(results_dir, 'eval_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    labels = sorted(np.unique(y))
    label_names = [cfg.fault_modes[int(label)].title().replace('_', ' ') for label in labels]
    cm = confusion_matrix(y, preds, labels=labels)
    plot_confusion_matrix(cm, label_names, os.path.join(results_dir, 'confusion_matrix_eval.png'))
    plot_roc(y, probs, label_names, os.path.join(results_dir, 'roc_curve_eval.png'))
    report = classification_report(y, preds, labels=labels, target_names=label_names, zero_division=0)
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    baseline_path = os.path.join(model_dir, 'baseline_models')
    baseline_summary = {}
    if os.path.isdir(baseline_path):
        for path in Path(baseline_path).glob('*.joblib'):
            baseline = joblib.load(path)
            y_pred = baseline.predict(Xs)
            y_prob = baseline.predict_proba(Xs) if hasattr(baseline, 'predict_proba') else None
            baseline_summary[path.stem] = evaluate(y, y_pred, y_prob)
        with open(os.path.join(results_dir, 'baseline_comparison.json'), 'w') as f:
            json.dump(baseline_summary, f, indent=2)

    comparison_rows = []
    comparison_rows.append({'model': 'Hybrid', **{k: metrics.get(k) for k in ['accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'cohen_kappa', 'log_loss', 'roc_auc_ovr']}})
    for name, baseline_metrics in baseline_summary.items():
        comparison_rows.append({'model': name, **baseline_metrics})
    pd.DataFrame(comparison_rows).to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)

    print('Evaluation complete. Results written to', results_dir)


if __name__ == '__main__':
    main()
