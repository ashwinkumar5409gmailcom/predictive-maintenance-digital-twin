# Predictive Maintenance System

This repository implements a complete end-to-end real-time predictive maintenance system for mechatronic equipment. The implementation follows the IEEE paper specification to deliver a physics-based digital twin, multi-channel telemetry, feature extraction, hybrid AI ensemble, fusion, confidence estimation, RUL modeling, real-time ingestion, dashboarding, and evaluation.

## Project Structure

- `src/`
  - `digital_twin.py` — physics-based simulator producing vibration, current, temperature, and RPM channels
  - `dataset_generator.py` — creates synthetic dataset to reach ~49,788 samples
  - `feature_extractor.py` — sliding window feature extraction (48 features)
  - `models/hybrid_model.py` — ensemble of physics-informed MLP, relational GNN-style branch, and Random Forest
  - `fusion.py` — weighted probability fusion and entropy-based confidence
  - `rul.py` — trend-based remaining useful life estimation with bootstrap confidence
  - `storage.py` — SQLite prediction logging and latest telemetry persistence
  - `config.py` — central configuration values
  - `utils.py` — helper utilities
- `train.py` — training pipeline with metrics output
- `evaluate.py` — evaluation pipeline and confusion matrix export
- `inference.py` — batch inference script
- `stream_demo.py` — realtime producer/consumer demo with telemetry logging
- `dashboard/app.py` — Streamlit dashboard for live monitoring
- `outputs/` — saved models, metrics, and runtime outputs
- `data/` — generated dataset files
- `tests/` — unit tests

## Installation

Create a Python 3.10+ virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Generate dataset and train

```bash
python train.py
```

This command will generate the dataset if it is missing, extract features, train the hybrid ensemble, save the scaler and model, and export `outputs/metrics.json` and `outputs/confusion_matrix.png`.

### Evaluate model

```bash
python evaluate.py
```

Generates `outputs/eval_metrics.json`, `outputs/classification_report.txt`, and `outputs/confusion_matrix_eval.png`.

### Run inference

```bash
python inference.py
```

Produces `outputs/predictions.csv` with window-level predictions.

### Start realtime demo

```bash
python stream_demo.py
```

This demo starts a producer thread simulating telemetry, an inference consumer, and logs results to SQLite as well as latest telemetry data.

### Start dashboard

```bash
streamlit run dashboard/app.py
```

Open the Streamlit UI in your browser to view confidence, prediction history, RUL trends, and live waveform telemetry.

## Notes

- The system supports fault modes: `healthy`, `bearing_wear`, `misalignment`, and `overload`.
- Severity levels are `mild`, `moderate`, and `severe`.
- The dashboard reads live telemetry and prediction records from `outputs/`.
- The ensemble uses validation-optimized fusion weights to combine branch outputs.

## Testing

Run the included unit test:

```bash
python3 -m pytest -q tests/test_feature_extractor.py
```
