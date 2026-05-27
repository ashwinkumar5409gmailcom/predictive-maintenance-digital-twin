Detailed README for Predictive Maintenance Project

Installation

1. Create a Python 3.10+ virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Generating dataset

```bash
python -m src.dataset_generator
# or from project root
python train.py
```

Training

```bash
python train.py
```

Evaluation

```bash
python evaluate.py
```

Realtime demo

```bash
python stream_demo.py
```

Dashboard

```bash
streamlit run dashboard/app.py
```

Files

- `src/digital_twin.py` — physics-based simulator
- `src/feature_extractor.py` — sliding window features
- `src/models/hybrid_model.py` — ensemble model
- `stream_demo.py` — realtime demo
- `dashboard/app.py` — Streamlit dashboard

