"""Realtime streaming demo: producer -> queue -> sliding window -> inference -> logging.
Runs locally and demonstrates near-realtime flow with database tracking and telemetry persistence.
"""
import argparse
import os
import threading
import queue
import time
import numpy as np
import joblib
from src.config import cfg
from src.digital_twin import simulate
from src.experiment import get_latest_run
from src.feature_extractor import FeatureExtractor
from src.fusion import entropy_confidence, confidence_level
from src.models.hybrid_model import HybridEnsemble
from src.rul import RULTracker
from src.storage import init_db, log_prediction, save_latest_telemetry


def producer(out_q: queue.Queue, run_event: threading.Event, chunk_seconds: float = 1.0):
    sr = cfg.sample_rate
    while run_event.is_set():
        channels, meta = simulate(
            chunk_seconds,
            sr,
            fault=np.random.choice(cfg.fault_modes),
            severity=np.random.choice(['mild', 'moderate', 'severe']),
            speed_rpm=np.random.choice([1000, 1400, 1800, 2200]),
            load_fraction=np.random.choice([0.2, 0.45, 0.65, 0.85]),
            rng_seed=np.random.randint(0, 1_000_000)
        )
        out_q.put((channels, meta))
        time.sleep(chunk_seconds * 0.1)


def consumer(in_q: queue.Queue, model_dir: str, run_event: threading.Event):
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    model = HybridEnsemble()
    model.load(os.path.join(model_dir, 'hybrid_model'))
    fe = FeatureExtractor(window_size=cfg.window_size, step=cfg.window_step, sample_rate=cfg.sample_rate)
    rul = RULTracker(history_len=200)

    while run_event.is_set() or not in_q.empty():
        try:
            channels, meta = in_q.get(timeout=1.0)
        except queue.Empty:
            continue
        feats = fe.sliding_extract(channels)
        if feats.shape[0] == 0:
            continue
        Xs = scaler.transform(feats)
        probs = model.predict_proba(Xs)
        preds = np.argmax(probs, axis=1)
        confs = entropy_confidence(probs)
        pred_mode = int(np.bincount(preds).argmax())
        conf_mean = float(np.mean(confs))
        sev_map = {'mild': 1.0, 'moderate': 2.0, 'severe': 3.0}
        sev = sev_map.get(meta.get('severity', 'mild'), 1.0)
        rul.push(sev)
        est_rul, low, high = rul.estimate_rul()
        ts = time.time()
        log_prediction(ts, pred_mode, conf_mean, sev, est_rul)
        save_latest_telemetry(channels, meta, ts)
        level = confidence_level(conf_mean)
        print(f"[{ts:.1f}] Pred:{pred_mode} Conf:{conf_mean:.3f} Level:{level} RUL:{est_rul:.1f}")


def main(model_dir: str = 'outputs', duration: float = 20.0, queue_size: int = 10):
    init_db()
    if model_dir == 'outputs' or model_dir.endswith(os.path.sep):
        latest_run = get_latest_run()
        if latest_run is not None:
            model_dir = latest_run
    q = queue.Queue(maxsize=queue_size)
    run_event = threading.Event()
    run_event.set()

    producer_thread = threading.Thread(target=producer, args=(q, run_event, 1.0), daemon=True)
    consumer_thread = threading.Thread(target=consumer, args=(q, model_dir, run_event), daemon=True)
    producer_thread.start()
    consumer_thread.start()

    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        pass
    run_event.clear()
    producer_thread.join()
    consumer_thread.join()
    print('Demo finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a realtime predictive maintenance streaming demo.')
    parser.add_argument('--model_dir', default='outputs', help='Directory containing saved model artifacts.')
    parser.add_argument('--duration', type=float, default=20.0, help='Total runtime in seconds.')
    parser.add_argument('--queue_size', type=int, default=10, help='Maximum queued chunks between producer and consumer.')
    args = parser.parse_args()
    main(model_dir=args.model_dir, duration=args.duration, queue_size=args.queue_size)
