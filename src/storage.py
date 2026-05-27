"""Simple SQLite storage for predictions, timestamps, confidence, severity, and RUL."""
import sqlite3
import os
import numpy as np
from typing import Optional

DB_NAME = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'predictions.db')


def init_db(db_path: Optional[str] = None):
    if db_path is None:
        db_path = DB_NAME
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 ts REAL,
                 prediction INTEGER,
                 confidence REAL,
                 severity REAL,
                 rul REAL
                 )''')
    conn.commit()
    conn.close()


def log_prediction(ts: float, prediction: int, confidence: float, severity: float, rul: float, db_path: Optional[str] = None):
    if db_path is None:
        db_path = DB_NAME
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('INSERT INTO predictions (ts, prediction, confidence, severity, rul) VALUES (?,?,?,?,?)', (ts, prediction, confidence, severity, rul))
    conn.commit()
    conn.close()


def save_latest_telemetry(channels, meta, timestamp: float, out_path: Optional[str] = None):
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'latest_telemetry.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, channels=channels, meta=meta, ts=timestamp)

if __name__ == "__main__":
    init_db()
    log_prediction(0.0, 1, 0.8, 2.0, 100.0)
    save_latest_telemetry(np.zeros((4,2048)), {'fault':'healthy','severity':'mild'}, 0.0)
    print('Logged example')
