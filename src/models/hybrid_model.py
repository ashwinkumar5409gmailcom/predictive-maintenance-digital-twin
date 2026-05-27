"""Hybrid AI ensemble: Physics-Informed Feature Branch (PIFB), Relational Modeling Branch (RMB), and Random Forest branch.
Also includes dynamic fusion, calibration, and branch confidence awareness.
"""
from typing import Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted
import joblib
from torch.utils.data import DataLoader, TensorDataset


class PIFB(nn.Module):
    """Physics-Informed Feature Branch: MLP with auxiliary severity head and robust feature encoding."""
    def __init__(self, input_dim: int = 96, physics_idx: Optional[list] = None, n_classes: int = 7):
        super().__init__()
        self.input_dim = input_dim
        self.physics_idx = physics_idx
        in_dim = len(self.physics_idx) if self.physics_idx is not None else input_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.classifier = nn.Linear(32, n_classes)
        self.severity_head = nn.Linear(32, 1)

    def forward(self, x):
        if self.physics_idx is not None:
            x = x[:, self.physics_idx]
        h = self.net(x)
        logits = self.classifier(h)
        sev = self.severity_head(h).squeeze(-1)
        return logits, sev


class SimpleRMB(nn.Module):
    """Relational Modeling Branch: GNN-style message passing over the 4 sensor channels."""
    def __init__(self, input_dim: int = 98, n_classes: int = 7):
        super().__init__()
        node_feat = input_dim // 4
        self.node_in = nn.Linear(node_feat, 32)
        self.msg = nn.Linear(32, 32)
        self.update = nn.GRUCell(32, 32)
        self.pool = nn.Linear(32, 32)
        self.classifier = nn.Linear(32, n_classes)
        self.register_buffer('adj', torch.tensor([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]], dtype=torch.float32))

    def forward(self, x):
        batch = x.shape[0]
        node_feat = x.view(batch, 4, -1)
        h = torch.relu(self.node_in(node_feat))
        for _ in range(3):
            h_nodes = h
            msgs = torch.matmul(self.adj, h_nodes)
            msgs = torch.relu(self.msg(msgs))
            h_flat = h.view(-1, 32)
            msg_flat = msgs.view(-1, 32)
            h = (h_flat + self.update(msg_flat, h_flat)).view(batch, 4, 32)
        pooled = h.mean(dim=1)
        out = self.classifier(torch.relu(self.pool(pooled)))
        return out


class HybridEnsemble:
    def __init__(self, input_dim: int = 96, device: str = "cpu"):
        self.device = device
        self.pifb = PIFB(input_dim=input_dim)
        self.rmb = SimpleRMB(input_dim=input_dim)
        self.rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        self.pifb.to(self.device)
        self.rmb.to(self.device)
        self.pifb_opt = optim.Adam(self.pifb.parameters(), lr=1e-3, weight_decay=1e-5)
        self.rmb_opt = optim.Adam(self.rmb.parameters(), lr=1e-3, weight_decay=1e-5)
        self.branch_accuracy = np.ones(3, dtype=float) / 3.0
        self.temperature = 1.0
        self.meta_fusion = None

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)

    @staticmethod
    def _sample_confidence(probs: np.ndarray) -> np.ndarray:
        return np.max(probs, axis=1)

    def _rf_fitted(self) -> bool:
        try:
            check_is_fitted(self.rf)
            return True
        except Exception:
            return False

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        logits = np.log(probs + 1e-12)
        scaled = logits / np.clip(self.temperature, 1e-3, 10.0)
        return self._softmax(scaled)

    def calibrate_temperature(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        fused_probs = self.predict_proba(X_val)
        labels = np.eye(fused_probs.shape[1])[y_val]

        def nll(temp):
            temp = float(np.clip(temp[0], 0.1, 10.0))
            logits = np.log(fused_probs + 1e-12) / temp
            scaled = self._softmax(logits)
            return -np.mean(np.sum(labels * np.log(scaled + 1e-12), axis=1))

        result = minimize(nll, x0=[1.0], bounds=[(0.1, 10.0)])
        self.temperature = float(np.clip(result.x[0], 0.1, 10.0))
        return self.temperature

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, severity_train: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, epochs: int = 10):
        self.rf.fit(X_train, y_train)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        sev_t = torch.tensor(severity_train.astype(float), dtype=torch.float32).to(self.device) if severity_train is not None else None

        class_counts = torch.bincount(y_t)
        class_weights = (class_counts.float().sum() / (class_counts.float() + 1e-12)).to(self.device)
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        mse_loss = nn.MSELoss()

        dataset = TensorDataset(X_t, y_t) if sev_t is None else TensorDataset(X_t, y_t, sev_t)
        loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)

        for epoch in range(epochs):
            self.pifb.train()
            self.rmb.train()
            epoch_loss = 0.0
            for batch in loader:
                if sev_t is None:
                    xb, yb = batch
                    sb = None
                else:
                    xb, yb, sb = batch
                self.pifb_opt.zero_grad()
                self.rmb_opt.zero_grad()

                logits_p, sev_pred = self.pifb(xb)
                logits_r = self.rmb(xb)
                loss_p = criterion(logits_p, yb)
                loss_r = criterion(logits_r, yb)
                loss = loss_p + loss_r
                if sb is not None:
                    loss = loss + 0.05 * mse_loss(sev_pred, sb)
                loss.backward()
                self.pifb_opt.step()
                self.rmb_opt.step()
                epoch_loss += float(loss.detach().cpu().numpy()) * xb.shape[0]

            if X_val is not None and y_val is not None:
                val_acc = accuracy_score(y_val, self.predict(X_val))
                print(f"Hybrid train epoch {epoch + 1}/{epochs}: loss={epoch_loss / len(dataset):.4f} val_acc={val_acc:.4f}")
            else:
                print(f"Hybrid train epoch {epoch + 1}/{epochs}: loss={epoch_loss / len(dataset):.4f}")

        if X_val is not None and y_val is not None:
            branches = ['pifb', 'rmb', 'rf']
            accuracies = []
            for branch in branches:
                probs = self.predict_proba_branch(X_val, branch)
                preds = np.argmax(probs, axis=1)
                accuracies.append(accuracy_score(y_val, preds))
            self.branch_accuracy = np.array(accuracies, dtype=float)
            self.branch_accuracy = (self.branch_accuracy + 1e-6) / np.sum(self.branch_accuracy + 1e-6)

            # Train a lightweight meta learner on validation branch outputs to learn
            # how to fuse branch probabilities more effectively than a fixed average.
            p_p = self.predict_proba_branch(X_val, 'pifb')
            p_r = self.predict_proba_branch(X_val, 'rmb')
            p_f = self.predict_proba_branch(X_val, 'rf')
            meta_features = np.concatenate([p_p, p_r, p_f], axis=1)
            self.meta_fusion = LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', random_state=42)
            self.meta_fusion.fit(meta_features, y_val)

            self.calibrate_temperature(X_val, y_val)

    def predict_proba_branch(self, X: np.ndarray, branch: str = 'pifb') -> np.ndarray:
        if branch == 'rf':
            if self._rf_fitted():
                probs = self.rf.predict_proba(X)
            else:
                probs = np.ones((X.shape[0], self.pifb.classifier.out_features), dtype=float) / self.pifb.classifier.out_features
            return probs
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        if branch == 'pifb':
            self.pifb.eval()
            with torch.no_grad():
                logits, _ = self.pifb(X_t)
                return torch.softmax(logits, dim=1).cpu().numpy()
        if branch == 'rmb':
            self.rmb.eval()
            with torch.no_grad():
                logits = self.rmb(X_t)
                return torch.softmax(logits, dim=1).cpu().numpy()
        raise ValueError('Unknown branch')

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_p = self.predict_proba_branch(X, 'pifb')
        p_r = self.predict_proba_branch(X, 'rmb')
        p_f = self.predict_proba_branch(X, 'rf')

        n_classes = p_p.shape[1]
        if p_f.shape[1] != n_classes:
            aligned = np.zeros((p_f.shape[0], n_classes), dtype=float)
            for i, cls in enumerate(getattr(self.rf, 'classes_', [])):
                aligned[:, int(cls)] = p_f[:, i]
            p_f = aligned

        conf_p = self._sample_confidence(p_p)
        conf_r = self._sample_confidence(p_r)
        conf_f = self._sample_confidence(p_f)
        branch_confidence = np.stack([conf_p, conf_r, conf_f], axis=1)

        weights = self.branch_accuracy[np.newaxis, :] * branch_confidence
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-12)

        stacked = np.stack([p_p, p_r, p_f], axis=1)
        fused = np.sum(weights[:, :, np.newaxis] * stacked, axis=1)
        fused = fused / (np.sum(fused, axis=1, keepdims=True) + 1e-12)
        if self.meta_fusion is not None:
            meta_input = np.concatenate([p_p, p_r, p_f], axis=1)
            fused = self.meta_fusion.predict_proba(meta_input)
        return self._apply_temperature(fused)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, path_prefix: str):
        torch.save(self.pifb.state_dict(), path_prefix + "_pifb.pt")
        torch.save(self.rmb.state_dict(), path_prefix + "_rmb.pt")
        joblib.dump(self.rf, path_prefix + "_rf.joblib")
        if self.meta_fusion is not None:
            joblib.dump(self.meta_fusion, path_prefix + "_meta.joblib")
        np.save(path_prefix + "_weights.npy", self.branch_accuracy)
        np.save(path_prefix + "_temperature.npy", np.array([self.temperature], dtype=float))

    def load(self, path_prefix: str):
        self.pifb.load_state_dict(torch.load(path_prefix + "_pifb.pt", map_location=self.device))
        self.rmb.load_state_dict(torch.load(path_prefix + "_rmb.pt", map_location=self.device))
        self.rf = joblib.load(path_prefix + "_rf.joblib")
        meta_path = path_prefix + "_meta.joblib"
        self.meta_fusion = joblib.load(meta_path) if os.path.exists(meta_path) else None
        self.branch_accuracy = np.load(path_prefix + "_weights.npy")
        self.temperature = float(np.load(path_prefix + "_temperature.npy")[0])
