"""
Architecture Classifier — Maps fingerprint vectors to architecture classes.
Uses a rule-based heuristic model (no training data required for demo).
Replace with a trained sklearn/PyTorch meta-model for production use.
"""

import numpy as np
from typing import Dict, List

ARCHITECTURE_CLASSES = [
    "DenseMLP",
    "CNN_Basic",
    "CNN_Deep",
    "ResNet",
    "VGG",
    "AlexNet",
    "Lightweight_CNN",
    "Transformer_Encoder",
]

# Heuristic weight matrix: maps fingerprint features → class scores
# Row i = feature importance weights for class i
# In production, train this with meta-learning on simulated API responses
_CLASS_WEIGHTS = np.random.default_rng(42).normal(0, 0.1, size=(len(ARCHITECTURE_CLASSES), 64))

# Architecture profiles: [avg_layers, avg_width, activation, normalization]
_ARCH_PROFILES: Dict[str, Dict] = {
    "DenseMLP":           {"layers_range": (3, 8),   "width_range": (64, 1024),  "norm": "None"},
    "CNN_Basic":          {"layers_range": (4, 8),   "width_range": (16, 128),   "norm": "BatchNorm"},
    "CNN_Deep":           {"layers_range": (12, 30),  "width_range": (32, 512),  "norm": "BatchNorm"},
    "ResNet":             {"layers_range": (18, 152), "width_range": (64, 512),  "norm": "BatchNorm"},
    "VGG":                {"layers_range": (16, 19),  "width_range": (64, 512),  "norm": "BatchNorm"},
    "AlexNet":            {"layers_range": (8, 8),    "width_range": (96, 4096), "norm": "LocalResponseNorm"},
    "Lightweight_CNN":    {"layers_range": (5, 12),   "width_range": (16, 128),  "norm": "BatchNorm"},
    "Transformer_Encoder":{"layers_range": (4, 12),   "width_range": (256, 2048),"norm": "LayerNorm"},
}

_ACTIVATIONS = ["ReLU", "LeakyReLU", "Tanh", "Sigmoid", "GELU", "ELU", "SiLU"]


class ArchitectureClassifier:
    """
    Predicts neural network architecture class from a fingerprint vector.
    Uses pre-trained meta-classifier (or heuristic fallback).
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self._use_heuristic = True

        if model_path:
            try:
                self._load_model(model_path)
                self._use_heuristic = False
            except Exception:
                pass

    def _load_model(self, path: str):
        """Load a trained sklearn/PyTorch classifier from disk."""
        import joblib
        self._model = joblib.load(path)
        self._use_heuristic = False

    def _heuristic_predict(self, fingerprint: np.ndarray) -> Dict:
        """
        Rule-based architecture inference from fingerprint statistics.
        Replace with trained model for higher accuracy.
        """
        # Feature indices in fingerprint:
        latency_mean = fingerprint[0]
        latency_std = fingerprint[1]
        output_stability = fingerprint[8]
        curvature_mag = fingerprint[24]  # from activation signature
        power_law_slope = fingerprint[28]  # from epsilon sensitivity

        # Heuristic: deeper networks → higher latency variance
        # High curvature → ReLU-class activations
        # LayerNorm vs BatchNorm → output stability pattern

        # Score each architecture
        scores = {}
        for arch in ARCHITECTURE_CLASSES:
            profile = _ARCH_PROFILES[arch]
            l_min, l_max = profile["layers_range"]

            # Estimate depth score from latency stats
            depth_score = np.clip(latency_std / (latency_mean + 1e-9), 0, 1)
            layer_mid = (l_min + l_max) / 2

            # Normalize per architecture expectation
            if layer_mid < 10:
                arch_score = depth_score * 0.7 + curvature_mag * 0.3
            elif layer_mid < 50:
                arch_score = depth_score * 0.5 + output_stability * 0.3 + abs(power_law_slope) * 0.2
            else:
                arch_score = depth_score * 0.4 + output_stability * 0.4 + abs(power_law_slope) * 0.2

            scores[arch] = float(arch_score)

        predicted = max(scores, key=scores.get)
        confidence = scores[predicted] / (sum(scores.values()) + 1e-9)

        # Estimate layer count from fingerprint
        profile = _ARCH_PROFILES[predicted]
        l_min, l_max = profile["layers_range"]
        estimated_layers = int(l_min + (l_max - l_min) * np.clip(latency_std, 0, 1))

        w_min, w_max = profile["width_range"]
        estimated_widths = [w_max, (w_min + w_max) // 2, w_min]

        activation = "ReLU" if curvature_mag > 0.3 else "GELU" if curvature_mag > 0.15 else "Tanh"

        return {
            "architecture_class": predicted,
            "estimated_layers": estimated_layers,
            "estimated_widths": estimated_widths,
            "activation": activation,
            "normalization": profile["norm"],
            "confidence": round(float(np.clip(confidence, 0, 1)), 4),
        }

    def predict(self, fingerprint: np.ndarray) -> Dict:
        """
        Predict architecture class from fingerprint vector.
        fingerprint: shape (64,) normalized feature vector.
        """
        if self._use_heuristic:
            return self._heuristic_predict(fingerprint)

        proba = self._model.predict_proba(fingerprint.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        return {
            "architecture_class": ARCHITECTURE_CLASSES[idx],
            "estimated_layers": _ARCH_PROFILES[ARCHITECTURE_CLASSES[idx]]["layers_range"][0],
            "estimated_widths": [],
            "activation": "ReLU",
            "normalization": _ARCH_PROFILES[ARCHITECTURE_CLASSES[idx]]["norm"],
            "confidence": float(proba[idx]),
        }

    def train(self, X: np.ndarray, y: np.ndarray, save_path: str = "model.joblib"):
        """
        Train the meta-classifier on fingerprint vectors with known labels.
        X: shape (N, 64) fingerprint vectors
        y: shape (N,) architecture class indices
        """
        from sklearn.ensemble import GradientBoostingClassifier
        import joblib

        clf = GradientBoostingClassifier(n_estimators=200, max_depth=6)
        clf.fit(X, y)
        joblib.dump(clf, save_path)
        self._model = clf
        self._use_heuristic = False
        print(f"[+] Meta-classifier trained and saved to {save_path}")
