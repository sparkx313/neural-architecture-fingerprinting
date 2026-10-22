"""
Train meta-classifier on simulated fingerprint data.

Generates synthetic fingerprints for known architectures and trains
a GradientBoostingClassifier to map fingerprint vectors → architecture class.

Usage: python experiments/train_meta_model.py --epochs 1000 --output model.joblib
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

ARCH_CLASSES = [
    "DenseMLP", "CNN_Basic", "CNN_Deep", "ResNet",
    "VGG", "AlexNet", "Lightweight_CNN", "Transformer_Encoder",
]

# Synthetic fingerprint generator per architecture class
_ARCH_SIGNATURES = {
    "DenseMLP":            {"latency_mean": 0.003, "latency_std": 0.001, "stability": 0.01, "curvature": 0.4},
    "CNN_Basic":           {"latency_mean": 0.012, "latency_std": 0.004, "stability": 0.05, "curvature": 0.5},
    "CNN_Deep":            {"latency_mean": 0.045, "latency_std": 0.018, "stability": 0.08, "curvature": 0.45},
    "ResNet":              {"latency_mean": 0.078, "latency_std": 0.032, "stability": 0.12, "curvature": 0.42},
    "VGG":                 {"latency_mean": 0.092, "latency_std": 0.041, "stability": 0.10, "curvature": 0.44},
    "AlexNet":             {"latency_mean": 0.022, "latency_std": 0.009, "stability": 0.06, "curvature": 0.48},
    "Lightweight_CNN":     {"latency_mean": 0.007, "latency_std": 0.003, "stability": 0.04, "curvature": 0.46},
    "Transformer_Encoder": {"latency_mean": 0.065, "latency_std": 0.055, "stability": 0.20, "curvature": 0.30},
}


def generate_fingerprint(arch_class: str, rng: np.random.Generator) -> np.ndarray:
    """Simulate a 64-dim fingerprint vector for a given architecture."""
    sig = _ARCH_SIGNATURES[arch_class]
    fp = np.zeros(64, dtype=np.float32)

    # Latency stats (indices 0-7)
    fp[0] = sig["latency_mean"] + rng.normal(0, 0.002)
    fp[1] = sig["latency_std"] + rng.normal(0, 0.001)

    # Output stability (indices 8-15)
    fp[8] = sig["stability"] + rng.normal(0, 0.01)

    # Activation signature (indices 16-27)
    fp[24] = sig["curvature"] + rng.normal(0, 0.03)

    # Epsilon sensitivity power-law slope (indices 28-35)
    # Deeper nets → steeper slope
    depth_factor = sig["latency_mean"] * 10
    fp[28] = depth_factor + rng.normal(0, 0.5)

    # Fill remaining with small noise
    fp[36:] = rng.normal(0, 0.02, size=28).astype(np.float32)

    # Normalize
    norm = np.linalg.norm(fp) + 1e-9
    return fp / norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000, help="Samples per class")
    parser.add_argument("--output", default="model.joblib")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    X, y = [], []

    print(f"[*] Generating {args.samples} synthetic fingerprints per architecture class...")
    for class_idx, arch in enumerate(ARCH_CLASSES):
        for _ in range(args.samples):
            X.append(generate_fingerprint(arch, rng))
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[*] Training GradientBoostingClassifier...")
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[*] Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=ARCH_CLASSES))

    joblib.dump(clf, args.output)
    print(f"[+] Model saved → {args.output}")


if __name__ == "__main__":
    main()
