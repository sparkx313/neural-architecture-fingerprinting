"""
Probe Generators — Structured input suites for API fingerprinting.
"""

import numpy as np
from typing import Dict, List, Tuple


class ProbeGenerator:
    """
    Generates diverse structured probe vectors to elicit
    architecture-revealing response patterns from target API.
    """

    def __init__(self, input_shape: Tuple[int, ...] = (1, 784), seed: int = 42):
        self.input_shape = input_shape
        self.rng = np.random.default_rng(seed)
        self.flat_dim = int(np.prod(input_shape))

    def _impulse(self, position: int) -> np.ndarray:
        """Single-point impulse at given index."""
        v = np.zeros(self.flat_dim, dtype=np.float32)
        v[position] = 1.0
        return v.reshape(self.input_shape)

    def _ramp(self, start: float = 0.0, end: float = 1.0) -> np.ndarray:
        """Linear ramp signal."""
        v = np.linspace(start, end, self.flat_dim, dtype=np.float32)
        return v.reshape(self.input_shape)

    def _sinusoidal(self, freq: float = 1.0) -> np.ndarray:
        """Sinusoidal probe — reveals non-linearity boundaries."""
        x = np.linspace(0, 2 * np.pi, self.flat_dim, dtype=np.float32)
        v = np.sin(freq * x).astype(np.float32)
        return v.reshape(self.input_shape)

    def _random_uniform(self, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        v = self.rng.uniform(low, high, size=self.flat_dim).astype(np.float32)
        return v.reshape(self.input_shape)

    def _random_normal(self, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        v = self.rng.normal(mean, std, size=self.flat_dim).astype(np.float32)
        return v.reshape(self.input_shape)

    def _step(self, threshold: float = 0.5) -> np.ndarray:
        """Heaviside step at given threshold."""
        v = (np.linspace(0, 1, self.flat_dim) > threshold).astype(np.float32)
        return v.reshape(self.input_shape)

    def _epsilon_perturbation(self, base: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Tiny perturbation to test precision sensitivity."""
        noise = self.rng.normal(0, eps, size=base.shape).astype(np.float32)
        return base + noise

    def _edge_detector(self, dim: int = 16) -> np.ndarray:
        """Structured 2D edge pattern (if applicable to input shape)."""
        v = np.zeros(self.flat_dim, dtype=np.float32)
        side = int(np.sqrt(self.flat_dim))
        if side * side == self.flat_dim:
            grid = np.zeros((side, side), dtype=np.float32)
            mid = side // 2
            grid[:, :mid] = 0.0
            grid[:, mid:] = 1.0
            v = grid.flatten()
        return v.reshape(self.input_shape)

    def generate_suite(self, num_probes: int) -> List[Dict]:
        """
        Generate a diverse suite of probes across categories.
        Returns list of dicts: {vector, type, params}
        """
        suite = []

        # 1. Impulse probes (depth detection)
        impulse_positions = self.rng.integers(0, self.flat_dim, size=max(50, num_probes // 5))
        for pos in impulse_positions:
            suite.append({
                "vector": self._impulse(int(pos)),
                "type": "impulse",
                "params": {"position": int(pos)},
            })

        # 2. Ramp probes (linearity test)
        for start, end in [(0, 1), (1, 0), (-1, 1), (0, 0)]:
            suite.append({
                "vector": self._ramp(start, end),
                "type": "ramp",
                "params": {"start": start, "end": end},
            })

        # 3. Sinusoidal probes at various frequencies
        freqs = np.logspace(-1, 2, num=min(30, num_probes // 4))
        for freq in freqs:
            suite.append({
                "vector": self._sinusoidal(float(freq)),
                "type": "sinusoidal",
                "params": {"freq": float(freq)},
            })

        # 4. Random uniform and normal samples
        n_random = max(50, num_probes // 3)
        for _ in range(n_random):
            suite.append({
                "vector": self._random_uniform(),
                "type": "uniform",
                "params": {},
            })
            suite.append({
                "vector": self._random_normal(),
                "type": "normal",
                "params": {},
            })

        # 5. Step functions at various thresholds
        for threshold in np.linspace(0, 1, 20):
            suite.append({
                "vector": self._step(float(threshold)),
                "type": "step",
                "params": {"threshold": float(threshold)},
            })

        # 6. Epsilon perturbation pairs (precision sensitivity)
        base = self._random_uniform()
        for eps in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
            suite.append({
                "vector": self._epsilon_perturbation(base, eps),
                "type": "epsilon",
                "params": {"eps": eps, "base_id": "fixed_base_0"},
            })

        # 7. Edge pattern
        suite.append({
            "vector": self._edge_detector(),
            "type": "edge",
            "params": {},
        })

        # Trim or return as-is
        return suite[:num_probes] if len(suite) > num_probes else suite
