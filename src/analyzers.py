"""
Response Analyzer — Extracts architecture-revealing features from API responses.
"""

import numpy as np
from typing import Dict, List
from scipy import stats


class ResponseAnalyzer:
    """
    Builds a fixed-length fingerprint vector from raw API probe responses.
    """

    # Fingerprint vector dimensions
    FINGERPRINT_SIZE = 64

    def __init__(self):
        pass

    def _latency_stats(self, responses: List[Dict]) -> np.ndarray:
        """Latency distribution stats — reveals network depth (more layers → higher variance)."""
        latencies = np.array([r["latency"] for r in responses if r.get("success")])
        if len(latencies) == 0:
            return np.zeros(8)
        return np.array([
            latencies.mean(),
            latencies.std(),
            latencies.min(),
            latencies.max(),
            np.percentile(latencies, 25),
            np.percentile(latencies, 75),
            stats.skew(latencies),
            stats.kurtosis(latencies),
        ], dtype=np.float32)

    def _output_stability(self, responses: List[Dict]) -> np.ndarray:
        """Variance of outputs across repeated calls — reveals dropout/normalization presence."""
        outputs = []
        for r in responses:
            if not r.get("success"):
                continue
            out = r.get("output", {})
            # Flatten numeric output values
            vals = self._extract_numerics(out)
            if vals is not None:
                outputs.append(vals)

        if len(outputs) < 2:
            return np.zeros(8)

        arr = np.array(outputs, dtype=np.float32)
        row_var = arr.var(axis=0)
        return np.array([
            row_var.mean(),
            row_var.std(),
            row_var.min(),
            row_var.max(),
            np.median(row_var),
            stats.skew(row_var),
            stats.kurtosis(row_var),
            arr.shape[-1] if arr.ndim > 1 else 0,
        ], dtype=np.float32)

    def _activation_signature(self, responses: List[Dict]) -> np.ndarray:
        """
        Analyze step-response curves to detect activation function type.
        Looks at how outputs change with threshold-step inputs.
        """
        step_outputs = []
        for r in responses:
            if r.get("probe_type") == "step" and r.get("success"):
                out_vals = self._extract_numerics(r["output"])
                if out_vals is not None:
                    threshold = r["probe_params"].get("threshold", 0.5)
                    step_outputs.append((threshold, out_vals.mean()))

        if len(step_outputs) < 3:
            return np.zeros(12)

        thresholds = np.array([t for t, _ in step_outputs])
        means = np.array([m for _, m in step_outputs])
        order = np.argsort(thresholds)
        means = means[order]

        # Detect non-linearity shape
        diff1 = np.diff(means)
        diff2 = np.diff(diff1)
        return np.array([
            means.mean(),
            means.std(),
            diff1.mean(),
            diff1.std(),
            diff2.mean(),
            diff2.std(),
            np.abs(diff2).max(),   # curvature magnitude
            (diff1 < 0).sum() / (len(diff1) + 1e-9),  # non-monotonicity ratio
            len(means),
            np.percentile(means, 10),
            np.percentile(means, 90),
            stats.skew(means),
        ], dtype=np.float32)[:12]

    def _epsilon_sensitivity(self, responses: List[Dict]) -> np.ndarray:
        """
        Measure output change in response to tiny input perturbations.
        Reveals float precision handling (float32 vs float64) and numerical depth.
        """
        eps_pairs = []
        for r in responses:
            if r.get("probe_type") == "epsilon" and r.get("success"):
                out_vals = self._extract_numerics(r["output"])
                eps = r["probe_params"].get("eps", 0)
                if out_vals is not None:
                    eps_pairs.append((eps, out_vals))

        if len(eps_pairs) < 2:
            return np.zeros(8)

        # Compute output distance from base for each eps
        base_output = eps_pairs[-1][1]  # largest eps as reference
        distances = []
        for eps, out in eps_pairs:
            dist = np.linalg.norm(out - base_output)
            distances.append((eps, dist))

        eps_arr = np.array([e for e, _ in distances])
        dist_arr = np.array([d for _, d in distances])

        # Fit power-law: distance ~ eps^alpha  (alpha reveals depth)
        valid = (eps_arr > 0) & (dist_arr > 0)
        if valid.sum() >= 2:
            log_eps = np.log(eps_arr[valid])
            log_dist = np.log(dist_arr[valid])
            slope, intercept, _, _, _ = stats.linregress(log_eps, log_dist)
        else:
            slope, intercept = 0.0, 0.0

        return np.array([
            slope,
            intercept,
            dist_arr.mean(),
            dist_arr.std(),
            dist_arr.max(),
            (dist_arr > 1e-6).sum() / (len(dist_arr) + 1e-9),
            eps_arr.min(),
            eps_arr.max(),
        ], dtype=np.float32)

    def _probe_type_distribution(self, responses: List[Dict]) -> np.ndarray:
        """Success rate and latency by probe category."""
        type_map: Dict[str, List[float]] = {}
        for r in responses:
            t = r.get("probe_type", "unknown")
            if t not in type_map:
                type_map[t] = []
            type_map[t].append(r["latency"] if r.get("success") else np.nan)

        known_types = ["impulse", "ramp", "sinusoidal", "uniform", "normal", "step", "epsilon", "edge"]
        stats_arr = []
        for t in known_types:
            vals = np.array(type_map.get(t, [np.nan]))
            mask = ~np.isnan(vals)
            if mask.sum() > 0:
                stats_arr.extend([vals[mask].mean(), vals[mask].std()])
            else:
                stats_arr.extend([0.0, 0.0])
        return np.array(stats_arr, dtype=np.float32)[:32]

    def _extract_numerics(self, obj) -> np.ndarray:
        """Recursively extract numeric values from a JSON-like response."""
        if isinstance(obj, (int, float)):
            return np.array([float(obj)], dtype=np.float32)
        if isinstance(obj, list):
            parts = [self._extract_numerics(i) for i in obj]
            return np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        if isinstance(obj, dict):
            parts = [self._extract_numerics(v) for v in obj.values()]
            return np.concatenate(parts) if parts else np.array([], dtype=np.float32)
        return np.array([], dtype=np.float32)

    def build_fingerprint(self, responses: List[Dict]) -> np.ndarray:
        """
        Compose a fixed-length fingerprint vector from all response features.
        """
        parts = [
            self._latency_stats(responses),          # 8
            self._output_stability(responses),       # 8
            self._activation_signature(responses),   # 12
            self._epsilon_sensitivity(responses),    # 8
            self._probe_type_distribution(responses),# 32
        ]
        fingerprint = np.concatenate(parts).astype(np.float32)

        # Pad or truncate to fixed size
        if len(fingerprint) < self.FINGERPRINT_SIZE:
            fingerprint = np.pad(fingerprint, (0, self.FINGERPRINT_SIZE - len(fingerprint)))
        else:
            fingerprint = fingerprint[:self.FINGERPRINT_SIZE]

        # Normalize
        norm = np.linalg.norm(fingerprint) + 1e-9
        return fingerprint / norm
