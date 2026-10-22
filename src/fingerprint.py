"""
Architecture Fingerprinter — Core Engine
Infers neural network architecture from API response patterns only.
"""

import time
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

from .probers import ProbeGenerator
from .analyzers import ResponseAnalyzer
from .classifier import ArchitectureClassifier


@dataclass
class FingerprintResult:
    architecture_class: str
    estimated_layers: int
    estimated_widths: List[int]
    activation: str
    normalization: str
    confidence: float
    raw_fingerprint: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        return {
            "architecture_class": self.architecture_class,
            "estimated_layers": self.estimated_layers,
            "estimated_widths": self.estimated_widths,
            "activation": self.activation,
            "normalization": self.normalization,
            "confidence": round(self.confidence, 4),
        }


class ArchitectureFingerprinter:
    """
    Black-box neural architecture inference engine.

    Probes an API endpoint with structured inputs, analyzes response
    patterns, and infers the underlying network architecture.
    """

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

    ACTIVATION_CANDIDATES = ["ReLU", "LeakyReLU", "Tanh", "Sigmoid", "GELU", "ELU", "SiLU"]

    def __init__(
        self,
        api_endpoint: str,
        input_shape: Tuple[int, ...] = (1, 784),
        num_probes: int = 500,
        timeout: float = 5.0,
        headers: Optional[Dict] = None,
    ):
        self.api_endpoint = api_endpoint
        self.input_shape = input_shape
        self.num_probes = num_probes
        self.timeout = timeout
        self.headers = headers or {"Content-Type": "application/json"}

        self.prober = ProbeGenerator(input_shape=input_shape)
        self.analyzer = ResponseAnalyzer()
        self.classifier = ArchitectureClassifier()

    def _query_api(self, input_vector: np.ndarray) -> Dict:
        """Send a single probe to the target API and record response metadata."""
        payload = {"input": input_vector.tolist()}
        start = time.perf_counter()
        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            elapsed = time.perf_counter() - start
            output = response.json()
            return {
                "latency": elapsed,
                "status_code": response.status_code,
                "output": output,
                "success": True,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            return {
                "latency": elapsed,
                "error": str(e),
                "success": False,
            }

    def collect_responses(self) -> List[Dict]:
        """Generate and send all probes; collect response data."""
        responses = []
        probes = self.prober.generate_suite(self.num_probes)

        print(f"[*] Probing {self.api_endpoint} with {self.num_probes} structured inputs...")
        for probe in tqdm(probes, desc="Probing API"):
            result = self._query_api(probe["vector"])
            result["probe_type"] = probe["type"]
            result["probe_params"] = probe["params"]
            responses.append(result)

        return responses

    def infer(self) -> FingerprintResult:
        """Full pipeline: probe → analyze → classify → report."""
        responses = self.collect_responses()

        # Extract feature vector from response patterns
        fingerprint_vector = self.analyzer.build_fingerprint(responses)

        # Classify architecture from fingerprint
        prediction = self.classifier.predict(fingerprint_vector)

        return FingerprintResult(
            architecture_class=prediction["architecture_class"],
            estimated_layers=prediction["estimated_layers"],
            estimated_widths=prediction["estimated_widths"],
            activation=prediction["activation"],
            normalization=prediction["normalization"],
            confidence=prediction["confidence"],
            raw_fingerprint=fingerprint_vector,
        )
