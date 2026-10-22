# Neural Architecture Fingerprinting from API Black-Boxes

> Reverse-engineer neural network architectures by analyzing only API response patterns — no weights, no gradients, no model access.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

---

## Abstract

This tool performs **black-box neural architecture extraction** — given only query access to an API endpoint (e.g., a classification service), it infers the underlying architecture (depth, layer types, activation functions, width) through carefully designed input perturbation analysis and response-pattern fingerprinting.

**No model weights. No gradients. No architecture disclosure. Just an API.**

---

## How It Works

1. **Query Probing** — Systematically probe the API with structured inputs (impulse signals, ramps, frequency sweeps)
2. **Response Analysis** — Measure latency variance, numerical precision patterns, and output sensitivity gradients
3. **Feature Extraction** — Extract a "fingerprint vector" from response behavior (depth signatures, activation boundary detection)
4. **Architecture Inference** — Use a meta-learned classifier to map fingerprints to architecture classes (ResNet vs VGG vs Dense vs CNN variants)
5. **Confidence Scoring** — Report inferred architecture with probabilistic confidence scores

---

## Key Techniques

| Technique | What It Reveals |
|-----------|----------------|
| Impulse response latency | Network depth (layer count) |
| Numerical precision drift | Activation function type |
| Output sensitivity gradient | Layer width patterns |
| Repeated-call variance | Dropout/normalization presence |
| Adversarial perturbation decay | Loss surface curvature → architecture class |

---

## Installation

```bash
git clone https://github.com/sparkx313/neural-architecture-fingerprinting.git
cd neural-architecture-fingerprinting
pip install -r requirements.txt
```

---

## Usage

### Basic Fingerprinting

```python
from src.fingerprint import ArchitectureFingerprinter

target_api = "http://localhost:8080/predict"
fingerprinter = ArchitectureFingerprinter(
    api_endpoint=target_api,
    num_probes=500,
    input_shape=(1, 784)  # example: MNIST-sized input
)

result = fingerprinter.infer()
print(result)
# {
#   "architecture_class": "DenseMLP",
#   "estimated_layers": 5,
#   "estimated_widths": [784, 256, 128, 64, 10],
#   "activation": "ReLU",
#   "normalization": "None",
#   "confidence": 0.87
# }
```

### CLI Mode

```bash
python -m src.cli \
    --api-endpoint "http://target-api/predict" \
    --input-dim 784 \
    --probes 1000 \
    --output report.json \
    --verbose
```

### Batch Analysis (multiple APIs)

```bash
python -m src.batch_analyze \
    --targets endpoints.txt \
    --parallel 4 \
    --output-dir ./reports/
```

---

## Project Structure

```
neural-architecture-fingerprinting/
├── src/
│   ├── fingerprint.py       # Core fingerprinting engine
│   ├── probers.py           # Input probe generators
│   ├── analyzers.py         # Response pattern analysis
│   ├── classifier.py        # Architecture inference model
│   ├── cli.py               # Command-line interface
│   └── batch_analyze.py    # Multi-target analysis
├── experiments/
│   ├── train_meta_model.py  # Train the fingerprint classifier
│   └── validate_accuracy.py # Benchmark against known architectures
├── data/
│   └── fingerprints/        # Stored fingerprint vectors
├── notebooks/
│   └── exploration.ipynb    # Research notebook
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Research Context

This builds on work in:
- **Model Extraction Attacks** (Tramer et al., 2016)
- **Neural Network Fingerprinting** (Oh et al., 2019)
- **Black-Box Architecture Inference** (Yosinski et al., 2016)

Unlike prior work requiring gradient access, this tool operates **purely on API-level observations**.

---

## Disclaimer

This is a **security research tool**. Use only against APIs you own or have explicit written permission to analyze. Unauthorized architecture extraction may violate terms of service or applicable laws.

---

## License

MIT © [sparkx313](https://github.com/sparkx313)
