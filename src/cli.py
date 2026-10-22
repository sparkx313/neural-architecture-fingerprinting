"""
CLI interface for Neural Architecture Fingerprinting tool.
Usage: python -m src.cli --api-endpoint <url> --input-dim <int> --probes <int>
"""

import argparse
import json
import sys
import time
from pathlib import Path

from .fingerprint import ArchitectureFingerprinter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Black-box Neural Architecture Fingerprinting Tool"
    )
    parser.add_argument(
        "--api-endpoint", "-u",
        required=True,
        help="Target API endpoint URL (POST endpoint accepting JSON {\"input\": [...]})",
    )
    parser.add_argument(
        "--input-dim", "-d",
        type=int,
        default=784,
        help="Flat dimension of input vectors (default: 784 for MNIST)",
    )
    parser.add_argument(
        "--input-shape", "-s",
        nargs="+",
        type=int,
        default=[1, 784],
        help="Input tensor shape, e.g. --input-shape 1 28 28",
    )
    parser.add_argument(
        "--probes", "-n",
        type=int,
        default=500,
        help="Number of probes to send (default: 500)",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=5.0,
        help="API request timeout in seconds",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON report file path",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_shape = tuple(args.input_shape)

    if args.verbose:
        print(f"[CLI] Target:   {args.api_endpoint}")
        print(f"[CLI] Shape:    {input_shape}")
        print(f"[CLI] Probes:   {args.probes}")
        print(f"[CLI] Timeout:  {args.timeout}s")
        print()

    fingerprinter = ArchitectureFingerprinter(
        api_endpoint=args.api_endpoint,
        input_shape=input_shape,
        num_probes=args.probes,
        timeout=args.timeout,
    )

    start = time.perf_counter()
    result = fingerprinter.infer()
    elapsed = time.perf_counter() - start

    report = result.to_dict()
    report["_meta"] = {
        "api_endpoint": args.api_endpoint,
        "num_probes": args.probes,
        "elapsed_seconds": round(elapsed, 2),
        "tool": "neural-architecture-fingerprinting v1.0",
    }

    print("\n[RESULT] Inferred Architecture:")
    print(json.dumps(report, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n[+] Report saved → {out_path}")


if __name__ == "__main__":
    main()
