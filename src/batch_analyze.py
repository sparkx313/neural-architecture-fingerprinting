"""
Batch analysis of multiple API endpoints.
"""

import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .fingerprint import ArchitectureFingerprinter


def analyze_single(endpoint: str, input_shape, num_probes, timeout) -> dict:
    fp = ArchitectureFingerprinter(
        api_endpoint=endpoint,
        input_shape=tuple(input_shape),
        num_probes=num_probes,
        timeout=timeout,
    )
    try:
        result = fp.infer()
        return {"endpoint": endpoint, "status": "ok", "result": result.to_dict()}
    except Exception as e:
        return {"endpoint": endpoint, "status": "error", "error": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch API fingerprinting")
    parser.add_argument("--targets", required=True, help="File with one API URL per line")
    parser.add_argument("--input-shape", nargs="+", type=int, default=[1, 784])
    parser.add_argument("--probes", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--output-dir", default="./reports/")
    args = parser.parse_args()

    targets = Path(args.targets).read_text().strip().splitlines()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Analyzing {len(targets)} endpoints ({args.parallel} parallel)...")
    results = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(analyze_single, url, args.input_shape, args.probes, args.timeout): url
            for url in targets
        }
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            status = "OK" if r["status"] == "ok" else "FAIL"
            print(f"  [{status}] {r['endpoint']}")

    out_file = out_dir / "batch_report.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\n[+] Full report → {out_file}")


if __name__ == "__main__":
    main()
