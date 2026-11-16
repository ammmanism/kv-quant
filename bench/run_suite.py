#!/usr/bin/env python3
"""Benchmark runner for kv-quant."""

import argparse
from pathlib import Path

import pandas as pd


def run_benchmark(
    concurrency: int,
    isl: int,
    osl: int,
    warmup: int,
    runs: int,
    model: str,
    output: Path,
) -> None:
    """Run benchmark suite and save results to CSV."""
    # Placeholder - actual implementation would:
    # 1. Load model
    # 2. Run inference with quantized KV cache
    # 3. Measure accuracy, VRAM, latency, throughput
    # 4. Save to CSV

    print(f"Running benchmark: concurrency={concurrency}, isl={isl}, osl={osl}")
    print(f"Model: {model}")
    print(f"Warmup: {warmup}, Runs: {runs}")

    # Dummy results for skeleton
    results = pd.DataFrame(
        {
            "metric": ["accuracy", "vram_mib", "latency_ms", "throughput_tps"],
            "value": [0.95, 1024.0, 45.2, 1250.0],
            "concurrency": [concurrency] * 4,
            "isl": [isl] * 4,
            "osl": [osl] * 4,
            "model": [model] * 4,
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"Results saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Run kv-quant benchmark suite")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrency level")
    parser.add_argument("--isl", type=int, default=128, help="Input sequence length")
    parser.add_argument("--osl", type=int, default=32, help="Output sequence length")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Timed runs")
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Model to benchmark"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench/data/quality-current.csv"),
        help="Output CSV path",
    )

    args = parser.parse_args()

    run_benchmark(
        concurrency=args.concurrency,
        isl=args.isl,
        osl=args.osl,
        warmup=args.warmup,
        runs=args.runs,
        model=args.model,
        output=args.output,
    )


if __name__ == "__main__":
    main()
