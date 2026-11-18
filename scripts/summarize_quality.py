#!/usr/bin/env python3
"""Summarize a raw quality-sweep CSV into the metric,value golden format.

The golden-compare gate (``scripts/check_golden_curve.py``, Invariant 7)
compares a two-row ``metric,value`` table against the frozen
``bench/baselines/quality-golden.csv``; the checker's default current-path is
``bench/data/quality-current.csv`` (the tracked ``run_suite.py`` stub, left
untouched), so CI invokes the checker with
``--current bench/data/quality-current-summary.csv`` after this script. The
sweep CLI writes one row per (config, context) cell instead, so this script
reduces a sweep output to the two gated metrics using the same aggregation as
``bench/quality_sweep.py --refresh-golden``: mean ``acc_mean`` and mean
``theoretical_vram_mib`` over ``status == "OK"`` rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Reduce a sweep CSV to golden metrics")
    parser.add_argument("--input", type=Path, default=Path("bench/data/quality-current-sweep.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("bench/data/quality-current-summary.csv")
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    ok = df[df["status"] == "OK"] if "status" in df.columns else df
    if len(ok) == 0:
        print("[FAIL] no OK rows in sweep output; refusing to write an empty summary")
        return 1
    summary = pd.DataFrame(
        [
            {"metric": "accuracy", "value": float(ok["acc_mean"].mean())},
            {"metric": "vram_mib", "value": float(ok["theoretical_vram_mib"].mean())},
        ]
    )
    summary.to_csv(args.output, index=False)
    print(f"[OK] summarized {len(ok)}/{len(df)} OK rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
