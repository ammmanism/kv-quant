#!/usr/bin/env python3
"""Generate the Phase 5 ecological-task baseline CSV + manifest."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.base import SweepManifest
from tasks.longcode import CrossFileCodeCompletionTask
from tasks.longsum import LongDocSummarizationTask
from tasks.multihop import MultiHopBridgeTask

DATA_DIR = Path("bench/data")
CSV = DATA_DIR / "p5_ecological_baseline.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = SweepManifest.capture(seed=42, gpu_name="cpu-only-ci")

    frames = []
    for name, task in (
        ("multihop", MultiHopBridgeTask(seed=42, trials_per_cell=20)),
        ("longsum", LongDocSummarizationTask(seed=42, trials_per_cell=20)),
        ("longcode", CrossFileCodeCompletionTask(seed=42, trials_per_cell=20)),
    ):
        df = task.to_dataframe(task.run(manifest))
        df.insert(0, "task", name)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(CSV, index=False)

    manifest.with_csv_hash(CSV).write(CSV.with_suffix(CSV.suffix + ".manifest.json"))

    meta = {
        "file": CSV.name,
        "sha256": sha256(CSV),
        "description": "Ecological task baseline: multi-hop bridge QA, fact-retention longsum, AST longcode",
        "generated_by": "scripts/generate_ecological_baseline.py",
        "columns": list(combined.columns),
        "num_rows": len(combined),
        "tasks": ["multihop", "longsum", "longcode"],
    }
    CSV.with_suffix(CSV.suffix + ".manifest.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] {CSV} ({len(combined)} rows, {len(combined.columns)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
