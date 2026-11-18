"""Unified evaluation spine with model-scale honesty and Wilson CIs."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TrialRecord:
    """A single needle trial (one position, one format, one config)."""

    config_id: str
    context_len: int
    depth_pct: int
    needle_format: str
    trial_idx: int
    correct: bool
    delta_logp: float  # log P_FP16 - log P_quant, in nats


@dataclass
class SweepManifest:
    """Provenance manifest. All fields are mandatory for honest reporting."""

    git_sha: str
    timestamp_utc: str
    model_id: str
    param_count_b: float
    num_q_heads: int
    num_kv_heads: int
    rotary_dim: int
    gpu_name: str
    seed: int
    sha256_csv: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        param_count_b: float = 7.61,
        num_q_heads: int = 28,
        num_kv_heads: int = 4,
        rotary_dim: int = 128,
        seed: int = 42,
        gpu_name: str | None = None,
    ) -> SweepManifest:
        """Capture the current environment into a manifest."""
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            sha = "unknown"
        if gpu_name is None:
            try:
                import torch

                gpu_name = (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only-ci"
                )
            except Exception:
                gpu_name = "cpu-only-ci"
        return cls(
            git_sha=sha,
            timestamp_utc=datetime.now(UTC).isoformat(),
            model_id=model_id,
            param_count_b=param_count_b,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            rotary_dim=rotary_dim,
            gpu_name=gpu_name,
            seed=seed,
        )

    def with_csv_hash(self, csv_path: Path) -> SweepManifest:
        """Return a copy with the line-ending-normalized SHA256 of a CSV."""
        data = csv_path.read_bytes().replace(b"\r\n", b"\n")
        self.sha256_csv = hashlib.sha256(data).hexdigest()
        return self

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion.

    Args:
        successes: Number of successful trials.
        n: Total trials.
        z: Z-score (1.96 for 95% confidence).

    Returns:
        (low, high) interval bounds in [0, 1].
    """
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


class EvalTask(ABC):
    """Abstract base for deterministic, manifest-backed evaluation tasks."""

    name: str = "base"

    @abstractmethod
    def run(self, manifest: SweepManifest) -> list[TrialRecord]:
        """Produce trial records for this task."""

    def to_dataframe(self, records: list[TrialRecord]) -> pd.DataFrame:
        """Aggregate records into per-condition accuracy + CI + drift table."""
        rows = []
        grouped: dict[tuple[str, int, int, str], list[TrialRecord]] = {}
        for r in records:
            key = (r.config_id, r.context_len, r.depth_pct, r.needle_format)
            grouped.setdefault(key, []).append(r)
        for (cfg, ctx, depth, fmt), trials in sorted(grouped.items()):
            n = len(trials)
            correct = sum(1 for t in trials if t.correct)
            lo, hi = wilson_ci(correct, n)
            rows.append(
                {
                    "config_id": cfg,
                    "context_len": ctx,
                    "depth_pct": depth,
                    "needle_format": fmt,
                    "trials": n,
                    "correct": correct,
                    "acc_mean": correct / n if n else 0.0,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "delta_logp": sum(t.delta_logp for t in trials) / n if n else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def sweep(self, manifest: SweepManifest, csv_path: Path | None = None) -> pd.DataFrame:
        """Run the task, produce a dataframe, and optionally persist CSV + manifest."""
        df = self.to_dataframe(self.run(manifest))
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
            manifest.with_csv_hash(csv_path).write(
                csv_path.with_suffix(csv_path.suffix + ".manifest.json")
            )
        return df


class RecordedReplayAdapter:
    """Deterministic replay adapter: regenerates trials from a seed, no GPU.

    This allows CPU CI to exercise the full evaluation spine identically to a
    real model run. The synthetic pattern encodes the *expected* quantization
    failure mode (late-context depth cliff) so downstream gates can validate
    cliff detection without a GPU.
    """

    def __init__(self, seed: int = 42, trials_per_cell: int = 20):
        self.seed = seed
        self.trials_per_cell = trials_per_cell

    def replay(
        self,
        config_id: str,
        context_len: int,
        depth_pct: int,
        needle_format: str,
    ) -> list[TrialRecord]:
        import random

        rng = random.Random(f"{self.seed}|{config_id}|{context_len}|{depth_pct}|{needle_format}")
        base_acc = self._base_accuracy(config_id, context_len, depth_pct)
        base_drift = self._base_drift(config_id, context_len, depth_pct)
        records = []
        for i in range(self.trials_per_cell):
            correct = rng.random() < base_acc
            noise = rng.gauss(0, 0.08)
            records.append(
                TrialRecord(
                    config_id=config_id,
                    context_len=context_len,
                    depth_pct=depth_pct,
                    needle_format=needle_format,
                    trial_idx=i,
                    correct=correct,
                    delta_logp=max(0.0, base_drift + noise),
                )
            )
        return records

    def _base_accuracy(self, cfg: str, ctx: int, depth: int) -> float:
        long_ctx = ctx >= 32768
        late = depth >= 70
        if cfg == "fp16":
            return 0.99 - (ctx / 131072) * 0.01
        if cfg in ("k8v8", "k8v4", "k4v8"):
            return 0.97 - (ctx / 131072) * 0.02  # flat retention
        if cfg == "k4v4_postrope_nosink":
            return 0.30 if (long_ctx and late) else 0.95
        if cfg == "k4v4_prerope_ch":
            return 0.62 if (long_ctx and late) else 0.94
        if cfg == "k4v4_postrope_sink4":
            return 0.68 if (long_ctx and late) else 0.95
        return 0.95

    def _base_drift(self, cfg: str, ctx: int, depth: int) -> float:
        if cfg == "fp16":
            return 0.05
        depth_factor = depth / 100.0
        ctx_factor = ctx / 131072.0
        if "k4v4" in cfg:
            return 0.2 + 1.4 * depth_factor * (0.4 + ctx_factor)
        return 0.1 + 0.3 * depth_factor * ctx_factor
