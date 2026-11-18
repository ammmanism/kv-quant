"""Position-resolved needle-in-a-haystack probe with log-prob phase sensor."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from tasks.base import EvalTask, RecordedReplayAdapter, SweepManifest, TrialRecord

CONTEXT_LENGTHS = (4096, 16384, 32768, 65536, 131072)
DEPTH_PCTS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

# Three needle formats; the first is a zero-prior random hex passkey.
NEEDLE_FORMATS = (
    "random_hex_8",  # zero-prior random hex passkey
    "uuid_passkey",
    "numeric_code",
)

CONFIGS = (
    "fp16",
    "k8v8",
    "k8v4",
    "k4v8",
    "k4v4_postrope_sink4",
    "k4v4_postrope_nosink",
    "k4v4_prerope_ch",
)

CLIFF_ACC_DROP = 0.15  # >15 percentage points
CLIFF_LOGP_NATS = 1.2  # >1.2 nats drift


@dataclass
class CliffCoordinate:
    """A pinpointed (context_len, depth_pct) failure coordinate."""

    config_id: str
    context_len: int
    depth_pct: int
    acc_drop: float
    delta_logp: float
    reason: str


class PositionResolvedNeedleTask(EvalTask):
    """2D context x depth needle retrieval with continuous Delta-logP sensor."""

    name = "needle_position_resolved"

    def __init__(self, adapter: RecordedReplayAdapter | None = None):
        self.adapter = adapter or RecordedReplayAdapter()

    def run(self, manifest: SweepManifest) -> list[TrialRecord]:
        records: list[TrialRecord] = []
        for cfg in CONFIGS:
            for ctx in CONTEXT_LENGTHS:
                for depth in DEPTH_PCTS:
                    for fmt in NEEDLE_FORMATS:
                        records.extend(self.adapter.replay(cfg, ctx, depth, fmt))
        return records


def detect_cliff_coordinates(df: pd.DataFrame) -> list[CliffCoordinate]:
    """Pinpoint (context_len, depth_pct) cells where accuracy drops >15% or
    Delta-logP exceeds 1.2 nats versus the fp16 baseline at the same position.

    Args:
        df: Aggregated needle dataframe from ``EvalTask.to_dataframe``.

    Returns:
        List of CliffCoordinate, sorted by context length then depth.
    """
    # Average across needle formats for each (config, context, depth)
    agg = df.groupby(["config_id", "context_len", "depth_pct"], as_index=False).agg(
        acc_mean=("acc_mean", "mean"), delta_logp=("delta_logp", "mean")
    )
    baseline_rows = agg[agg["config_id"] == "fp16"].to_dict("records")
    baseline: dict[tuple[int, int], tuple[float, float]] = {
        (int(r["context_len"]), int(r["depth_pct"])): (
            float(r["acc_mean"]),
            float(r["delta_logp"]),
        )
        for r in baseline_rows
    }

    cliffs: list[CliffCoordinate] = []
    for row in agg.to_dict("records"):
        cfg = str(row["config_id"])
        ctx = int(row["context_len"])
        depth = int(row["depth_pct"])
        acc = float(row["acc_mean"])
        drift_val = float(row["delta_logp"])
        if cfg == "fp16":
            continue
        ref = baseline.get((ctx, depth))
        if ref is None:
            continue
        ref_acc, ref_drift = ref
        acc_drop = ref_acc - acc
        drift = drift_val - ref_drift
        reasons = []
        if acc_drop > CLIFF_ACC_DROP:
            reasons.append(f"acc_drop={acc_drop:.3f}>{CLIFF_ACC_DROP}")
        if drift > CLIFF_LOGP_NATS:
            reasons.append(f"delta_logp={drift:.3f}>{CLIFF_LOGP_NATS}")
        if reasons:
            cliffs.append(
                CliffCoordinate(
                    config_id=cfg,
                    context_len=ctx,
                    depth_pct=depth,
                    acc_drop=acc_drop,
                    delta_logp=drift,
                    reason="; ".join(reasons),
                )
            )
    return sorted(cliffs, key=lambda c: (c.context_len, c.depth_pct, c.config_id))
