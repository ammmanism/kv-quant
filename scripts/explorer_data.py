"""Deterministic data layer for the interactive explorer.

Enumerates git-tracked CSVs under bench/data/, verifies each manifest's
SHA256 (dual schema: legacy ``sha256`` and modern ``sha256_csv``, both
LF-normalized), and aggregates the sweep data into a JSON-serializable
payload.  No timestamps, no absolute paths, no randomness: the same tree
must always produce byte-identical output (enforced by
scripts/check_explorer_drift.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pandas as pd

DATA_DIR = Path("bench/data")
# CI-regenerated stub; its manifest is deliberately gitignored (see .gitignore).
EXCLUDED = {"quality-current.csv"}

# Epistemic mode per artifact, matching the repo's own vocabulary
# (README provenance header, manifest `mode`/`measure_method` fields).
MODE_LABELS = {
    "p10_hadamard_rescue.csv": "projected",
    "p10_hadamard_rotation.csv": "synthetic",
    "p11_head_rescue_curve.csv": "projected",
    "p11_head_sensitivity_schedule.csv": "synthetic",
    "p12_qjl_mse_ablation.csv": "synthetic",
    "p13_ragged_concurrency.csv": "modeled",
    "p14_agent_trajectory_drift.csv": "simulated",
    "p1_hadamard_qjl_ablation.csv": "synthetic",
    "p1_sqnr_error_bounds.csv": "synthetic",
    "p2_cow_prefix_memory.csv": "measured (CPU)",
    "p2_head_sensitivity_map.csv": "synthetic",
    "p3_hbm_traffic_model.csv": "analytical",
    "p4_needle_depth_matrix.csv": "replay-baseline",
    "p5_ecological_baseline.csv": "replay-baseline",
    "p6_v1_frontier_sweep.csv": "replay-baseline",
    "p7_latency_ttft_itl.csv": "modeled",
    "p7_ncu_bandwidth.csv": "ncu-modeled",
    "p7_vram_crossval.csv": "measured (CPU)",
    "arch_compat_matrix.csv": "reference",
}

# Curated headline configs per task family (subset of what the CSVs hold;
# chosen to carry the v1.0 story: cliff, asymmetry law, sink effect).
FRONTIER_CONFIGS = {
    "needle": (
        # p6 names its FP16 baseline k16v16_post_rope_s4 — NOT "fp16".
        # The UI relabels this config as "FP16 (p6 baseline)".
        "k16v16_post_rope_s4",
        "k8v4_post_rope_s4",
        "k4v8_post_rope_s4",
        "k8v8_post_rope_s4",
        "k4v4_post_rope_s4",
        "k4v4_post_rope_s0",
    ),
    "multihop": (
        "k16v16_post_rope_s4",
        "k8v4_post_rope_s4",
        "k4v8_post_rope_s4",
        "k8v8_post_rope_s4",
        "k4v4_post_rope_s4",
        "k4v4_post_rope_s0",
    ),
    "longsum": ("fp16", "k8v4", "k4v8", "k8v8", "k4v4_postrope"),
    "longcode": ("fp16", "k8v4", "k4v8", "k8v8", "k4v4_postrope"),
}
DEPTH_CONFIGS = {
    "needle": (
        "fp16",
        "k8v4",
        "k4v8",
        "k8v8",
        "k4v4_postrope_sink4",
        "k4v4_postrope_nosink",
    ),
    "multihop": ("fp16", "k8v4", "k4v8", "k8v8", "k4v4_postrope_sink4"),
    "longsum": ("fp16", "k8v4", "k4v8", "k8v8", "k4v4_postrope_sink4"),
    "longcode": ("fp16", "k8v4", "k4v8", "k8v8", "k4v4_postrope_sink4"),
}
# p5 carries one task-specific headline metric alongside acc_mean.
TASK_METRIC = {
    "needle": ("acc_mean", "needle accuracy"),
    "multihop": ("acc_mean", "multi-hop accuracy"),
    "longsum": ("fact_retention_pct", "fact retention"),
    "longcode": ("ast_exact_match", "AST exact match"),
}


def csv_sha256(path: Path) -> str:
    """SHA256 over LF-normalized bytes (cross-platform stable)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def tracked_csvs() -> list[Path]:
    """Git-tracked CSVs in bench/data/, sorted, minus the CI stub."""
    out = subprocess.run(
        ["git", "ls-files", "bench/data"], capture_output=True, text=True, check=True
    )
    files = [Path(f) for f in out.stdout.split() if f.endswith(".csv")]
    return sorted(f for f in files if Path(f).name not in EXCLUDED)


def verified_artifacts() -> list[dict[str, object]]:
    """Verify every tracked CSV against its manifest; return provenance rows."""
    rows: list[dict[str, object]] = []
    for csv_path in tracked_csvs():
        mf_path = csv_path.with_name(csv_path.name + ".manifest.json")
        if not mf_path.exists():
            raise FileNotFoundError(f"missing manifest for {csv_path}")
        meta = json.loads(mf_path.read_text(encoding="utf-8"))
        want = meta.get("sha256_csv") or meta.get("sha256")
        actual = csv_sha256(DATA_DIR / csv_path.name)
        if want != actual:
            raise ValueError(f"SHA256 mismatch for {csv_path}: manifest {want} vs file {actual}")
        frame = pd.read_csv(DATA_DIR / csv_path.name)
        declared: object = meta.get("num_rows")
        if declared is None and isinstance(meta.get("extra"), dict):
            declared = meta["extra"].get("rows")
        if declared is not None and int(declared) != len(frame):  # type: ignore[arg-type]
            raise ValueError(
                f"row-count mismatch for {csv_path}: manifest {declared} vs file {len(frame)}"
            )
        rows.append(
            {
                "file": csv_path.name,
                "sha256": actual,
                "rows": len(frame),
                "mode": MODE_LABELS.get(csv_path.name, "measured"),
                "model_id": meta.get("model_id", ""),
            }
        )
    return rows


def _num(value: object) -> float | None:
    """Round to 6 decimals; NaN/inf -> None (JSON-safe, deterministic)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), 6)


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def _agg_depth() -> dict[str, dict[str, dict[str, list[list[float]]]]]:
    """Depth-resolved curves: task -> config -> ctx -> [[depth, metric], ...]."""
    out: dict[str, dict[str, dict[str, list[list[float]]]]] = {}
    frames = {"needle": _load("p4_needle_depth_matrix.csv")}
    eco = _load("p5_ecological_baseline.csv")
    for task in ("multihop", "longsum", "longcode"):
        frames[task] = eco[eco.task == task]
    for task, frame in frames.items():
        metric = TASK_METRIC[task][0]
        keep = [c for c in DEPTH_CONFIGS[task] if c in set(frame.config_id)]
        sub = frame[frame.config_id.isin(keep)]
        out[task] = {}
        for config, group in sub.groupby("config_id"):
            out[task][config] = {}
            # Bench formats are repeated measurements of the same (ctx, depth)
            # cell — average them into ONE point per depth. Emitting the raw
            # rows produced duplicate-X sawtooth spikes in the depth chart.
            for (ctx, depth), cell in group.groupby(["context_len", "depth_pct"]):
                out[task][config].setdefault(int(ctx), []).append(
                    [_num(depth), _num(cell[metric].mean())]
                )
            for ctx in out[task][config]:
                out[task][config][ctx].sort(key=lambda p: p[0])
    return out


def _agg_frontier() -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
    """Context-resolved v1.0 frontier: task -> config -> ctx -> stats."""
    frame = _load("p6_v1_frontier_sweep.csv")
    out: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {}
    for task in ("needle", "multihop", "longsum", "longcode"):
        sub = frame[(frame.task == task) & (frame.config_id.isin(FRONTIER_CONFIGS[task]))]
        out[task] = {}
        for config, group in sub.groupby("config_id"):
            out[task][config] = {
                int(ctx): {
                    "acc": _num(ctx_group.acc_mean.mean()),
                    "dlogp": _num(ctx_group.delta_logp.mean()),
                    "vram_mib": _num(ctx_group.theoretical_vram_mib.mean()),
                }
                for ctx, ctx_group in group.groupby("context_len")
            }
    return out


def _agg_rescue() -> dict[str, dict[str, dict[str, dict[str, float | str | None]]]]:
    """v2.0 rescue curves: config -> ctx -> stats (p10 + p11, source-tagged)."""
    out: dict[str, dict[str, dict[str, dict[str, float | str | None]]]] = {}
    for name, tag in (("p10_hadamard_rescue.csv", "p10"), ("p11_head_rescue_curve.csv", "p11")):
        frame = _load(name)
        for row in frame.itertuples():
            config = row.config_id
            entry = out.setdefault(config, {}).setdefault(int(row.context_len), {})
            entry.update(
                {
                    "acc": _num(row.acc_mean),
                    "lo": _num(row.ci95_low),
                    "hi": _num(row.ci95_high),
                    "dlogp": _num(row.delta_logp),
                    "source": tag,
                }
            )
    return out


def _agg_fleet() -> list[dict[str, float | int | bool | str]]:
    frame = _load("p13_ragged_concurrency.csv")
    return [
        {
            "budget": int(r.budget_gib),
            "ctx": int(r.context_len),
            "config": str(r.config),
            "cow": bool(r.cow),
            "sessions": int(r.max_concurrent_sessions),
            "tok_s": _num(r.modeled_decode_tok_s),
            "bytes_per_token": _num(r.bytes_per_token),
            "private_b": int(r.private_bytes_per_session),
            "shared_b": int(r.shared_prefix_bytes),
            "used_b": int(r.used_bytes_at_max),
        }
        for r in frame.itertuples()
    ]


def _agg_agent() -> dict[str, dict[str, dict[str, float | None]]]:
    frame = _load("p14_agent_trajectory_drift.csv")
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for (config, ctx), group in frame.groupby(["config", "context_len"]):
        finals = group[group.turn == group.turn.max()]
        collapses = group.groupby("seed").loop_collapse_turn.first()
        out.setdefault(config, {})[int(ctx)] = {
            "amnesia": _num(group.k_amnesia_incident.mean()),
            "v_schema": _num(group.v_schema_incident.mean()),
            "resolved": _num(finals.resolved.mean()),
            "collapsed": _num((collapses >= 0).mean()),
        }
    return out


def build_payload() -> dict[str, object]:
    """Verify manifests and aggregate all explorer sources deterministically."""
    artifacts = verified_artifacts()
    by_file = {a["file"]: a for a in artifacts}  # type: ignore[index]
    anchor = by_file["p14_agent_trajectory_drift.csv"]
    depth = _agg_depth()
    frontier = _agg_frontier()
    fleet = _agg_fleet()
    agent = _agg_agent()
    return {
        "meta": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "param_count_b": 7.61,
            "q_heads": 28,
            "kv_heads": 4,
            "gqa": "7:1",
            "rotary_dim": 128,
            "latency_model": "meta-llama/Llama-2-7b-hf",
            "artifacts": artifacts,
            "verified": len(artifacts),
        },
        "depth": depth,
        "frontier": frontier,
        "rescue": _agg_rescue(),
        "fleet": fleet,
        "agent": agent,
        "task_metric": {t: {"key": k, "label": lbl} for t, (k, lbl) in TASK_METRIC.items()},
        "avail": {
            "depth": {
                t: sorted(int(c) for c in cfgs[next(iter(cfgs))]) for t, cfgs in depth.items()
            },
            "frontier": {
                t: sorted(int(c) for c in cfgs[next(iter(cfgs))]) for t, cfgs in frontier.items()
            },
            "fleet": sorted({int(f["ctx"]) for f in fleet}),
            "agent": sorted({int(c) for cfg in agent.values() for c in cfg}),
        },
        "anchor_sha": anchor["sha256"],  # type: ignore[index]
    }
