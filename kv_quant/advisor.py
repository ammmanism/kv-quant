"""CSV-backed Pareto advisor: decision-ready quantization recommendations.

Queries the committed sweep artifacts in ``bench/data/`` (never live models)
and returns configurations that satisfy user constraints:

- quality drop vs FP16 at the requested context (from ``p6_v1_frontier_sweep``)
- KV VRAM budget for the requested model profile (``kv_cache_calculator_bytes``)
- expected decode ITL speedup (from ``p7_latency_ttft_itl``; k4v8 is mapped to
  the k8v4 row: identical effective bits/element -> identical modeled traffic)

CLI: ``python -m kv_quant.advisor --context-len 128000 --task multihop
--max-drop-pct 1.0 --vram-budget-gb 24``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from kv_quant.config import QuantConfig
from kv_quant.vram import kv_cache_calculator_bytes

FRONTIER_CSV = Path("bench/data/p6_v1_frontier_sweep.csv")
LATENCY_CSV = Path("bench/data/p7_latency_ttft_itl.csv")

# Model profiles: layer/head geometry from the sweep manifests (Qwen2.5-7B).
MODEL_PROFILES: dict[str, dict[str, float | int]] = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "num_layers": 28,
        "num_q_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "param_count_b": 7.61,
    },
}

# Candidate configurations (label, k_bits, v_bits, stage, sink_tokens, p7 key).
CANDIDATES: list[tuple[str, int, int, str, int, str]] = [
    ("FP16", 16, 16, "post_rope", 4, "fp16"),
    ("K8V8", 8, 8, "post_rope", 4, "k8v8"),
    ("K8V4 (asymmetric V)", 8, 4, "post_rope", 4, "k8v4"),
    ("K4V8", 4, 8, "post_rope", 4, "k8v4"),  # proxy: same effective bits/elem
    ("K4V4 post-rope + 4 FP16 sinks", 4, 4, "post_rope", 4, "k4v4"),
]


@dataclass(frozen=True)
class AdvisorRecommendation:
    """One decision row: quality, VRAM, and speed for a quantization config."""

    label: str
    quant_config: QuantConfig
    context_len: int
    task: str
    acc_mean: float
    drop_pct: float
    kv_bytes: int
    compression_ratio: float
    itl_speedup: float


@lru_cache(maxsize=1)
def _frontier() -> pd.DataFrame:
    return pd.read_csv(FRONTIER_CSV)


@lru_cache(maxsize=1)
def _latency() -> pd.DataFrame:
    return pd.read_csv(LATENCY_CSV)


def _nearest_context(df: pd.DataFrame, context_len: int) -> int:
    contexts = sorted(df["context_len"].unique())
    return min(contexts, key=lambda c: abs(c - context_len))


def fp16_baseline(task: str, context_len: int) -> tuple[float, int]:
    """FP16 quality at the nearest committed context (like-for-like variants)."""
    df = _frontier()
    fp16 = df[
        (df["task"] == task)
        & (df["k_bits"] == 16)
        & (df["v_bits"] == 16)
        & (df["k_per_channel"] == False)  # noqa: E712
        & (df["k_encoding"] == "int")
        & (df["k_quant_stage"] == "post_rope")
        & (df["sink_tokens"] == 4)
    ]
    ctx = _nearest_context(fp16, context_len)
    sub = fp16[fp16["context_len"] == ctx]
    return float(sub["acc_mean"].mean()), int(ctx)


def _candidate_rows(task: str, ctx: int, kb: int, vb: int, stage: str, sink: int) -> pd.DataFrame:
    df = _frontier()
    return df[
        (df["task"] == task)
        & (df["k_bits"] == kb)
        & (df["v_bits"] == vb)
        & (df["k_quant_stage"] == stage)
        & (df["sink_tokens"] == sink)
        & (df["k_per_channel"] == False)  # noqa: E712
        & (df["k_encoding"] == "int")
        & (df["context_len"] == ctx)
    ]


def recommend(
    context_len: int,
    task: str,
    max_drop_pct: float,
    vram_budget_gb: float,
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
) -> list[AdvisorRecommendation]:
    """Return configurations satisfying the quality and VRAM constraints."""
    profile = MODEL_PROFILES[model_id]
    num_layers = int(profile["num_layers"])
    num_kv_heads = int(profile["num_kv_heads"])
    head_dim = int(profile["head_dim"])
    channels = num_kv_heads * head_dim

    fp16_acc, ctx = fp16_baseline(task, context_len)
    fp16_kv = kv_cache_calculator_bytes(
        num_layers, num_kv_heads, head_dim, ctx, 16, 16, sink_tokens=4
    )
    kv_budget = vram_budget_gb * (1 << 30) - profile["param_count_b"] * 2 * 1e9

    lat = _latency()
    recs: list[AdvisorRecommendation] = []
    for label, kb, vb, stage, sink, p7_key in CANDIDATES:
        rows = _candidate_rows(task, ctx, kb, vb, stage, sink)
        if rows.empty:
            continue
        acc = float(rows["acc_mean"].mean())
        drop = (fp16_acc - acc) / fp16_acc * 100.0
        if drop > max_drop_pct:
            continue
        kv_bytes = kv_cache_calculator_bytes(
            num_layers, num_kv_heads, head_dim, ctx, kb, vb, sink_tokens=sink
        )
        if kv_budget <= 0 or kv_bytes > kv_budget:
            continue
        lat_row = lat[(lat["config"] == p7_key) & (lat["context_len"] == ctx)]
        itl = float(lat_row["itl_speedup_p50"].iloc[0]) if not lat_row.empty else 1.0
        recs.append(
            AdvisorRecommendation(
                label=label,
                quant_config=QuantConfig(  # type: ignore[arg-type]
                    k_bits=kb,  # type: ignore[arg-type]
                    v_bits=vb,  # type: ignore[arg-type]
                    k_quant_stage=stage,  # type: ignore[arg-type]
                    sink_tokens=sink,
                ),
                context_len=ctx,
                task=task,
                acc_mean=acc,
                drop_pct=round(drop, 3),
                kv_bytes=kv_bytes,
                compression_ratio=round(fp16_kv / kv_bytes, 3),
                itl_speedup=round(itl, 3),
            )
        )
    recs.sort(key=lambda r: -r.compression_ratio)
    return recs


def render_markdown(recs: list[AdvisorRecommendation], fp16_acc: float, ctx: int) -> str:
    """Render the decision table with the recommended QuantConfig."""
    lines = [
        f"## KV-Quant Advisor - task=`{recs[0].task if recs else '?'}` "
        f"context={ctx} (FP16 baseline acc={fp16_acc:.4f})",
        "",
        "| Config | acc | drop vs FP16 | KV VRAM | compression | ITL speedup |",
        "|---|---|---|---|---|---|",
    ]
    for r in recs:
        lines.append(
            f"| {r.label} | {r.acc_mean:.4f} | {r.drop_pct:+.2f}% | "
            f"{r.kv_bytes / (1 << 30):.2f} GiB | {r.compression_ratio:.2f}x | "
            f"{r.itl_speedup:.2f}x |"
        )
    if recs:
        best = recs[0]
        q = best.quant_config
        lines += [
            "",
            f"**Recommended `QuantConfig`:** `k_bits={q.k_bits}, v_bits={q.v_bits}, "
            f"k_quant_stage='{q.k_quant_stage}', sink_tokens={q.sink_tokens}, "
            f"group_size={q.group_size}`",
            "",
            "Quality/VRAM/speed figures derive from the committed sweep artifacts "
            "(`bench/data/p6_v1_frontier_sweep.csv`, `bench/data/p7_latency_ttft_itl.csv`) "
            "at the nearest committed context; ITL speedups are modeled (see p7 manifest).",
        ]
    else:
        lines.append("No configuration satisfies the constraints.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="CSV-backed KV quantization advisor")
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument(
        "--task", required=True, choices=["needle", "multihop", "longsum", "longcode"]
    )
    parser.add_argument("--max-drop-pct", type=float, required=True)
    parser.add_argument("--vram-budget-gb", type=float, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    fp16_acc, ctx = fp16_baseline(args.task, args.context_len)
    recs = recommend(
        args.context_len, args.task, args.max_drop_pct, args.vram_budget_gb, args.model_id
    )
    if not recs:
        print(
            f"No configuration satisfies drop<={args.max_drop_pct}% and "
            f"VRAM<={args.vram_budget_gb} GiB at context {args.context_len}."
        )
        return 1
    print(render_markdown(recs, fp16_acc, ctx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
