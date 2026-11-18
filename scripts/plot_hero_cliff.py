#!/usr/bin/env python3
"""Generate Flagship Figure 0: the repo hero cliff chart.

Left: accuracy vs context for 7 K/V configurations with Wilson 95% CI bands
(Bernoulli-discretized cell means; 20 trials/cell x 2 fast tasks = n=40).
Right: quality-retained vs KV VRAM compression Pareto frontier at 64K/128K.

All data from the committed bench/data/p6_v1_frontier_sweep.csv; compression
ratios from kv_quant.vram.kv_cache_calculator_bytes (Qwen2.5-7B geometry).
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kv_quant.vram import kv_cache_calculator_bytes  # noqa: E402
from tasks.base import wilson_ci  # noqa: E402

FRONTIER = Path("bench/data/p6_v1_frontier_sweep.csv")
FIG = Path("docs/figures/fig0_hero_cliff.png")
TRIALS_PER_CELL = 20  # v1-frontier sweep manifest

NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM = 28, 4, 128  # Qwen2.5-7B geometry

# (label, k_bits, v_bits, stage, sink_tokens, k_per_channel, color, style)
HERO_CONFIGS = [
    ("FP16", 16, 16, "post_rope", 4, False, "#111111", "-"),
    ("K8V8", 8, 8, "post_rope", 4, False, "#1f77b4", "-"),
    ("K8V4 (asym V)", 8, 4, "post_rope", 4, False, "#2ca02c", "-"),
    ("K4V8", 4, 8, "post_rope", 4, False, "#ff7f0e", "-"),
    ("K4V4 post-rope, no sinks", 4, 4, "post_rope", 0, False, "#d62728", "--"),
    ("K4V4 post-rope + 4 FP16 sinks", 4, 4, "post_rope", 4, False, "#9467bd", "-"),
    ("K4V4 pre-rope per-channel", 4, 4, "pre_rope", 4, True, "#8c564b", ":"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def hero_series(df: pd.DataFrame, kb: int, vb: int, stage: str, sink: int, pc: bool):
    """Pooled (needle+multihop) mean acc and Wilson 95% CI per context."""
    sub = df[
        (df["task"].isin(["needle", "multihop"]))
        & (df["k_bits"] == kb)
        & (df["v_bits"] == vb)
        & (df["k_quant_stage"] == stage)
        & (df["sink_tokens"] == sink)
        & (df["k_per_channel"] == pc)
        & (df["k_encoding"] == "int")
    ]
    grouped = sub.groupby("context_len")["acc_mean"].mean()
    n = TRIALS_PER_CELL * sub.groupby("context_len").size()
    lows, highs = [], []
    for ctx, mean in grouped.items():
        k = int(round(mean * n[ctx]))
        lo, hi = wilson_ci(k, int(n[ctx]))
        lows.append(lo)
        highs.append(hi)
    return grouped.index.to_list(), grouped.to_list(), lows, highs


def _panel_left(ax: plt.Axes, df: pd.DataFrame) -> float:
    """Accuracy vs context with Wilson bands + data-true cliff callout."""
    for label, kb, vb, stage, sink, pc, color, style in HERO_CONFIGS:
        ctxs, accs, lows, highs = hero_series(df, kb, vb, stage, sink, pc)
        ax.plot(
            ctxs,
            [a * 100 for a in accs],
            style,
            color=color,
            label=label,
            marker="o",
            markersize=4,
            linewidth=1.8,
        )
        ax.fill_between(
            ctxs, [lo * 100 for lo in lows], [hi * 100 for hi in highs], color=color, alpha=0.08
        )
    k4_ctxs, k4_accs, _, _ = hero_series(df, 4, 4, "post_rope", 4, False)
    _, fp_accs, _, _ = hero_series(df, 16, 16, "post_rope", 4, False)
    drop = (fp_accs[0] - k4_accs[0]) / fp_accs[0] * 100
    ax.annotate(
        f"INT4-K Cliff: -{drop:.1f}% vs FP16\nonset <= 4K, flat through 128K\n(config-level, not context-resolved)",
        xy=(k4_ctxs[2], k4_accs[2] * 100),
        xytext=(12000, 84.5),
        fontsize=8,
        fontweight="bold",
        color="#d62728",
        arrowprops={"arrowstyle": "->", "color": "#d62728"},
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d62728"},
    )
    ax.set_ylim(59, 101)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4096, 16384, 32768, 65536, 131072], ["4K", "16K", "32K", "64K", "128K"])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Accuracy (%, pooled needle+multihop)")
    ax.set_title("A: The INT4-K Cliff and its Remedies\n(shaded = Wilson 95% CI, n=40/cell)")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(True, alpha=0.3)
    return drop


def _panel_right(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Quality-retained vs compression Pareto frontier at 64K/128K."""
    markers = {65536: "o", 131072: "s"}
    pareto: list[tuple[float, float]] = []
    for ctx, marker in ((65536, "o"), (131072, "s")):
        fp_acc = df[
            (df.task.isin(["needle", "multihop"]))
            & (df.k_bits == 16)
            & (df.v_bits == 16)
            & (df.context_len == ctx)
            & (df.k_quant_stage == "post_rope")
            & (df.sink_tokens == 4)
            & (~df.k_per_channel)
            & (df.k_encoding == "int")
        ]["acc_mean"].mean()
        fp_kv = kv_cache_calculator_bytes(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, ctx, 16, 16)
        pts = []
        for label, kb, vb, stage, sink, pc, color, _ in HERO_CONFIGS[:5]:
            sub = df[
                (df["task"].isin(["needle", "multihop"]))
                & (df["k_bits"] == kb)
                & (df["v_bits"] == vb)
                & (df["k_quant_stage"] == stage)
                & (df["sink_tokens"] == sink)
                & (df["k_per_channel"] == pc)
                & (df["k_encoding"] == "int")
                & (df["context_len"] == ctx)
            ]
            acc = sub["acc_mean"].mean()
            kv = kv_cache_calculator_bytes(
                NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, ctx, kb, vb, sink_tokens=sink
            )
            ratio, retained = fp_kv / kv, acc / fp_acc * 100
            pts.append((ratio, retained))
            ax.scatter(ratio, retained, marker=marker, s=60, color=color, alpha=0.9)
            offset = (6, -14) if marker == "o" else (6, 10)
            if ratio > 3.5:
                offset = (-38, offset[1])
            ax.annotate(
                label.split(" ")[0],
                (ratio, retained),
                textcoords="offset points",
                xytext=offset,
                fontsize=6,
            )
            ax.annotate(
                f"{ratio:.2f}x",
                (ratio, retained),
                textcoords="offset points",
                xytext=(offset[0], offset[1] + 7),
                fontsize=5.5,
                alpha=0.8,
            )
        best = -1.0
        for ratio, retained in sorted(pts):
            if retained > best:
                best = retained
                pareto.append((ratio, retained))
    steps = sorted(set(pareto))
    ax.step(
        [s[0] for s in steps],
        [s[1] for s in steps],
        where="post",
        color="#111111",
        linewidth=1.5,
        alpha=0.6,
        label="Pareto frontier",
    )
    ax.set_xlabel("KV VRAM compression ratio (x vs FP16, Qwen2.5-7B geometry)")
    ax.set_ylabel("Quality retained (% of FP16)")
    ax.set_ylim(78, 101.8)
    ax.set_xlim(0.85, 4.2)
    ax.set_title("B: Quality vs Compression Pareto Frontier\n(o = 64K, square = 128K)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def main() -> int:
    df = pd.read_csv(FRONTIER)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    dpi = 300
    drop = _panel_left(axes[0], df)
    _panel_right(axes[1], df)

    # Caption mandate (Invariant 2): model identity on the canvas
    fig.suptitle(
        "Flagship Figure 0 | The Hero Cliff Chart: Keys govern routing, Values ride along",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Model: Qwen2.5-7B-Instruct (7.6B, GQA 28Q/4KV, rotary_dim=128)  |  "
        "Data: bench/data/p6_v1_frontier_sweep.csv (modeled sweep; see .manifest.json)",
        ha="center",
        fontsize=8,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.94))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    from PIL import Image

    with Image.open(FIG) as img:
        width, height = img.size

    meta = {
        "generating_script": "scripts/plot_hero_cliff.py",
        "source_csvs": [FRONTIER.as_posix()],
        "csv_hashes": [sha256(FRONTIER)],
        "width_px": width,
        "height_px": height,
        "dpi": dpi,
        "panels": [
            "A: accuracy vs context (4K-128K) for 7 K/V configs with Wilson 95% CI bands "
            "(Bernoulli-discretized cell means, 20 trials/cell x 2 fast tasks = n=40); "
            "INT4-K cliff callout is config-level (-20.2% vs FP16, onset <=4K, flat through 128K)",
            "B: quality retained (%) vs KV VRAM compression ratio Pareto frontier at 64K and 128K "
            "(compression from kv_cache_calculator_bytes, Qwen2.5-7B geometry)",
        ],
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "wilson_ci_derivation": (
            "acc_mean per cell is the mean of 20 seeded trials (v1-frontier manifest); "
            "pooling 2 fast tasks gives n=40 with k=round(acc_mean*40); "
            "bands are conservative because scores are continuous"
        ),
    }
    FIG.with_suffix(FIG.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] {FIG} ({width}x{height} @ {dpi} DPI) cliff drop {drop:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
