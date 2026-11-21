#!/usr/bin/env python3
"""Render Flagship Figure 5: agent amnesia vs syntax drift (ADR-012).

Panel A1: per-turn key distortion (the K channel), against the measured routing
          threshold -- the smallest post-quantization retrieval margin observed.
Panel A2: per-turn failure incidence at 64K (solid = K-amnesia, dashed = V-schema).
Panel B : end-to-end 20-turn resolution rate vs KV memory per session.

Source of truth: ``bench/data/p14_agent_trajectory_drift.csv`` (+ manifest).
Memory is derived from the pinned ``kv_quant.layout`` equation at Qwen2.5-7B
KV geometry (32 layers x 4 KV heads x 128 head_dim = 16384 elems/token).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from kv_quant.layout import bytes_per_element  # noqa: E402
from tasks.agent_trajectory import CONFIGS  # noqa: E402

DATA = Path("bench/data/p14_agent_trajectory_drift.csv")
META = Path("docs/figures/fig5_agent_amnesia_vs_syntax.png.meta.json")
OUT = Path("docs/figures/fig5_agent_amnesia_vs_syntax.png")
LAYERS, KV_HEADS, HEAD_DIM = 32, 4, 128
ELEMS = LAYERS * KV_HEADS * HEAD_DIM

COLORS = {
    "fp16": "#333333",
    "k8v4": "#1f77b4",
    "k4v8_postrope": "#d62728",
    "k8v2_asym": "#ff7f0e",
    "k4v4_hadamard": "#2ca02c",
}
LABELS = {
    "fp16": "FP16",
    "k8v4": "K8V4",
    "k4v8_postrope": "K4V8 (post-RoPE)",
    "k8v2_asym": "K8V2 (asym)",
    "k4v4_hadamard": "K4V4 + Hadamard",
}


def kv_mib(ctx: int, name: str) -> float:
    """Per-session KV footprint in MiB at a given context length."""
    spec = next(c for c in CONFIGS if c.name == name)
    per_elem = bytes_per_element(spec.k_bits, 64, False) + bytes_per_element(
        spec.v_bits, 64, spec.v_scheme == "asym"
    )
    return ELEMS * per_elem * ctx / (1024**2)


def lf_sha256(path: Path) -> str:
    """Line-ending-normalized hash (matches check_figure_provenance.py)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def panel_a1(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Key distortion per turn against the smallest observed retrieval margin."""
    sub = df[df.context_len == 65536]
    for name, color in COLORS.items():
        by_turn = sub[sub.config == name].groupby("turn").key_distortion.mean()
        ax.plot(by_turn.index + 1, by_turn, color=color, lw=2, label=LABELS[name])
    # Turn 0 has a single cached file, so target-vs-other is undefined there.
    # The 1st percentile (not the minimum) is the honest "how close does routing
    # normally come to flipping" statistic; a handful of ties sit near zero.
    threshold = float(sub[sub.turn > 0].retrieval_margin.quantile(0.01))
    ax.axhline(
        threshold,
        color="grey",
        ls="--",
        lw=1.4,
        label=f"1st-pct retrieval margin ({threshold:.3f})",
    )
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_title("A1. Key distortion vs 1st-pct retrieval margin @ 64K", fontsize=9, loc="left")
    ax.set_xlabel("Turn")
    ax.set_ylabel("mean distortion (1 - cos)")
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.0, 0.03)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=6.5, loc="center left", framealpha=0.92)


def panel_a2(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Outcome incidence: K-amnesia (solid) vs V-schema corruption (dashed)."""
    sub = df[df.context_len == 65536]
    for name, color in COLORS.items():
        by_turn = (
            sub[sub.config == name]
            .groupby("turn")[["k_amnesia_incident", "v_schema_incident"]]
            .mean()
        )
        ax.plot(by_turn.index + 1, by_turn.k_amnesia_incident, color=color, lw=2)
        ax.plot(by_turn.index + 1, by_turn.v_schema_incident, color=color, lw=1.5, ls="--")
    ax.set_title("A2. Incidence @ 64K (solid K-amnesia, dashed V-schema)", fontsize=9, loc="left")
    ax.set_xlabel("Turn")
    ax.set_ylabel("fraction of runs")
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)


def panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    """End-to-end resolution against KV memory: the serving trade-off."""
    for name, color in COLORS.items():
        sub = df[df.config == name]
        xs, ys = [], []
        for ctx in (32768, 65536):
            rows = sub[sub.context_len == ctx]
            ys.append(rows.groupby("seed").resolved.max().mean())
            xs.append(kv_mib(ctx, name))
        ax.plot(xs, ys, marker="o", color=color, lw=2.0, label=LABELS[name])
        dy = -11 if ys[-1] > 0.95 else 5
        ax.annotate(
            f"{ys[-1]:.0%}",
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(5, dy),
            fontsize=7,
            color=color,
        )
    ax.set_title("B. 20-turn resolution vs KV memory", fontsize=9, loc="left")
    ax.set_xlabel("KV cache per session (MiB)  [32K -> 64K]")
    ax.set_ylabel("task resolved rate")
    ax.set_ylim(-0.08, 1.14)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, loc="lower left")


def main() -> int:
    df = pd.read_csv(DATA)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.5, 1.5, 1.2]})
    for ax in axes:
        ax.tick_params(labelsize=7.5)
    panel_a1(axes[0], df)
    panel_a2(axes[1], df)
    panel_b(axes[2], df)
    fig.suptitle(
        "Figure 5 - K-amnesia vs V-syntax drift under KV-cache quantization",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    from PIL import Image

    with Image.open(OUT) as img:
        width, height = img.size
    META.write_text(
        json.dumps(
            {
                "generating_script": "scripts/generate_phase14_artifacts.py",
                "source_csvs": [DATA.as_posix()],
                "csv_hashes": [lf_sha256(DATA)],
                "width_px": width,
                "height_px": height,
                "dpi": 300,
                "panels": [
                    "A1: per-turn key distortion vs the minimum observed retrieval margin",
                    "A2: per-turn K-amnesia (solid) and V-schema (dashed) incidence at 64K",
                    "B: 20-turn resolution rate vs KV MiB per session at 32K/64K",
                ],
                "memory_model": (
                    "kv_quant.layout.bytes_per_element at 32 layers x 4 KV heads x 128 head_dim"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({width}x{height} @300dpi) and {META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
