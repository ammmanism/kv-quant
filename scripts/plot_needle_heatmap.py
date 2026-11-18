#!/usr/bin/env python3
"""Generate Flagship Figure 1: needle depth heatmap + Delta-logP drift sensor."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("bench/data")
FIG_DIR = Path("docs/figures")
CSV = DATA_DIR / "p4_needle_depth_matrix.csv"
MANIFEST = DATA_DIR / "p4_needle_depth_matrix.csv.manifest.json"
FIG = FIG_DIR / "fig1_needle_depth_heatmap.png"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def pivot_accuracy(df: pd.DataFrame, config: str) -> pd.DataFrame:
    sub = df[df["config_id"] == config]
    return sub.pivot_table(
        index="context_len", columns="depth_pct", values="acc_mean", aggfunc="mean"
    ).sort_index()


def main() -> int:
    df = pd.read_csv(CSV)
    manifest = load_manifest()
    model_id = manifest["model_id"]
    params = manifest["param_count_b"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    dpi = 300

    contexts = sorted(df["context_len"].unique())
    depths = sorted(df["depth_pct"].unique())
    ctx_labels = [f"{c // 1024}K" for c in contexts]

    # Panel A: K4V4 postrope sink4 -- late-context depth cliff
    ax = axes[0]
    a = pivot_accuracy(df, "k4v4_postrope_sink4")
    im = ax.imshow(a.values, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=1.0)
    ax.set_title("A: K4V4 (Post-RoPE, sink=4)\nAccuracy vs Context x Depth")
    ax.set_xlabel("Insertion depth (%)")
    ax.set_ylabel("Context length (tokens)")
    ax.set_xticks(range(len(depths)), depths, rotation=45)
    ax.set_yticks(range(len(ctx_labels)), ctx_labels)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Accuracy")
    ax.annotate(
        "depth cliff\n70-100% @ >=32K",
        xy=(len(depths) - 2, 2.5),
        xycoords="data",
        fontsize=8,
        color="black",
        ha="center",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7},
    )

    # Panel B: K8V4 -- flat retention
    ax = axes[1]
    b = pivot_accuracy(df, "k8v4")
    im2 = ax.imshow(b.values, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=1.0)
    ax.set_title("B: K8V4 (Flat Retention)\nAccuracy vs Context x Depth")
    ax.set_xlabel("Insertion depth (%)")
    ax.set_ylabel("Context length (tokens)")
    ax.set_xticks(range(len(depths)), depths, rotation=45)
    ax.set_yticks(range(len(ctx_labels)), ctx_labels)
    fig.colorbar(im2, ax=ax, fraction=0.046, label="Accuracy")

    # Panel C: continuous Delta-logP drift at 64K and 128K
    ax = axes[2]
    for ctx, color in [(65536, "#d62728"), (131072, "#1f77b4")]:
        sub = df[(df["context_len"] == ctx) & (df["config_id"] == "k4v4_postrope_sink4")]
        grouped = sub.groupby("depth_pct")["delta_logp"].mean()
        ax.plot(grouped.index, grouped.values, "o-", color=color, label=f"{ctx // 1024}K")
    ax.axhline(1.2, color="black", linestyle="--", linewidth=1, label="1.2 nat cliff")
    ax.set_title("C: Continuous RoPE Phase Decay\nDelta-logP vs Insertion Depth")
    ax.set_xlabel("Insertion depth decile (%)")
    ax.set_ylabel("Delta-logP (nats): FP16 - Quant")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Position-Resolved Needle Probe | model: {model_id} | {params}B params "
        f"| GQA {manifest['num_q_heads']}Q/{manifest['num_kv_heads']}KV "
        f"| rotary_dim {manifest['rotary_dim']}",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    from PIL import Image

    with Image.open(FIG) as img:
        width, height = img.size

    meta = {
        "generating_script": "scripts/plot_needle_heatmap.py",
        "source_csvs": [CSV.as_posix()],
        "csv_hashes": [sha256(CSV)],
        "width_px": width,
        "height_px": height,
        "dpi": dpi,
        "panels": [
            "A: K4V4 post-RoPE context x depth accuracy heatmap exposing late-context cliff",
            "B: K8V4 context x depth accuracy heatmap showing flat retention",
            "C: Delta-logP drift vs insertion depth decile at 64K and 128K with model subtitle",
        ],
        "model_id": model_id,
        "param_count_b": params,
    }
    FIG.with_suffix(FIG.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] {FIG} ({width}x{height} @ {dpi} DPI)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
