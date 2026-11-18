#!/usr/bin/env python3
"""Generate Flagship Figure 2: K/V precision asymmetry contour + Pareto frontier."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA = Path("bench/data/p6_v1_frontier_sweep.csv")
FIG = Path("docs/figures/fig2_kv_asymmetry_contour.png")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    df = pd.read_csv(DATA)
    manifest = json.loads(DATA.with_suffix(DATA.suffix + ".manifest.json").read_text())
    model_id = manifest["model_id"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    dpi = 300
    kb = [16, 8, 6, 4]
    vb = [16, 8, 6, 4]

    # Panel A: 4x4 K/V asymmetry heatmap at 64K/128K
    ax = axes[0]
    sub = df[(df["task"] == "needle") & (df["context_len"].isin([65536, 131072]))]
    grid = sub.groupby(["k_bits", "v_bits"])["acc_mean"].mean().unstack()
    mat = np.array([[grid.loc[k, v] for v in vb] for k in kb])
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.7, vmax=1.0)
    ax.set_xticks(range(4), [f"V{v}" for v in vb])
    ax.set_yticks(range(4), [f"K{k}" for k in kb])
    ax.set_xlabel("Value bits")
    ax.set_ylabel("Key bits")
    ax.set_title(
        "A: K/V Precision Asymmetry (64K/128K)\nKeys routing-critical, values compressible"
    )
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j] * 100:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Accuracy (%)")
    fp = mat[0, 0]
    k4v8 = mat[3, 2]
    k8v4 = mat[1, 0]
    ax.annotate(
        f"K4V8 -{(fp - k4v8) * 100:.1f}%",
        xy=(2, 3),
        xytext=(0.2, 3.4),
        fontsize=8,
        arrowprops={"arrowstyle": "->"},
    )
    ax.annotate(
        f"K8V4 -{(fp - k8v4) * 100:.2f}%",
        xy=(0, 1),
        xytext=(1.3, 0.6),
        fontsize=8,
        arrowprops={"arrowstyle": "->"},
    )

    # Panel B: Quality vs effective bits/element Pareto frontier across 4 tasks
    ax = axes[1]
    colors = {
        "needle": "#1f77b4",
        "multihop": "#ff7f0e",
        "longsum": "#2ca02c",
        "longcode": "#d62728",
    }
    for task, color in colors.items():
        t = df[df["task"] == task]
        grouped = t.groupby("effective_bits_per_elem")["acc_mean"].mean()
        ax.plot(
            grouped.index,
            grouped.values * 100,
            "o-",
            color=color,
            label=task,
            markersize=4,
            alpha=0.85,
        )
    ax.set_xlabel("Effective bits / KV element")
    ax.set_ylabel("Quality (mean score %)")
    ax.set_title("B: Quality vs Effective Bits/Element Pareto Frontier")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Flagship Figure 2 | K/V Precision Asymmetry | model: {model_id}",
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
        "generating_script": "scripts/plot_kv_asymmetry.py",
        "source_csvs": [DATA.as_posix()],
        "csv_hashes": [sha256(DATA)],
        "width_px": width,
        "height_px": height,
        "dpi": dpi,
        "panels": [
            "A: 4x4 K-bits x V-bits accuracy contour at 64K/128K showing K4V8 -20% and K8V4 -0.37%",
            "B: Quality vs effective bits/element Pareto frontier across needle, multihop, longsum, longcode",
        ],
        "model_id": model_id,
    }
    FIG.with_suffix(FIG.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] {FIG} ({width}x{height} @ {dpi} DPI)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
