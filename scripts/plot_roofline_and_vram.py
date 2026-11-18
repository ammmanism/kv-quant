#!/usr/bin/env python3
"""Generate Flagship Figure 3: NCU roofline + TTFT/ITL + VRAM cross-validation.

Panel A: theoretical vs ncu-modeled DRAM traffic (naive vs fused) with decode
ITL speedup overlay across context lengths.
Panel B: kv-cache-calculator predicted vs measured VRAM scatter + regression
fit (R^2, MAPE).
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BW_CSV = Path("bench/data/p7_ncu_bandwidth.csv")
LAT_CSV = Path("bench/data/p7_latency_ttft_itl.csv")
VRAM_CSV = Path("bench/data/p7_vram_crossval.csv")
FIG = Path("docs/figures/fig3_roofline_and_vram.png")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _panel_a(ax: plt.Axes, bw: pd.DataFrame, lat: pd.DataFrame) -> None:
    """Theory vs ncu-modeled HBM traffic (k4v4) + ITL speedup overlay."""
    k4 = bw[bw["config"] == "k4v4"]
    seqs = sorted(k4["seq_len"].unique())
    width = 0.35
    x = np.arange(len(seqs))
    for kernel, offset, color, hatch in (
        ("naive", -width / 2, "#d62728", "//"),
        ("fused", width / 2, "#1f77b4", ""),
    ):
        sub = k4[k4["kernel"] == kernel].set_index("seq_len").loc[seqs]
        theo = sub["theoretical_total_bytes"].to_numpy() / 1e6
        meas = sub["dram_total_bytes"].to_numpy() / 1e6
        ax.bar(
            x + offset,
            meas,
            width,
            color=color,
            hatch=hatch,
            alpha=0.9,
            label=f"{kernel} measured (ncu-modeled)",
        )
        ax.plot(
            x + offset,
            theo,
            "o",
            color="black",
            markersize=4,
            label="theory" if kernel == "naive" else None,
        )
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{s // 1024}K" for s in seqs])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("DRAM traffic per decode pass (MB, log)")
    ax.set_title("A: Theory vs ncu-Measured HBM Traffic (k4v4)\nFused never materializes FP16 K/V")
    ax2 = ax.twinx()
    speed = lat[lat["config"] == "k4v4"].set_index("context_len").loc[seqs]["itl_speedup_p50"]
    ax2.plot(x, speed, "s--", color="#2ca02c", label="decode ITL speedup")
    ax2.set_ylabel("Decode ITL speedup (naive/fused, x)", color="#2ca02c")
    ax2.set_ylim(0, max(speed) * 1.25)
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, fontsize=7, loc="upper left")


def _panel_b(ax: plt.Axes, vr: pd.DataFrame) -> None:
    """kv-cache-calculator predicted vs measured VRAM scatter + OLS fit."""
    sizes = {"4096": 25, "16384": 55}
    for seq, marker in ((4096, "o"), (16384, "s")):
        sub = vr[vr["seq_len"] == seq]
        ax.scatter(
            sub["calculator_predicted_bytes"] / 2**20,
            sub["measured_peak_bytes"] / 2**20,
            marker=marker,
            s=sizes[str(seq)],
            alpha=0.85,
            label=f"seq {seq // 1024}K",
        )
    all_x = vr["calculator_predicted_bytes"].to_numpy(dtype=float) / 2**20
    all_y = vr["measured_peak_bytes"].to_numpy(dtype=float) / 2**20
    lo, hi = min(all_x.min(), all_y.min()), max(all_x.max(), all_y.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    slope, intercept = np.polyfit(all_x, all_y, 1)
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, slope * xs + intercept, "-", color="#ff7f0e", alpha=0.7, label="OLS fit")
    r2 = float(vr["r_squared"].iloc[0])
    mape = float(vr["mape_pct"].iloc[0])
    ax.set_xlabel("kv-cache-calculator predicted (MiB)")
    ax.set_ylabel("Measured peak (MiB)")
    ax.set_title(f"B: VRAM Cross-Validation (20 configs)\n$R^2$={r2:.4f}, MAPE={mape:.3f}%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> int:
    bw = pd.read_csv(BW_CSV)
    lat = pd.read_csv(LAT_CSV)
    vr = pd.read_csv(VRAM_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    dpi = 300
    _panel_a(axes[0], bw, lat)
    _panel_b(axes[1], vr)

    r2 = float(vr["r_squared"].iloc[0])
    mape = float(vr["mape_pct"].iloc[0])
    fig.suptitle(
        "Flagship Figure 3 | Hardware Truth: NCU Roofline, TTFT/ITL, VRAM Cross-Val",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    from PIL import Image

    with Image.open(FIG) as img:
        width_px, height_px = img.size

    meta = {
        "generating_script": "scripts/plot_roofline_and_vram.py",
        "source_csvs": [BW_CSV.as_posix(), LAT_CSV.as_posix(), VRAM_CSV.as_posix()],
        "csv_hashes": [sha256(BW_CSV), sha256(LAT_CSV), sha256(VRAM_CSV)],
        "width_px": width_px,
        "height_px": height_px,
        "dpi": dpi,
        "panels": [
            "A: theoretical vs ncu-modeled DRAM traffic (naive vs fused, k4v4, log scale) "
            "with decode ITL speedup overlay across 4K-128K contexts",
            "B: kv-cache-calculator predicted vs measured peak VRAM scatter with OLS fit "
            "(R^2 and MAPE annotated) across 20 K/V configs",
        ],
        "model_id": "meta-llama/Llama-2-7b-hf (A), Qwen/Qwen2.5-7B-Instruct (B)",
    }
    FIG.with_suffix(FIG.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] {FIG} ({width_px}x{height_px} @ {dpi} DPI) R^2={r2:.4f} MAPE={mape:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
