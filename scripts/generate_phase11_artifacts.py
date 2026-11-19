#!/usr/bin/env python3
"""Phase 11: retrieval-head profiler schedule and flagship Figure 4.

Emits:
- bench/data/p11_head_sensitivity_schedule.csv (+ .manifest.json)
- bench/data/p11_head_rescue_curve.csv (+ .manifest.json)
- docs/figures/fig4_hadamard_and_head_rescue.png (+ .png.meta.json)

Honesty notes (also recorded in the manifests):
- No real calibration activations exist in this repository. The schedule is
  profiled on a *declared synthetic bimodal* calibration set at the mandated
  shape (16 sequences x 8K context, 12 layers x 8 KV heads): ground truth is
  ~20% retrieval-biased heads, and the compiler is free to pick its own top
  fraction. A null control on the committed micro_kv_2k.pt fixture (16 query
  rows x 2K context, Gaussian) is committed alongside it.
- Panel B is a replay-adapter projection, not a measured forward pass: the
  Hadamard gain is the committed p10 measurement and the head-adaptive curve
  blends the replay's K8V4 and K4V4 baselines by the compiled schedule's
  retrieval fraction. All projection math is stated in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kv_quant.profiler import HeadProfile, compile_head_schedule, profile_calibration, profile_heads
from tasks.base import RecordedReplayAdapter, SweepManifest, wilson_ci
from tasks.needle import PositionResolvedNeedleTask

DATA_DIR = Path("bench/data")
FIG_DIR = Path("docs/figures")
ACT_DIR = Path("bench/activations")

LAYERS, KV_HEADS = 12, 8
SEQS, Q_LEN, K_LEN = 16, 16, 8192
SINK, LOCAL = 4, 128
RETRIEVAL_GROUND_TRUTH = 0.20  # synthetic mixture: ~20% retrieval-biased heads
TARGET_AVG_BITS = 4.5
CONTEXTS = (4096, 16384, 32768, 65536, 131072)
P10_ROTATION_CSV = DATA_DIR / "p10_hadamard_rotation.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def synthetic_model_manifest() -> SweepManifest:
    """Provenance for synthetic-fixture artifacts (never inherit model defaults)."""
    return SweepManifest.capture(
        model_id="synthetic/bimodal-calibration (no model weights)",
        param_count_b=0.0,
        num_q_heads=KV_HEADS,
        num_kv_heads=KV_HEADS,
        rotary_dim=64,
        seed=42,
    )


def write_manifest(csv_path: Path, description: str, extra: dict) -> None:
    manifest = synthetic_model_manifest()
    manifest.extra = {"description": description, **extra}
    manifest.with_csv_hash(csv_path).write(csv_path.with_suffix(csv_path.suffix + ".manifest.json"))


def bimodal_logits(layer: int, head: int, seq: int) -> torch.Tensor:
    """One calibration sequence for one head: retrieval- or streaming-biased."""
    generator = torch.Generator().manual_seed(2026 + layer * 131 + head * 17 + seq * 3)
    u = (layer * KV_HEADS + head) / (LAYERS * KV_HEADS)
    retrieval = u < RETRIEVAL_GROUND_TRUTH
    distant_scale, local_scale = (3.0, 0.5) if retrieval else (0.5, 3.0)
    distant_span = K_LEN - SINK - LOCAL
    logits = torch.zeros(Q_LEN, K_LEN)
    logits[:, SINK : SINK + distant_span] = (
        torch.randn(Q_LEN, distant_span, generator=generator) * distant_scale
    )
    logits[:, SINK + distant_span :] = torch.randn(Q_LEN, LOCAL, generator=generator) * local_scale
    return logits


def profile_bimodal() -> list[HeadProfile]:
    """Profile the synthetic bimodal calibration set (per-sequence averaging)."""
    per_head: dict[tuple[int, int], list[HeadProfile]] = {}
    for layer in range(LAYERS):
        for head in range(KV_HEADS):
            for seq in range(SEQS):
                probs = torch.softmax(bimodal_logits(layer, head, seq), dim=-1)
                per_head.setdefault((layer, head), []).append(
                    profile_heads(probs.unsqueeze(0), layer_idx=layer)[0]
                )
    profiles = []
    for layer in range(LAYERS):
        for head in range(KV_HEADS):
            seqs = per_head[(layer, head)]
            profiles.append(
                HeadProfile(
                    layer_idx=layer,
                    kv_head_idx=head,
                    distant_attn_mass=sum(p.distant_attn_mass for p in seqs) / SEQS,
                    attn_entropy=sum(p.attn_entropy for p in seqs) / SEQS,
                    kl_sensitivity=sum(p.kl_sensitivity for p in seqs) / SEQS,
                )
            )
    return profiles


def profile_fixture_null_control() -> list[HeadProfile]:
    """Null control: the committed Gaussian fixture has no retrieval structure."""
    fixture = torch.load(ACT_DIR / "micro_kv_2k.pt", map_location="cpu", weights_only=True)
    logits = fixture["attn_logits"].unsqueeze(2)  # (L, H, 1, 16 queries, 2048 keys)
    return profile_calibration(logits, sink_tokens=SINK, local_window=LOCAL)


def schedule_rows(profiles: list[HeadProfile], calibration_set: str) -> tuple[list[dict], dict]:
    frame = pd.DataFrame(
        [
            {
                "layer_idx": p.layer_idx,
                "kv_head_idx": p.kv_head_idx,
                "distant_attn_mass": round(p.distant_attn_mass, 6),
                "attn_entropy": round(p.attn_entropy, 6),
                "kl_perturbation": round(p.kl_sensitivity, 6),
            }
            for p in profiles
        ]
    )
    schedule = compile_head_schedule(frame, target_avg_bits=TARGET_AVG_BITS)
    rows = []
    for _, r in frame.iterrows():
        k_bits, v_bits = schedule[(int(r["layer_idx"]), int(r["kv_head_idx"]))]
        rows.append(
            {
                "calibration_set": calibration_set,
                "layer_idx": int(r["layer_idx"]),
                "kv_head_idx": int(r["kv_head_idx"]),
                "distant_attn_mass": r["distant_attn_mass"],
                "attn_entropy": r["attn_entropy"],
                "kl_perturbation": r["kl_perturbation"],
                "head_class": "retrieval_k8v4" if k_bits == 8 else "streaming_k4v4_hadamard",
                "assigned_k_bits": k_bits,
                "assigned_v_bits": v_bits,
            }
        )
    return rows, schedule


def hadamard_delta_db() -> float:
    """The committed p10 block-diagonal Hadamard INT4 SQNR gain (measured)."""
    df = pd.read_csv(P10_ROTATION_CSV)
    row = df[(df.fixture.str.startswith("micro_kv_2k_outliers")) & (df.scope == "block_diag")]
    return float(row.sqnr_delta_db.iloc[0])


def rescue_curve(delta_db: float, retrieval_fraction: float) -> pd.DataFrame:
    """Replay-adapter projection of the Figure-4 rescue panel.

    Per (context, depth, format) cell, with factor = 10**(-delta_db/10):
      k4v4_hadamard:  err = err(K4V4 post-RoPE) * factor
      head-adaptive:   err = retrieval_fraction * err(K8V4)
                              + (1 - retrieval_fraction) * err(K4V4) * factor
    where err(X) = max(0, acc_fp16 - acc_X) per cell. Positive parts only, so
    cells where a config already matches FP16 contribute nothing.
    """
    factor = 10.0 ** (-delta_db / 10.0)
    task = PositionResolvedNeedleTask(RecordedReplayAdapter(trials_per_cell=20))
    raw = task.to_dataframe(task.run(SweepManifest.capture(seed=42)))
    raw = raw[raw.context_len.isin(CONTEXTS)]
    keys = ["context_len", "depth_pct", "needle_format"]
    base = raw[raw.config_id.isin(["fp16", "k4v4_postrope_sink4", "k8v4"])]
    merged = (
        base[base.config_id == "fp16"]
        .merge(base[base.config_id == "k4v4_postrope_sink4"], on=keys, suffixes=("_f", "_q"))
        .merge(
            base[base.config_id == "k8v4"].rename(
                columns={"acc_mean": "acc_mean_k", "delta_logp": "delta_logp_k"}
            ),
            on=keys,
        )
    )
    err_q = (merged.acc_mean_f - merged.acc_mean_q).clip(lower=0.0)
    err_k = (merged.acc_mean_f - merged.acc_mean_k).clip(lower=0.0)
    merged = merged.assign(
        acc_hadamard=merged.acc_mean_f - err_q * factor,
        acc_adaptive=merged.acc_mean_f
        - retrieval_fraction * err_k
        - (1 - retrieval_fraction) * err_q * factor,
        drift_hadamard=merged.delta_logp_f + (merged.delta_logp_q - merged.delta_logp_f) * factor,
        drift_adaptive=merged.delta_logp_f
        + retrieval_fraction * (merged.delta_logp_k - merged.delta_logp_f)
        + (1 - retrieval_fraction) * (merged.delta_logp_q - merged.delta_logp_f) * factor,
    )

    def _emit(config_id: str, acc_col: str, drift_col: str, note: str) -> list[dict]:
        rows = []
        for ctx in CONTEXTS:
            cells = merged[merged.context_len == ctx]
            trials = int(cells["trials_f"].sum())
            correct = int(round(float((cells[acc_col] * cells["trials_f"]).sum())))
            lo, hi = wilson_ci(correct, trials)
            rows.append(
                {
                    "config_id": config_id,
                    "context_len": ctx,
                    "trials": trials,
                    "acc_mean": round(correct / trials, 4),
                    "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4),
                    "delta_logp": round(float(cells[drift_col].mean()), 4),
                    "projection": note,
                }
            )
        return rows

    out: list[dict] = []
    for ctx in CONTEXTS:
        for cfg in ("fp16", "k4v4_postrope"):
            cells = raw[raw.config_id == (("fp16") if cfg == "fp16" else "k4v4_postrope_sink4")]
            cells = cells[cells.context_len == ctx]
            correct, n = int(cells["correct"].sum()), int(cells["trials"].sum())
            lo, hi = wilson_ci(correct, n)
            out.append(
                {
                    "config_id": cfg,
                    "context_len": ctx,
                    "trials": n,
                    "acc_mean": round(correct / n, 4),
                    "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4),
                    "delta_logp": round(float(cells["delta_logp"].mean()), 4),
                    "projection": "replay-baseline",
                }
            )
    out += _emit(
        "k4v4_hadamard", "acc_hadamard", "drift_hadamard", "projected:p10-delta shift on k4v4"
    )
    out += _emit(
        "head_adaptive_4p5_hadamard",
        "acc_adaptive",
        "drift_adaptive",
        "projected:schedule-weighted blend of k8v4 and hadamard-shifted k4v4",
    )
    return pd.DataFrame(out)


def make_figure(schedule_csv: Path, curve_csv: Path, sched_df: pd.DataFrame) -> Path:
    """Flagship Figure 4: heatmap (A) + rescue curve (B)."""
    bimodal = sched_df[sched_df.calibration_set == "bimodal_16x8k"]
    pivot = bimodal.pivot(index="layer_idx", columns="kv_head_idx", values="distant_attn_mass")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    dpi = 300

    ax = axes[0]
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    retrieval = bimodal[bimodal.head_class == "retrieval_k8v4"]
    ax.scatter(
        retrieval.kv_head_idx,
        retrieval.layer_idx,
        marker="s",
        s=42,
        facecolors="none",
        edgecolors="red",
        linewidths=1.4,
        label="assigned K8V4 (retrieval)",
    )
    ax.set_title("A: Distant Attention Mass (Retrieval vs Streaming Heads)")
    ax.set_xlabel("KV Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_xticks(range(pivot.shape[1]))
    fig.colorbar(im, ax=ax, fraction=0.046, label="Distant mass")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    curve = pd.read_csv(curve_csv)
    styles = {
        "fp16": ("o-", "FP16"),
        "k4v4_postrope": ("s-", "K4V4 post-RoPE (v1.0 cliff)"),
        "k4v4_hadamard": ("^-", "K4V4 + block-diag Hadamard (v1.1)"),
        "head_adaptive_4p5_hadamard": ("d-", "4.5-bit head-adaptive + Hadamard (v1.2)"),
    }
    for config_id, (style, label) in styles.items():
        rows = curve[curve.config_id == config_id]
        ax.plot(rows.context_len / 1024, rows.acc_mean, style, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([c // 1024 for c in CONTEXTS])
    ax.set_xticklabels([f"{c // 1024}K" for c in CONTEXTS])
    ax.set_title("B: The Rescue Curve (replay-adapter projection)")
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Needle Accuracy")
    ax.set_ylim(0.80, 1.01)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "fig4_hadamard_and_head_rescue.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    from PIL import Image

    with Image.open(fig_path) as img:
        width, height = img.size
    meta = {
        "generating_script": "scripts/generate_phase11_artifacts.py",
        "source_csvs": [
            schedule_csv.as_posix(),
            curve_csv.as_posix(),
            P10_ROTATION_CSV.as_posix(),
        ],
        "csv_hashes": [sha256(schedule_csv), sha256(curve_csv), sha256(P10_ROTATION_CSV)],
        "width_px": width,
        "height_px": height,
        "dpi": dpi,
        "panels": [
            "A: Layer x KV-head distant attention mass heatmap with the compiled "
            "K8V4 retrieval assignments outlined",
            "B: Accuracy vs context (4K..128K): v1.0 K4V4 post-RoPE cliff vs K4V4 "
            "+ Hadamard vs 4.5-bit head-adaptive + Hadamard vs FP16",
        ],
        "activation_fixture": ACT_DIR.joinpath("micro_kv_2k.pt").as_posix(),
    }
    fig_path.with_suffix(fig_path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return fig_path


def main() -> int:
    bimodal_rows, schedule = schedule_rows(profile_bimodal(), "bimodal_16x8k")
    null_rows, _ = schedule_rows(profile_fixture_null_control(), "fixture_null_control_2k")
    retrieval_fraction = sum(1 for v in schedule.values() if v == (8, 4)) / len(schedule)

    sched_df = pd.DataFrame(bimodal_rows + null_rows)
    sched_csv = DATA_DIR / "p11_head_sensitivity_schedule.csv"
    sched_df.to_csv(sched_csv, index=False)
    write_manifest(
        sched_csv,
        "Per-head sensitivity metrics and the compiled 4.5-bit mixed-precision schedule",
        {
            "calibration": "declared synthetic bimodal set: 16 sequences x 8K context, "
            f"{LAYERS}x{KV_HEADS} heads, ground truth ~{RETRIEVAL_GROUND_TRUTH:.0%} "
            "retrieval-biased; null control rows profile the committed Gaussian "
            "micro_kv_2k.pt fixture (16 query rows x 2K)",
            "schedule_target_avg_bits": TARGET_AVG_BITS,
            "retrieval_fraction": round(retrieval_fraction, 4),
        },
    )
    print(f"[OK] {sched_csv} (retrieval fraction {retrieval_fraction:.3f})")

    delta_db = hadamard_delta_db()
    curve_df = rescue_curve(delta_db, retrieval_fraction)
    curve_csv = DATA_DIR / "p11_head_rescue_curve.csv"
    curve_df.to_csv(curve_csv, index=False)
    write_manifest(
        curve_csv,
        "Modeled rescue-curve projection: v1.0 cliff vs Hadamard vs head-adaptive",
        {
            "hadamard_delta_sqnr_db": round(delta_db, 3),
            "hadamard_delta_source": P10_ROTATION_CSV.as_posix(),
            "projection": "err(X) = max(0, acc_fp16 - acc_X) per replay cell; "
            "hadamard: err(K4V4) * 10**(-delta_db/10); head-adaptive: "
            "retrieval_fraction * err(K8V4) + (1-retrieval_fraction) * err(K4V4) * factor",
            "retrieval_fraction": round(retrieval_fraction, 4),
        },
    )
    print(f"[OK] {curve_csv}")

    fig_path = make_figure(sched_csv, curve_csv, sched_df)
    print(f"[OK] {fig_path}")
    print(curve_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
