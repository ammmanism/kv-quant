#!/usr/bin/env python3
"""Generate Phase 2 committed datasets, activation fixture and figure artifact."""

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kv_quant.profiler import HeadProfile, compile_head_bit_schedule, profile_heads

DATA_DIR = Path("bench/data")
ACT_DIR = Path("bench/activations")
FIG_DIR = Path("docs/figures")
for d in (DATA_DIR, ACT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    """Line-ending-normalized SHA256 (cross-platform stable)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_manifest(csv_path: Path, description: str) -> None:
    df = pd.read_csv(csv_path)
    manifest = {
        "file": csv_path.name,
        "sha256": sha256(csv_path),
        "description": description,
        "generated_by": "scripts/generate_phase2_artifacts.py",
        "columns": list(df.columns),
        "num_rows": len(df),
    }
    csv_path.with_suffix(csv_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def generate_fixture() -> Path:
    """Real-model-style micro activation fixture (2k tokens)."""
    torch.manual_seed(2026)
    seq, heads, dim, layers = 2048, 8, 64, 4
    fixture = {
        "k": torch.randn(layers, seq, heads, dim, dtype=torch.float16),
        "v": torch.randn(layers, seq, heads, dim, dtype=torch.float16),
        "attn_logits": torch.randn(layers, heads, 16, seq, dtype=torch.float16),
        "metadata": {
            "seq_len": seq,
            "num_kv_heads": heads,
            "head_dim": dim,
            "num_layers": layers,
            "source": "synthetic micro-activation fixture (deterministic seed 2026)",
            "dtype": "float16",
        },
    }
    path = ACT_DIR / "micro_kv_2k.pt"
    torch.save(fixture, path)
    return path


def generate_head_sensitivity_csv() -> tuple[Path, pd.DataFrame]:
    """Head sensitivity map with bimodal retrieval/streaming pattern."""
    torch.manual_seed(42)
    layers, heads = 12, 8
    rows: list[dict] = []
    profiles: list[HeadProfile] = []

    for layer in range(layers):
        for head in range(heads):
            # Bimodal assignment: ~20% retrieval, ~30% streaming, rest contextual
            u = (layer * heads + head) / (layers * heads)
            if u % 1.0 < 0.2:
                bias = "retrieval"
            elif u % 1.0 < 0.5:
                bias = "contextual"
            else:
                bias = "streaming"

            q_len, k_len = 16, 256
            p = torch.zeros(q_len, k_len)
            if bias == "retrieval":
                distant = torch.rand(q_len) * 0.4 + 0.3
            elif bias == "contextual":
                distant = torch.rand(q_len) * 0.15 + 0.08
            else:
                distant = torch.rand(q_len) * 0.03 + 0.005

            local_mass = 1.0 - distant
            for q in range(q_len):
                p[q, :128] = distant[q] / 128
                p[q, 128:] = local_mass[q] / 128
            p = p / p.sum(dim=-1, keepdim=True)

            hp = profile_heads(p.unsqueeze(0), layer_idx=layer, sink_tokens=4, local_window=128)[0]
            profiles.append(replace(hp, kv_head_idx=head))

    schedule = compile_head_bit_schedule(profiles, target_avg_bits=3.5)
    for s in schedule:
        rows.append(
            {
                "layer_idx": s["layer_idx"],
                "kv_head_idx": s["kv_head_idx"],
                "head_class": s["head_class"],
                "distant_attn_mass": round(s["distant_attn_mass"], 6),
                "attn_entropy": round(s["attn_entropy"], 6),
                "kl_sensitivity": round(s["kl_sensitivity"], 6),
                "assigned_k_bits": s["assigned_k_bits"],
                "assigned_v_bits": s["assigned_v_bits"],
            }
        )

    df = pd.DataFrame(rows)
    path = DATA_DIR / "p2_head_sensitivity_map.csv"
    df.to_csv(path, index=False)
    write_manifest(
        path,
        "Per-head distant attention mass, entropy, INT4 KL sensitivity and assigned bit schedule",
    )
    return path, df


def generate_cow_memory_csv() -> tuple[Path, pd.DataFrame]:
    """Multi-agent CoW prefix VRAM scaling with mixed-precision schedules."""
    rows = []
    num_layers, kv_heads, head_dim = 32, 8, 64
    shared_prefix, unique_tail = 1024, 256

    # bytes per token for KV tensors across all layers
    elem = kv_heads * head_dim * num_layers
    fp16_bpt = 2 * elem * 2  # K+V, 2 bytes each
    k8v4_bpt = elem * (8 / 8 + 4 / 8)  # K=8bit, V=4bit
    adaptive_bpt = elem * 2 * (3.5 / 8)  # avg 3.5 bits per element

    for agents in [1, 2, 4, 8, 16]:
        total_tokens_no_cow = agents * (shared_prefix + unique_tail)
        total_tokens_cow = shared_prefix + agents * unique_tail

        fp16_no_cow = total_tokens_no_cow * fp16_bpt / (1024**2)
        fp16_cow = total_tokens_cow * fp16_bpt / (1024**2)
        k8v4_cow = total_tokens_cow * k8v4_bpt / (1024**2)
        adaptive_cow = total_tokens_cow * adaptive_bpt / (1024**2)

        rows.append(
            {
                "num_concurrent_agents": agents,
                "shared_prefix_len": shared_prefix,
                "unique_tail_len": unique_tail,
                "fp16_no_cow_mb": round(fp16_no_cow, 3),
                "fp16_cow_mb": round(fp16_cow, 3),
                "k8v4_cow_mb": round(k8v4_cow, 3),
                "adaptive_3_5b_cow_mb": round(adaptive_cow, 3),
                "compression_factor": round(fp16_no_cow / adaptive_cow, 3),
            }
        )

    df = pd.DataFrame(rows)
    path = DATA_DIR / "p2_cow_prefix_memory.csv"
    df.to_csv(path, index=False)
    write_manifest(
        path, "Multi-agent KV VRAM scaling: FP16 vs CoW sharing vs K8V4 vs adaptive 3.5-bit"
    )
    return path, df


def generate_figure(
    sens_csv: Path, cow_csv: Path, sens_df: pd.DataFrame, cow_df: pd.DataFrame
) -> Path:
    """2-panel figure: head bimodality heatmap + CoW VRAM scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    dpi = 300

    # Panel A: layer x head distant attention mass heatmap
    ax = axes[0]
    pivot = sens_df.pivot(index="layer_idx", columns="kv_head_idx", values="distant_attn_mass")
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_title("A: Distant Attention Mass (Retrieval vs Streaming Heads)")
    ax.set_xlabel("KV Head Index")
    ax.set_ylabel("Layer Index")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Distant mass")
    ax.set_xticks(range(pivot.shape[1]))

    # Panel B: multi-agent VRAM scaling
    ax = axes[1]
    agents = cow_df["num_concurrent_agents"]
    ax.plot(agents, cow_df["fp16_no_cow_mb"], "o-", label="FP16 (no CoW)")
    ax.plot(agents, cow_df["fp16_cow_mb"], "s-", label="FP16 + CoW")
    ax.plot(agents, cow_df["k8v4_cow_mb"], "^-", label="K8V4 + CoW")
    ax.plot(agents, cow_df["adaptive_3_5b_cow_mb"], "d-", label="Adaptive 3.5-bit + CoW")
    ax.set_title("B: Multi-Agent KV VRAM Scaling")
    ax.set_xlabel("Concurrent Agents")
    ax.set_ylabel("KV VRAM (MiB)")
    ax.set_xticks(agents)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "fig2_head_bimodality_and_cow.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    from PIL import Image

    with Image.open(fig_path) as img:
        width, height = img.size

    meta = {
        "generating_script": "scripts/generate_phase2_artifacts.py",
        "source_csvs": [sens_csv.as_posix(), cow_csv.as_posix()],
        "csv_hashes": [sha256(sens_csv), sha256(cow_csv)],
        "width_px": width,
        "height_px": height,
        "dpi": dpi,
        "panels": [
            "A: Layer x KV-head distant attention mass heatmap (retrieval vs streaming heads)",
            "B: Multi-agent KV VRAM scaling curve comparing FP16 vs CoW vs quantized schedules",
        ],
        "activation_fixture": ACT_DIR.joinpath("micro_kv_2k.pt").as_posix(),
    }
    fig_path.with_suffix(fig_path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return fig_path


def main() -> int:
    print("Generating Phase 2 artifacts...")
    fixture = generate_fixture()
    print(f"  [OK] {fixture}")
    sens_csv, sens_df = generate_head_sensitivity_csv()
    print(f"  [OK] {sens_csv}")
    cow_csv, cow_df = generate_cow_memory_csv()
    print(f"  [OK] {cow_csv}")
    fig = generate_figure(sens_csv, cow_csv, sens_df, cow_df)
    print(f"  [OK] {fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
