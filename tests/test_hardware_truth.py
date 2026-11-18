"""Tests for the hardware-truth phase: ncu counters, TTFT/ITL, VRAM cross-val."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch
from bench.bandwidth import modeled_ncu_counters, ncu_bandwidth_rows
from bench.latency import latency_rows
from kv_quant.quantizer import dequantize_group, quantize_group
from kv_quant.vram import cross_validate_vram, kv_cache_calculator_bytes

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------- quantizer fp16 bug
def test_quantize_group_bits16_strided_channel():
    """Regression: bits=16 on a strided channel slice must not crash on view."""
    block = torch.randn(64, 4, 128, dtype=torch.float16)
    channel = block[:, 1, 7]  # stride 512 view
    packed, qp = quantize_group(channel, 16, "sym")
    restored = dequantize_group(packed, qp, 16, "sym")
    assert torch.allclose(restored, channel, atol=0.0)


# ------------------------------------------------------------- ncu counters
def test_ncu_counters_fused_beats_naive():
    row_n = modeled_ncu_counters(65536, 128, 32, 4, 4, 64, "naive")
    row_f = modeled_ncu_counters(65536, 128, 32, 4, 4, 64, "fused")
    assert row_f["dram_total_bytes"] < row_n["dram_total_bytes"]
    ratio = row_n["dram_total_bytes"] / row_f["dram_total_bytes"]
    assert 8.0 < ratio < 10.5, f"k4v4 reduction {ratio} outside 8-10.5x band"


def test_ncu_counters_fp16_config_three_round_trips():
    fp16_n = next(
        r for r in ncu_bandwidth_rows() if r["config"] == "k16v16" and r["kernel"] == "naive"
    )  # type: ignore[arg-type]
    fp16_f = next(
        r for r in ncu_bandwidth_rows() if r["config"] == "k16v16" and r["kernel"] == "fused"
    )  # type: ignore[arg-type]
    assert float(fp16_n["dram_total_bytes"]) == pytest.approx(
        3 * float(fp16_f["dram_total_bytes"]), rel=0.02
    )


def test_ncu_counters_l2_residency_discount():
    """Scale vectors must be discounted by the GQA x split-K reuse factor."""
    row = modeled_ncu_counters(65536, 128, 32, 4, 4, 64, "fused")
    payload = 65536 * 32 * 128 * 1.0  # k4+v4 = 1 byte/elem
    meta_dram = row["dram_read_bytes"] - payload
    assert meta_dram < 65536 * 32 * 128 * 8 / 64 / 20  # >20x discount on meta


# --------------------------------------------------------------- latency
def test_latency_percentile_ordering_and_speedup():
    for row in latency_rows():
        assert row["itl_naive_p99_ms"] >= row["itl_naive_p90_ms"] >= row["itl_naive_p50_ms"]  # type: ignore[operator]
        assert row["itl_fused_p99_ms"] >= row["itl_fused_p90_ms"] >= row["itl_fused_p50_ms"]  # type: ignore[operator]
        assert row["itl_speedup_p50"] > 1.0  # type: ignore[operator]


def test_latency_crossover_amortizes_fast():
    """TTFT overhead must amortize within <=5 decode steps at every context."""
    for row in latency_rows():
        assert row["amortize_decode_steps"] <= 5.0  # type: ignore[operator]


# -------------------------------------------------------- VRAM cross-val
def test_calculator_formula_matches_hand_computation():
    # 32 layers x 4096 seq: sealed payload (k4v4 = 1 byte/elem) + sink/ring fp16
    predicted = kv_cache_calculator_bytes(32, 4, 128, 4096, 4, 4, sink_tokens=4, group_size=64)
    channels = 4 * 128
    sealed = 4096 - 4
    payload = 32 * (sealed // 64) * 64 * channels  # 1 byte/elem for k4v4
    ring = sealed % 64
    fp16_extra = 32 * (4 + ring) * channels * 4
    assert predicted == payload + fp16_extra


def test_cross_validate_vram_small():
    configs = [(4, 4, 512), (8, 8, 512), (4, 8, 1024), (8, 4, 1024), (16, 16, 1024), (6, 6, 512)]
    df = cross_validate_vram(configs, num_layers=4, num_kv_heads=2, head_dim=32)
    assert len(df) == 6
    assert df["r_squared"].iloc[0] > 0.999
    assert df["mape_pct"].iloc[0] < 1.0
    assert (df["measured_peak_bytes"] <= df["kv_calc_fp16_equiv_bytes"]).all()


# ------------------------------------------------------- committed CSVs
@pytest.mark.parametrize(
    ("name", "rows"),
    [
        ("p7_ncu_bandwidth.csv", 40),
        ("p7_latency_ttft_itl.csv", 20),
        ("p7_vram_crossval.csv", 20),
    ],
)
def test_committed_hardware_csvs(name: str, rows: int):
    df = pd.read_csv(ROOT / "bench" / "data" / name)
    assert len(df) == rows
    mode_col = "mode" if "mode" in df.columns else "measure_method"
    assert df[mode_col].nunique() == 1  # honest single execution mode
    manifest = json.loads((ROOT / "bench" / "data" / f"{name}.manifest.json").read_text())
    assert manifest["num_rows"] == rows
    assert len(manifest["sha256"]) == 64


def test_committed_crossval_regression_from_csv():
    """Recompute the R^2/MAPE fit from the committed CSV itself."""
    df = pd.read_csv(ROOT / "bench" / "data" / "p7_vram_crossval.csv")
    x = df["calculator_predicted_bytes"].to_numpy(dtype=float)
    y = df["measured_peak_bytes"].to_numpy(dtype=float)
    slope, intercept = np_polyfit(x, y)
    resid = y - (slope * x + intercept)
    r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
    mape = float((abs(y - x) / x).mean() * 100)
    assert r2 > 0.999
    assert mape < 1.0
    assert math.isclose(r2, float(df["r_squared"].iloc[0]), abs_tol=1e-4)


def np_polyfit(x, y):
    import numpy as np

    return np.polyfit(x, y, 1)
