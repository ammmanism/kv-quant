"""Tests for the CSV-backed Pareto advisor."""

import pytest
from kv_quant.advisor import (
    MODEL_PROFILES,
    AdvisorRecommendation,
    fp16_baseline,
    recommend,
    render_markdown,
)


def test_fp16_baseline_like_for_like():
    acc, ctx = fp16_baseline("multihop", 128000)
    assert ctx == 131072  # nearest committed context
    assert 0.9 < acc < 1.0


def test_recommend_mandated_query():
    recs = recommend(context_len=128000, task="multihop", max_drop_pct=1.0, vram_budget_gb=24)
    labels = [r.label for r in recs]
    # Feasible: FP16, K8V8, K8V4. INT4-K configs fail the 1% drop gate.
    assert "FP16" in labels
    assert "K8V4 (asymmetric V)" in labels
    assert all("K4V4" not in lab for lab in labels)
    assert all("K4V8" not in lab for lab in labels)
    # Sorted by highest compression first
    assert recs[0].compression_ratio >= recs[-1].compression_ratio
    assert all(isinstance(r, AdvisorRecommendation) for r in recs)


def test_recommend_respects_vram_budget():
    # 17 GiB total - 15.2 GiB bf16 weights leaves ~3.0 GiB for KV at 128K:
    # FP16 (7.0 GiB) and K8V8 (3.5 GiB) don't fit; INT4-class configs do.
    recs = recommend(context_len=128000, task="multihop", max_drop_pct=100.0, vram_budget_gb=17)
    by_label = {r.label: r for r in recs}
    assert "FP16" not in by_label
    assert "K8V8" not in by_label
    assert "K4V4 post-rope + 4 FP16 sinks" in by_label
    assert by_label["K4V4 post-rope + 4 FP16 sinks"].kv_bytes < 2.0 * (1 << 30)


def test_recommend_max_drop_gate():
    recs = recommend(context_len=65536, task="needle", max_drop_pct=0.1, vram_budget_gb=64)
    labels = [r.label for r in recs]
    assert "K4V4 post-rope + 4 FP16 sinks" not in labels
    assert all(r.drop_pct <= 0.1 for r in recs)


def test_render_markdown_contains_decision_columns():
    recs = recommend(context_len=128000, task="multihop", max_drop_pct=1.0, vram_budget_gb=24)
    fp16_acc, ctx = fp16_baseline("multihop", 128000)
    md = render_markdown(recs, fp16_acc, ctx)
    assert "| Config | acc | drop vs FP16 | KV VRAM | compression | ITL speedup |" in md
    assert "QuantConfig" in md
    assert "k_bits=8" in md  # highest-compression feasible config is K8V4
    assert "bench/data/p6_v1_frontier_sweep.csv" in md


def test_recommend_empty_returns_nothing():
    recs = recommend(context_len=4096, task="needle", max_drop_pct=0.0, vram_budget_gb=1.0)
    # 1 GiB budget minus weights is negative -> nothing feasible
    assert recs == []


def test_model_profile_geometry():
    profile = MODEL_PROFILES["Qwen/Qwen2.5-7B-Instruct"]
    assert profile["num_kv_heads"] == 4 and profile["num_q_heads"] == 28
    assert profile["head_dim"] == 128


@pytest.mark.parametrize("task", ["needle", "multihop"])
def test_recommend_deterministic(task: str):
    a = recommend(128000, task, 1.0, 24)
    b = recommend(128000, task, 1.0, 24)
    assert [(r.label, r.acc_mean, r.kv_bytes) for r in a] == [
        (r.label, r.acc_mean, r.kv_bytes) for r in b
    ]
