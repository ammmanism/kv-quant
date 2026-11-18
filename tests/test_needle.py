"""Tests for the needle probe, Wilson CIs, replay adapter, and cliff detector."""

import torch
from tasks.base import RecordedReplayAdapter, SweepManifest, TrialRecord, wilson_ci
from tasks.needle import (
    CONFIGS,
    CONTEXT_LENGTHS,
    DEPTH_PCTS,
    NEEDLE_FORMATS,
    PositionResolvedNeedleTask,
    detect_cliff_coordinates,
)


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(50, 100)
    assert 0 <= lo <= 0.5 <= hi <= 1
    lo0, hi0 = wilson_ci(0, 100)
    assert lo0 == 0.0
    assert hi0 > 0
    lon, hin = wilson_ci(100, 100)
    assert lon < 1
    assert hin > 0.99


def test_wilson_ci_narrows_with_n():
    lo10, hi10 = wilson_ci(80, 100)
    lo1000, hi1000 = wilson_ci(800, 1000)
    assert (hi1000 - lo1000) < (hi10 - lo10)


def test_wilson_ci_empty():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_manifest_mandatory_fields():
    m = SweepManifest.capture(seed=123)
    d = m.to_json()
    for field in (
        "git_sha",
        "timestamp_utc",
        "model_id",
        "param_count_b",
        "num_q_heads",
        "num_kv_heads",
        "rotary_dim",
        "gpu_name",
        "seed",
        "sha256_csv",
    ):
        assert field in d
    assert d["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert d["param_count_b"] == 7.61
    assert d["num_q_heads"] == 28
    assert d["num_kv_heads"] == 4
    assert d["rotary_dim"] == 128


def test_replay_deterministic():
    a = RecordedReplayAdapter(seed=7)
    b = RecordedReplayAdapter(seed=7)
    ra = a.replay("k8v4", 32768, 80, "random_hex_8")
    rb = b.replay("k8v4", 32768, 80, "random_hex_8")
    assert [r.correct for r in ra] == [r.correct for r in rb]
    assert [round(r.delta_logp, 6) for r in ra] == [round(r.delta_logp, 6) for r in rb]


def test_needle_task_grid():
    task = PositionResolvedNeedleTask(RecordedReplayAdapter(trials_per_cell=3))
    m = SweepManifest.capture(seed=1)
    records = task.run(m)
    expected = len(CONFIGS) * len(CONTEXT_LENGTHS) * len(DEPTH_PCTS) * len(NEEDLE_FORMATS) * 3
    assert len(records) == expected
    assert NEEDLE_FORMATS[0] == "random_hex_8"  # zero-prior passkey present


def test_needle_sweep_dataframe(tmp_path):
    task = PositionResolvedNeedleTask(RecordedReplayAdapter(trials_per_cell=4))
    m = SweepManifest.capture(seed=42)
    csv = tmp_path / "p4_needle_depth_matrix.csv"
    df = task.sweep(m, csv)
    assert {
        "config_id",
        "context_len",
        "depth_pct",
        "acc_mean",
        "ci95_low",
        "ci95_high",
        "delta_logp",
    }.issubset(df.columns)
    assert csv.exists()
    assert csv.with_suffix(csv.suffix + ".manifest.json").exists()
    assert len(df) == (len(CONFIGS) * len(CONTEXT_LENGTHS) * len(DEPTH_PCTS) * len(NEEDLE_FORMATS))


def test_cliff_detection():
    task = PositionResolvedNeedleTask(RecordedReplayAdapter(trials_per_cell=20))
    m = SweepManifest.capture(seed=42)
    df = task.to_dataframe(task.run(m))
    cliffs = detect_cliff_coordinates(df)
    assert cliffs
    # Cliffs should occur in late-context, long-context cells
    late_long = [c for c in cliffs if c.context_len >= 32768 and c.depth_pct >= 70]
    assert late_long
    assert any(c.config_id == "k4v4_postrope_sink4" for c in late_long)
    # fp16 should never be flagged
    assert all(c.config_id != "fp16" for c in cliffs)


def test_no_gpu_required():
    """The replay adapter runs without CUDA."""
    assert not torch.cuda.is_available()
    a = RecordedReplayAdapter(seed=1)
    records = a.replay("fp16", 4096, 0, "uuid_passkey")
    assert isinstance(records[0], TrialRecord)
    assert 0 <= records[0].delta_logp < 10
