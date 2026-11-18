"""Tests for the declarative sweep orchestrator and golden baseline."""

import json
from pathlib import Path

import pandas as pd
import yaml
from kv_quant.sweep import (
    SweepCell,
    SweepOrchestrator,
    expand_cells,
    load_sweep_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_load_and_expand_smoke():
    cfg = load_sweep_config(ROOT / "sweeps" / "smoke-ci.yaml")
    assert cfg["name"] == "smoke-ci"
    cells = expand_cells(cfg)
    assert len(cells) == 96  # budget-capped
    assert all(isinstance(c, SweepCell) for c in cells)


def test_expand_frontier_full_grid():
    cfg = load_sweep_config(ROOT / "sweeps" / "v1-frontier.yaml")
    cells = expand_cells(cfg)
    assert len(cells) > 2000  # full grid, no budget cap
    assert {c.task for c in cells} == {"needle", "multihop", "longsum", "longcode"}
    assert {c.k_bits for c in cells} >= {16, 8, 6, 4}


def test_orchestrator_run_and_manifest(tmp_path):
    output = tmp_path / "sweep.csv"
    orch = SweepOrchestrator(ROOT / "sweeps" / "smoke-ci.yaml", output)
    df = orch.run()
    assert len(df) == 96
    assert "model_id" in df.columns
    assert set(df["status"]).issubset({"OK", "OOM"})
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert len(manifest["sha256"]) == 64


def test_resume_is_idempotent(tmp_path):
    output = tmp_path / "sweep.csv"
    first = SweepOrchestrator(ROOT / "sweeps" / "smoke-ci.yaml", output).run()
    second = SweepOrchestrator(ROOT / "sweeps" / "smoke-ci.yaml", output, resume=True).run()
    assert len(first) == len(second)
    assert first["acc_mean"].tolist() == second["acc_mean"].tolist()


def test_committed_frontier_csv():
    path = ROOT / "bench" / "data" / "p6_v1_frontier_sweep.csv"
    df = pd.read_csv(path)
    assert len(df) == 2596
    sub = df[(df["task"] == "needle") & (df["context_len"].isin([65536, 131072]))]
    grid = sub.groupby(["k_bits", "v_bits"])["acc_mean"].mean().unstack()
    fp = grid.loc[16, 16].mean()
    # Keys routing-critical: K4V8 drops >18%
    assert (fp - grid.loc[4, 8].mean()) / fp > 0.18
    # Values compressible: K8V4 within 0.4% of FP16
    assert (fp - grid.loc[8, 4].mean()) / fp < 0.004


def test_golden_baseline_present():
    path = ROOT / "bench" / "baselines" / "quality-golden.csv"
    df = pd.read_csv(path)
    assert {"metric", "value"}.issubset(df.columns)
    assert set(df["metric"]).issuperset({"accuracy", "vram_mib"})


def test_yaml_configs_valid():
    for name in ("smoke-ci.yaml", "v1-frontier.yaml"):
        cfg = yaml.safe_load((ROOT / "sweeps" / name).read_text())
        assert cfg["model"]["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
        assert "fast_tasks" in cfg and "heavy_tasks" in cfg
