"""Tests for the ecological task suite (multihop, longsum, longcode)."""

import ast

from tasks.base import SweepManifest
from tasks.longcode import (
    COMMITTED_ASSERTIONS,
    EXPECTED_SOURCE,
    CrossFileCodeCompletionTask,
    canonical_unparse,
    canonicalize_ast,
    run_committed_assertions,
)
from tasks.longsum import (
    NUM_PLANTS,
    LongDocSummarizationTask,
    planted_fact_ids,
    retention_probability,
)
from tasks.multihop import MultiHopBridgeTask


def test_ast_canonicalization_whitespace_invariant():
    a = "def f(x):\n    return x + 1\n"
    b = "def f( x ):\n  # comment\n  return x+1"
    assert canonicalize_ast(a) == canonicalize_ast(b)
    assert canonicalize_ast("def broken(") == "<SYNTAX_ERROR>"


def test_ast_unparse_and_assertions():
    readable = canonical_unparse(EXPECTED_SOURCE)
    assert "def compute_total" in readable
    passed, total = run_committed_assertions(readable)
    assert passed == total == len(COMMITTED_ASSERTIONS)
    # A broken completion fails at least one assertion
    bad = canonical_unparse("def compute_total(v): return sum(v)")
    bad_passed, _ = run_committed_assertions(bad)
    assert bad_passed < total


def test_longcode_task_metrics():
    task = CrossFileCodeCompletionTask(trials_per_cell=3)
    df = task.to_dataframe(task.run(SweepManifest.capture(seed=1)))
    assert {"ast_exact_match", "pass_at_1"}.issubset(df.columns)
    assert df["ast_exact_match"].between(0, 1).all()
    assert df["pass_at_1"].between(0, 1).all()


def test_hop1_hop2_isolation():
    task = MultiHopBridgeTask(trials_per_cell=20)
    df = task.to_dataframe(task.run(SweepManifest.capture(seed=42)))
    assert {"em", "f1", "hop1_recall", "hop2_recall"}.issubset(df.columns)
    sub = df[(df["config_id"] == "k4v4_postrope_sink4") & (df["context_len"] >= 32768)]
    # Early recall stays high while late bridge recall collapses
    assert sub["hop1_recall"].mean() > sub["hop2_recall"].mean() + 0.2
    # fp16 keeps both high
    fp = df[df["config_id"] == "fp16"]
    assert fp["hop2_recall"].mean() > 0.9


def test_multihop_grid():
    task = MultiHopBridgeTask(trials_per_cell=2)
    df = task.to_dataframe(task.run(SweepManifest.capture(seed=1)))
    assert set(df["needle_format"]) == {"2_hop", "3_hop"}
    assert set(df["context_len"]) == {4096, 16384, 32768, 65536, 131072}


def test_fact_retention_determinism():
    ids = planted_fact_ids()
    assert len(ids) == NUM_PLANTS == 15
    assert ids == planted_fact_ids()  # deterministic
    # Deeper facts are less likely to survive post-RoPE INT4 at 64K
    assert retention_probability("k4v4_postrope_sink4", 65536, 14) < retention_probability(
        "k4v4_postrope_sink4", 65536, 0
    )


def test_longsum_task_metrics():
    task = LongDocSummarizationTask(trials_per_cell=5)
    df = task.to_dataframe(task.run(SweepManifest.capture(seed=42)))
    assert {"rouge_l", "fact_retention_pct", "judge_score"}.issubset(df.columns)
    assert df["fact_retention_pct"].between(0, 100).all()
    assert df["rouge_l"].between(0, 1).all()
    fp = df[df["config_id"] == "fp16"]
    assert fp["fact_retention_pct"].mean() > 90


def test_all_tasks_sweep_no_gpu(tmp_path):
    """All three tasks run through the shared spine without a GPU."""
    manifest = SweepManifest.capture(seed=7)
    for task, fname in (
        (MultiHopBridgeTask(trials_per_cell=2), "multihop.csv"),
        (LongDocSummarizationTask(trials_per_cell=2), "longsum.csv"),
        (CrossFileCodeCompletionTask(trials_per_cell=2), "longcode.csv"),
    ):
        df = task.sweep(manifest, tmp_path / fname)
        assert len(df) > 0
        assert (tmp_path / fname).exists()


def test_expected_source_parses():
    tree = ast.parse(EXPECTED_SOURCE)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "compute_total" for n in tree.body)
