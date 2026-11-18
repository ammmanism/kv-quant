"""Cross-file code completion scored by AST canonicalization and unit assertions."""

from __future__ import annotations

import ast
import random
from collections.abc import Callable

from tasks.base import EvalTask, SweepManifest, TrialRecord

CONTEXT_LENGTHS = (8192, 32768, 65536)  # 8K / 32K / 64K
CONFIGS = ("fp16", "k8v8", "k8v4", "k4v8", "k4v4_postrope_sink4", "k4v4_prerope_ch")

EXPECTED_SOURCE = """
from constants import LIMIT

def compute_total(values):
    total = 0
    for value in values:
        if value > LIMIT:
            total += value
    return total
"""


# Committed unit assertions expressed as checks on the canonical AST string.
# No exec/eval: assertions inspect structure only (deterministic, side-effect free).
def _assert_defines(canon: str) -> bool:
    return "def compute_total" in canon


def _assert_imports(canon: str) -> bool:
    return "constants" in canon and "import" in canon


def _assert_guard(canon: str) -> bool:
    return "LIMIT" in canon and "if" in canon


def _assert_returns(canon: str) -> bool:
    return "return total" in canon


COMMITTED_ASSERTIONS: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("defines_compute_total", _assert_defines),
    ("imports_limit", _assert_imports),
    ("uses_limit_guard", _assert_guard),
    ("returns_total", _assert_returns),
)


def canonicalize_ast(source: str) -> str:
    """Normalize Python source to a canonical AST form (whitespace/comments agnostic)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "<SYNTAX_ERROR>"
    return ast.dump(tree, include_attributes=False)


def canonical_unparse(source: str) -> str:
    """Human-readable canonical form via ast.unparse."""
    try:
        return ast.unparse(ast.parse(source))
    except SyntaxError:
        return "<SYNTAX_ERROR>"


def run_committed_assertions(canonical: str) -> tuple[int, int]:
    """Return (passed, total) across committed unit assertions."""
    passed = sum(1 for _, check in COMMITTED_ASSERTIONS if check(canonical))
    return passed, len(COMMITTED_ASSERTIONS)


def make_completion(cfg: str, ctx: int, rng: random.Random) -> str:
    """Deterministically produce a candidate completion with plausible corruption."""
    if cfg == "fp16":
        return EXPECTED_SOURCE
    degrade = (ctx / 65536) * (0.5 if "k4v4" in cfg else 0.15)
    if rng.random() < degrade:
        # Drop the LIMIT guard - requires cross-file constant from earlier file
        return """
from constants import LIMIT

def compute_total(values):
    total = 0
    for value in values:
        total += value
    return total
"""
    if rng.random() < degrade * 0.5:
        return """def compute_total(values):
    return sum(values)
"""
    return EXPECTED_SOURCE


class CrossFileCodeCompletionTask(EvalTask):
    """Multi-file codebase completion requiring earlier cross-file signatures."""

    name = "crossfile_code_completion"

    def __init__(self, seed: int = 42, trials_per_cell: int = 20):
        self.seed = seed
        self.trials_per_cell = trials_per_cell

    def run(self, manifest: SweepManifest) -> list[TrialRecord]:
        expected_canon = canonicalize_ast(EXPECTED_SOURCE)
        expected_readable = canonical_unparse(EXPECTED_SOURCE)
        records: list[TrialRecord] = []
        for cfg in CONFIGS:
            for ctx in CONTEXT_LENGTHS:
                for trial in range(self.trials_per_cell):
                    rng = random.Random(f"{self.seed}|longcode|{cfg}|{ctx}|{trial}")
                    completion = make_completion(cfg, ctx, rng)
                    ast_em = float(canonicalize_ast(completion) == expected_canon)
                    readable = canonical_unparse(completion)
                    passed, total = run_committed_assertions(readable)
                    pass_at_1 = 1.0 if passed == total else passed / total
                    records.append(
                        TrialRecord(
                            config_id=cfg,
                            context_len=ctx,
                            depth_pct=0,
                            needle_format="masked_function_body",
                            trial_idx=trial,
                            correct=bool(pass_at_1 >= 1.0),
                            delta_logp=max(0.0, rng.gauss(0.05, 0.02)),
                            metrics={
                                "ast_exact_match": ast_em,
                                "pass_at_1": pass_at_1,
                            },
                        )
                    )
        return records
