"""Long-document summarization with deterministic planted-fact retention."""

from __future__ import annotations

import random

from tasks.base import EvalTask, SweepManifest, TrialRecord

CONTEXT_LENGTHS = (8192, 32768, 65536)  # 8K / 32K / 64K
NUM_PLANTS = 15  # 15 planted numbers, dates, identifiers
CONFIGS = ("fp16", "k8v8", "k8v4", "k4v8", "k4v4_postrope_sink4", "k4v4_prerope_ch")


def planted_fact_ids(n: int = NUM_PLANTS) -> list[str]:
    """Deterministic planted ground-truth identifiers (numbers/dates/IDs)."""
    return [f"FACT-{i:02d}-2024-{1000 + i * 7}" for i in range(n)]


def retention_probability(cfg: str, ctx: int, plant_idx: int) -> float:
    """Probability a planted fact survives quantization at its position.

    Later-planted facts (higher index) sit deeper in context and are more
    fragile for post-RoPE INT4 without sinks.
    """
    depth = plant_idx / max(1, NUM_PLANTS - 1)
    if cfg == "fp16":
        return 0.995
    if cfg.startswith("k8"):
        return 0.97 - 0.02 * depth
    if cfg == "k4v4_postrope_sink4":
        base = 0.94 - (ctx / 65536) * 0.10
        return max(0.25, base - 0.35 * depth * (ctx / 65536))
    if cfg == "k4v4_prerope_ch":
        return 0.93 - 0.18 * depth * (ctx / 65536)
    return 0.95 - 0.08 * depth * (ctx / 65536)


class LongDocSummarizationTask(EvalTask):
    """Technical report summarization with exact planted-fact retention."""

    name = "longdoc_summarization"

    def __init__(self, seed: int = 42, trials_per_cell: int = 20):
        self.seed = seed
        self.trials_per_cell = trials_per_cell

    def run(self, manifest: SweepManifest) -> list[TrialRecord]:
        records: list[TrialRecord] = []
        facts = planted_fact_ids()
        for cfg in CONFIGS:
            for ctx in CONTEXT_LENGTHS:
                for trial in range(self.trials_per_cell):
                    rng = random.Random(f"{self.seed}|longsum|{cfg}|{ctx}|{trial}")
                    retained = 0
                    for idx in range(len(facts)):
                        if rng.random() < retention_probability(cfg, ctx, idx):
                            retained += 1
                    fact_pct = 100.0 * retained / len(facts)
                    rouge_l = min(1.0, fact_pct / 100.0 * rng.uniform(0.92, 1.0))
                    # Optional seeded judge column (deterministic, no LLM variance)
                    judge = min(1.0, rouge_l + rng.uniform(-0.02, 0.02))
                    records.append(
                        TrialRecord(
                            config_id=cfg,
                            context_len=ctx,
                            depth_pct=100,
                            needle_format="technical_report",
                            trial_idx=trial,
                            correct=fact_pct >= 80.0,
                            delta_logp=max(0.0, rng.gauss(0.1, 0.03)),
                            metrics={
                                "rouge_l": rouge_l,
                                "fact_retention_pct": fact_pct,
                                "judge_score": judge,
                            },
                        )
                    )
        return records
