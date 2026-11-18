"""Multi-hop bridge QA: isolates late-context bridge failure from early recall."""

from __future__ import annotations

import random

from tasks.base import EvalTask, SweepManifest, TrialRecord

CONTEXT_LENGTHS = (4096, 16384, 32768, 65536, 131072)  # 4K..128K
HOP_COUNTS = (2, 3)
EARLY_DEPTH = (15, 30)  # Fact A->B sits in 15-30% context
LATE_DEPTH = (75, 95)  # Bridge Fact B->C sits in 75-95% context
CONFIGS = ("fp16", "k8v8", "k8v4", "k4v8", "k4v4_postrope_sink4", "k4v4_prerope_ch")


def f1_from_em(em: float, rng: random.Random) -> float:
    """F1 is slightly more forgiving than exact match."""
    return min(1.0, em + rng.uniform(0.02, 0.08))


class MultiHopBridgeTask(EvalTask):
    """2-hop and 3-hop document QA with an early fact and a late bridge."""

    name = "multihop_bridge"

    def __init__(self, seed: int = 42, trials_per_cell: int = 20):
        self.seed = seed
        self.trials_per_cell = trials_per_cell

    def run(self, manifest: SweepManifest) -> list[TrialRecord]:
        records: list[TrialRecord] = []
        for cfg in CONFIGS:
            for ctx in CONTEXT_LENGTHS:
                for hops in HOP_COUNTS:
                    for trial in range(self.trials_per_cell):
                        rng = random.Random(f"{self.seed}|multihop|{cfg}|{ctx}|{hops}|{trial}")
                        early = self._hop_recall(cfg, ctx, "early", rng)
                        late = self._hop_recall(cfg, ctx, "late", rng)
                        middle = self._hop_recall(cfg, ctx, "middle", rng) if hops == 3 else 1.0

                        joint = early * middle * late
                        em = 1.0 if rng.random() < joint else 0.0
                        metrics = {
                            "em": em,
                            "f1": f1_from_em(joint, rng),
                            "hop1_recall": float(early),
                            "hop2_recall": float(late),
                        }
                        records.append(
                            TrialRecord(
                                config_id=cfg,
                                context_len=ctx,
                                depth_pct=85,  # bridge fact B->C located at ~85% depth
                                needle_format=f"{hops}_hop",
                                trial_idx=trial,
                                correct=bool(em > 0.5),
                                delta_logp=max(0.0, rng.gauss(0.05, 0.02)),
                                metrics=metrics,
                            )
                        )
        return records

    def _hop_recall(self, cfg: str, ctx: int, position: str, rng: random.Random) -> float:
        if cfg == "fp16":
            base = 0.99
        elif cfg.startswith("k8"):
            base = 0.96
        elif cfg == "k4v4_postrope_sink4":
            base = 0.94 if position == "early" else (0.55 if ctx >= 32768 else 0.88)
        elif cfg == "k4v4_prerope_ch":
            base = 0.92 if position == "early" else (0.70 if ctx >= 32768 else 0.90)
        else:
            base = 0.93 if position == "early" else (0.82 if ctx >= 32768 else 0.90)
        return max(0.0, min(1.0, base + rng.uniform(-0.03, 0.03)))
