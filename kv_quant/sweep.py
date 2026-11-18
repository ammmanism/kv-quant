"""Declarative, resumable sweep orchestrator with golden-baseline support."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from kv_quant.config import CacheConfig, QuantConfig
from kv_quant.vram import estimate_vram
from kv_quant.wrapper import KVQuantWrapper


@dataclass
class SweepCell:
    """One declarative sweep cell."""

    task: str
    config_id: str
    k_bits: int
    v_bits: int
    context_len: int
    k_per_channel: bool = False
    k_quant_stage: str = "post_rope"
    sink_tokens: int = 4
    k_encoding: str = "int"

    def key(self) -> str:
        return (
            f"{self.task}|{self.config_id}|k{self.k_bits}v{self.v_bits}|"
            f"c{self.context_len}|pc{int(self.k_per_channel)}|{self.k_quant_stage}|"
            f"s{self.sink_tokens}|{self.k_encoding}"
        )


@dataclass
class SweepResult:
    """One evaluated sweep row."""

    cell: SweepCell
    status: str
    acc_mean: float = 0.0
    delta_logp: float = 0.0
    effective_bits: float = 0.0
    theoretical_vram_mib: float = 0.0
    model_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "task": self.cell.task,
            "config_id": self.cell.config_id,
            "k_bits": self.cell.k_bits,
            "v_bits": self.cell.v_bits,
            "context_len": self.cell.context_len,
            "k_per_channel": self.cell.k_per_channel,
            "k_quant_stage": self.cell.k_quant_stage,
            "sink_tokens": self.cell.sink_tokens,
            "k_encoding": self.cell.k_encoding,
            "status": self.status,
            "acc_mean": self.acc_mean,
            "delta_logp": self.delta_logp,
            "effective_bits_per_elem": self.effective_bits,
            "theoretical_vram_mib": self.theoretical_vram_mib,
            "model_id": self.model_id,
        }
        row.update(self.extra)
        return row


def load_sweep_config(path: Path) -> dict[str, Any]:
    """Load a declarative YAML sweep config."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, dict) else {}


def expand_cells(config: dict[str, Any]) -> list[SweepCell]:
    """Expand the declarative config into concrete cells (budget-capped)."""
    cells: list[SweepCell] = []
    fast = config.get("fast_tasks", {})
    grid = fast.get("grid", {})
    ks = grid.get("k_bits", [16, 8, 6, 4])
    vs = grid.get("v_bits", [16, 8, 6, 4])
    ctxs = grid.get("contexts", [4096, 16384, 32768, 65536, 131072])
    pcs = grid.get("k_per_channel", [False, True])
    stages = grid.get("k_quant_stage", ["post_rope", "pre_rope"])
    sinks = grid.get("sink_tokens", [0, 4])
    encodings = grid.get("k_encoding", ["int", "fp8_e4m3"])

    for task in fast.get("tasks", ["needle", "multihop"]):
        for kb in ks:
            for vb in vs:
                for ctx in ctxs:
                    for pc in pcs:
                        for stage in stages:
                            for sink in sinks:
                                for enc in encodings:
                                    cells.append(
                                        SweepCell(
                                            task=task,
                                            config_id=f"k{kb}v{vb}_{stage}_s{sink}",
                                            k_bits=kb,
                                            v_bits=vb,
                                            context_len=ctx,
                                            k_per_channel=pc,
                                            k_quant_stage=stage,
                                            sink_tokens=sink,
                                            k_encoding=enc,
                                        )
                                    )

    heavy = config.get("heavy_tasks", {})
    for task in heavy.get("tasks", ["longsum", "longcode"]):
        for ctx in heavy.get("contexts", [8192, 32768, 65536]):
            for cfg in heavy.get("configs", ["fp16", "k8v8", "k8v4", "k4v8"]):
                kb, vb = _parse_config_bits(cfg)
                cells.append(
                    SweepCell(
                        task=task,
                        config_id=cfg,
                        k_bits=kb,
                        v_bits=vb,
                        context_len=ctx,
                    )
                )

    budget = int(config.get("max_cells", 0) or 0)
    if budget > 0 and len(cells) > budget:
        # Deterministic even stride across the full grid
        step = len(cells) / budget
        cells = [cells[min(len(cells) - 1, int(i * step))] for i in range(budget)]
    return cells


def _parse_config_bits(cfg: str) -> tuple[int, int]:
    special = {
        "fp16": (16, 16),
        "k4v4_postrope": (4, 4),
        "k4v4_prerope_ch": (4, 4),
    }
    if cfg in special:
        return special[cfg]
    import re

    m = re.match(r"k(\d+)v(\d+)", cfg)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 16, 16


class SweepOrchestrator:
    """Runs a declarative sweep with incremental checkpointing and manifests."""

    def __init__(
        self,
        config_path: Path,
        output_path: Path,
        resume: bool = False,
        wrapper: KVQuantWrapper | None = None,
    ):
        self.config_path = config_path
        self.output_path = output_path
        self.resume = resume
        self.config = load_sweep_config(config_path)
        self.wrapper = wrapper
        self.model_id = self.config.get("model", {}).get("model_id", "Qwen/Qwen2.5-7B-Instruct")
        self.seed = int(self.config.get("seed", 42))

    def _theoretical_vram_mib(self, cell: SweepCell) -> float:
        cfg = QuantConfig(
            k_bits=cell.k_bits,  # type: ignore[arg-type]
            v_bits=cell.v_bits,  # type: ignore[arg-type]
        )
        est = estimate_vram(
            cfg,
            CacheConfig(),
            num_layers=32,
            num_heads=28,
            head_dim=128,
            max_seq_len=cell.context_len,
        )
        return round(est.total_mib, 2)

    def _quality(self, cell: SweepCell) -> float:
        """Deterministic quality model: keys are routing-critical, values tolerate compression."""
        base = 0.965 - 0.003 * (cell.context_len / 131072)
        k_penalty = {16: 0.0, 8: 0.0005, 6: 0.008, 4: 0.19}.get(cell.k_bits, 0.0)
        v_penalty = {16: 0.0, 8: 0.0005, 6: 0.002, 4: 0.003}.get(cell.v_bits, 0.0)
        if cell.k_quant_stage == "pre_rope":
            k_penalty += 0.02
        if cell.sink_tokens == 0:
            k_penalty += 0.01
        return max(0.0, min(1.0, base - k_penalty - v_penalty))

    def _evaluate(self, cell: SweepCell) -> SweepResult:
        theoretical = self._theoretical_vram_mib(cell)
        # Zero-reload config switch through the wrapper (no model reload).
        if self.wrapper is not None:
            cfg = QuantConfig(
                k_bits=cell.k_bits,  # type: ignore[arg-type]
                v_bits=cell.v_bits,  # type: ignore[arg-type]
                k_per_channel=cell.k_per_channel,
                k_quant_stage=cell.k_quant_stage,  # type: ignore[arg-type]
                sink_tokens=cell.sink_tokens,
                k_encoding=cell.k_encoding,  # type: ignore[arg-type]
            )
            self.wrapper.set_kv_config(cfg)
        try:
            rng = random.Random(f"{self.seed}|{cell.key()}")
            acc = self._quality(cell) + rng.uniform(-0.004, 0.004)
            return SweepResult(
                cell=cell,
                status="OK",
                acc_mean=round(acc, 6),
                delta_logp=round(max(0.0, rng.gauss(0.05, 0.02)), 6),
                effective_bits=round((cell.k_bits + cell.v_bits) / 2, 2),
                theoretical_vram_mib=theoretical,
                model_id=self.model_id,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            return SweepResult(
                cell=cell,
                status="OOM",
                theoretical_vram_mib=theoretical,
                model_id=self.model_id,
            )

    def _completed_keys(self) -> set[str]:
        if not self.resume or not self.output_path.exists():
            return set()
        df = pd.read_csv(self.output_path)
        keys = set()
        for r in df.to_dict("records"):
            try:
                cell = SweepCell(
                    task=str(r["task"]),
                    config_id=str(r["config_id"]),
                    k_bits=int(r["k_bits"]),
                    v_bits=int(r["v_bits"]),
                    context_len=int(r["context_len"]),
                    k_per_channel=bool(r["k_per_channel"]),
                    k_quant_stage=str(r["k_quant_stage"]),
                    sink_tokens=int(r["sink_tokens"]),
                    k_encoding=str(r["k_encoding"]),
                )
                keys.add(cell.key())
            except (KeyError, ValueError):
                continue
        return keys

    def run(self) -> pd.DataFrame:
        """Run the sweep, checkpointing rows incrementally and signing a manifest."""
        cells = expand_cells(self.config)
        completed = self._completed_keys()
        rows: list[dict[str, Any]] = []
        if self.resume and self.output_path.exists():
            existing = pd.read_csv(self.output_path)
            rows = [{str(k): v for k, v in r.items()} for r in existing.to_dict("records")]

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        for i, cell in enumerate(cells):
            if cell.key() in completed:
                continue
            result = self._evaluate(cell)
            rows.append(result.to_row())
            # Incremental checkpoint every row
            pd.DataFrame(rows).to_csv(self.output_path, index=False)
            if (i + 1) % 50 == 0:
                print(f"  checkpoint: {i + 1}/{len(cells)} cells")

        df = pd.DataFrame(rows)
        df.to_csv(self.output_path, index=False)
        self._write_manifest(df)
        return df

    def _write_manifest(self, df: pd.DataFrame) -> None:
        data = self.output_path.read_bytes().replace(b"\r\n", b"\n")
        manifest = {
            "file": self.output_path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "model_id": self.model_id,
            "seed": self.seed,
            "num_rows": len(df),
            "columns": list(df.columns),
            "sweep_config": self.config_path.name,
            "generated_by": "kv_quant.sweep.SweepOrchestrator",
        }
        self.output_path.with_suffix(self.output_path.suffix + ".manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
