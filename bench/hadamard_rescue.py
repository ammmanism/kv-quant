#!/usr/bin/env python3
"""Phase 10: Hadamard rotation cliff-rescue measurements and needle sweep.

Two committed artifacts:

1. ``bench/data/p10_hadamard_rotation.csv`` -- *measured* rotation effects on
   activation fixtures (no model): per-token across-channel kurtosis and INT4
   group-quantization SQNR before/after ``fast_walsh_hadamard``.
2. ``bench/data/p10_hadamard_rescue.csv`` -- *modeled projection* of needle
   accuracy: replay-adapter cells for fp16 and k4v4_postrope (the committed
   v1.0 model) are shifted by the measured SQNR gain using the a-priori error
   -power rule acc = acc_fp16 - (acc_fp16 - acc_k4v4) * 10**(-delta_db/10).

Honesty: ``micro_kv_2k.pt`` is a Gaussian synthetic fixture (per-channel
kurtosis ~0), so it contains no channel outliers to rescue -- it is measured
and reported as a null control. The rescue is measured on
``micro_kv_2k_outliers.pt``, a deterministic outlier-injection derivative
built here (massive-channel scales, documented parameters), NOT on real model
activations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kv_quant.hadamard import compute_kurtosis, fast_walsh_hadamard  # noqa: E402
from kv_quant.quantizer import compute_sqnr_db, dequantize_group, quantize_group  # noqa: E402
from tasks.base import RecordedReplayAdapter, SweepManifest, wilson_ci  # noqa: E402
from tasks.needle import PositionResolvedNeedleTask  # noqa: E402

ACT_DIR = Path("bench/activations")
DATA_DIR = Path("bench/data")
PRISTINE = ACT_DIR / "micro_kv_2k.pt"
OUTLIERS = ACT_DIR / "micro_kv_2k_outliers.pt"

SEED = 2026
INJECT_N_CHANNELS = 2
INJECT_SCALE_LO = 8.0
INJECT_SCALE_HI = 32.0
SQNR_SAMPLE_ROWS = 512
CONTEXTS = (4096, 16384, 32768, 65536, 131072)
ROTARY_DIM = 32  # partial-RoPE instantiation on the 64-dim fixture head

# Pre-declared sensitivity ladder: the primary (conservative) injection is
# (2 channels, U(8,32)); stronger massive-activation regimes are reported as
# ablations, not used to pick the headline number.
SENSITIVITY = [
    ("outliers_2ch_8_32", 2, 8.0, 32.0),
    ("outliers_2ch_20_50", 2, 20.0, 50.0),
    ("outliers_3ch_20_60", 3, 20.0, 60.0),
    ("outliers_1ch_10_40", 1, 10.0, 40.0),
]


def inject_outliers(k: torch.Tensor, n_chan: int, lo: float, hi: float, seed: int) -> torch.Tensor:
    """Scale n_chan random channels per (layer, head) by U(lo, hi) (pure)."""
    out = k.float().clone()
    gen = torch.Generator().manual_seed(seed)
    num_layers, _, num_heads, head_dim = out.shape
    for layer in range(num_layers):
        for head in range(num_heads):
            idx = torch.randperm(head_dim, generator=gen)[:n_chan]
            scale = lo + (hi - lo) * torch.rand(n_chan, generator=gen)
            out[layer, :, head, idx] = out[layer, :, head, idx] * scale.view(1, -1)
    return out


def build_outlier_fixture() -> Path:
    """Deterministic massive-channel outlier derivative of micro_kv_2k.pt."""
    payload = torch.load(PRISTINE, map_location="cpu", weights_only=False)
    k = inject_outliers(payload["k"], INJECT_N_CHANNELS, INJECT_SCALE_LO, INJECT_SCALE_HI, SEED)
    payload["k"] = k.to(torch.float16)
    payload["metadata"] = {
        **payload["metadata"],
        "source": (
            "synthetic outlier-injection derivative of micro_kv_2k.pt "
            f"(seed {SEED}, {INJECT_N_CHANNELS} massive channels/head scaled "
            f"U({INJECT_SCALE_LO},{INJECT_SCALE_HI}))"
        ),
    }
    OUTLIERS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUTLIERS)
    (ACT_DIR / "micro_kv_2k_outliers.injection.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "channels_per_head": INJECT_N_CHANNELS,
                "scale_lo": INJECT_SCALE_LO,
                "scale_hi": INJECT_SCALE_HI,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return OUTLIERS


def _sample_rows(x: torch.Tensor, n: int = SQNR_SAMPLE_ROWS) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1])
    step = max(1, rows.shape[0] // n)
    return rows[::step][:n]


def _int4_sqnr(rows: torch.Tensor, group: int = 32) -> float:
    vals = []
    for row in rows:
        recon = torch.cat(
            [dequantize_group(*quantize_group(c, 4, "sym"), 4, "sym") for c in row.split(group)]
        )
        vals.append(compute_sqnr_db(row, recon))
    return float(torch.tensor(vals).mean())


def measure_fixture(name: str, k: torch.Tensor) -> list[dict]:
    """Kurtosis + INT4 SQNR before/after rotation, per scope, on one tensor.

    Kurtosis is reported as Pearson kurtosis (excess + 3): 3.0 is Gaussian.
    """
    k = k.float()
    sample = _sample_rows(k)
    base_sqnr = _int4_sqnr(sample)
    rows = []
    for scope in ("block_diag", "full"):
        # Kurtosis window follows the scope: block_diag rotates only the rotary
        # subspace, so that subspace is the meaningful window; full rotates all
        # channels, so restricting the window to rotary_dim would understate it.
        if scope == "block_diag":
            window = slice(0, ROTARY_DIM)
            window_name = "rotary_subspace"
        else:
            window = slice(0, k.shape[-1])
            window_name = "full_head"
        pre_kurt = compute_kurtosis(k[..., window], dim=-1).mean().item() + 3.0
        rotated = fast_walsh_hadamard(k, scope=scope, rotary_dim=ROTARY_DIM, seed=42)
        post_kurt = compute_kurtosis(rotated[..., window], dim=-1).mean().item() + 3.0
        sqnr = _int4_sqnr(_sample_rows(rotated))
        rows.append(
            {
                "fixture": name,
                "scope": scope,
                "kurtosis_window": window_name,
                "rotary_dim": ROTARY_DIM,
                "head_dim": k.shape[-1],
                "pre_kurtosis": round(pre_kurt, 3),
                "post_kurtosis": round(post_kurt, 3),
                "sqnr_before_db": round(base_sqnr, 3),
                "sqnr_after_db": round(sqnr, 3),
                "sqnr_delta_db": round(sqnr - base_sqnr, 3),
                "sample_rows": SQNR_SAMPLE_ROWS,
            }
        )
    return rows


def needle_projection(delta_db: float) -> pd.DataFrame:
    """Projected needle accuracy: replay baseline shifted by error-power ratio."""
    factor = 10.0 ** (-delta_db / 10.0)
    task = PositionResolvedNeedleTask(RecordedReplayAdapter(trials_per_cell=20))
    manifest = SweepManifest.capture(seed=42)
    raw = task.to_dataframe(task.run(manifest))
    base = raw[raw.config_id.isin(["fp16", "k4v4_postrope_sink4"])]
    out = []
    for cfg, label in (("fp16", "fp16"), ("k4v4_postrope_sink4", "k4v4_postrope")):
        for ctx in CONTEXTS:
            cells = base[(base.config_id == cfg) & (base.context_len == ctx)]
            correct, n = int(cells["correct"].sum()), int(cells["trials"].sum())
            lo, hi = wilson_ci(correct, n)
            out.append(
                {
                    "config_id": label,
                    "context_len": ctx,
                    "trials": n,
                    "acc_mean": round(correct / n, 4),
                    "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4),
                    "delta_logp": round(float(cells["delta_logp"].mean()), 4),
                    "projection_delta_db": 0.0,
                }
            )
    for ctx in CONTEXTS:
        f = base[(base.config_id == "fp16") & (base.context_len == ctx)]
        q4 = base[(base.config_id == "k4v4_postrope_sink4") & (base.context_len == ctx)]
        merged = f.merge(q4, on=["depth_pct", "needle_format"], suffixes=("_f", "_q"))
        proj = merged["acc_mean_f"] - (merged["acc_mean_f"] - merged["acc_mean_q"]) * factor
        trials = int(merged["trials_f"].sum())
        correct = int(round(float((proj * merged["trials_f"]).sum())))
        lo, hi = wilson_ci(correct, trials)
        drift = merged["delta_logp_f"] + (merged["delta_logp_q"] - merged["delta_logp_f"]) * factor
        out.append(
            {
                "config_id": "k4v4_hadamard_projected",
                "context_len": ctx,
                "trials": trials,
                "acc_mean": round(correct / trials, 4),
                "ci95_low": round(lo, 4),
                "ci95_high": round(hi, 4),
                "delta_logp": round(float(drift.mean()), 4),
                "projection_delta_db": round(delta_db, 3),
            }
        )
    return pd.DataFrame(out)


def write_manifest(csv_path: Path, description: str, extra: dict) -> None:
    # These CSVs are measured on a synthetic micro-fixture, not on a real
    # model: never inherit the default Qwen2.5-7B provenance from capture().
    manifest = SweepManifest.capture(
        model_id="synthetic/micro_kv_2k (no model weights)",
        param_count_b=0.0,
        num_q_heads=8,
        num_kv_heads=8,
        rotary_dim=ROTARY_DIM,
        seed=42,
    )
    manifest.extra = {"description": description, **extra}
    manifest.with_csv_hash(csv_path).write(csv_path.with_suffix(csv_path.suffix + ".manifest.json"))


def main() -> int:
    if not OUTLIERS.exists():
        build_outlier_fixture()
        print(f"[OK] built {OUTLIERS}")
    pristine = torch.load(PRISTINE, map_location="cpu", weights_only=False)["k"]
    rotation_rows = measure_fixture("micro_kv_2k (null control)", pristine) + measure_fixture(
        "micro_kv_2k_outliers (primary)",
        torch.load(OUTLIERS, map_location="cpu", weights_only=False)["k"],
    )
    for label, n_chan, lo, hi in SENSITIVITY[1:]:
        rotation_rows += measure_fixture(
            f"{label} (sensitivity)", inject_outliers(pristine, n_chan, lo, hi, SEED)
        )
    rot_df = pd.DataFrame(rotation_rows)
    rot_csv = DATA_DIR / "p10_hadamard_rotation.csv"
    rot_df.to_csv(rot_csv, index=False)
    write_manifest(
        rot_csv,
        "Measured Hadamard rotation effect on activation fixtures: kurtosis (Pearson) and INT4 SQNR",
        {
            "fixtures": [PRISTINE.as_posix(), OUTLIERS.as_posix()],
            "fixture_regeneration": "outliers fixture is gitignored and rebuilt "
            "deterministically by bench/hadamard_rescue.py from "
            "micro_kv_2k.pt + the committed injection.json",
            "measurement": "deterministic sample of 512 rows; INT4 sym group=32; "
            "kurtosis window is scope-dependent (see kurtosis_window column): "
            "rotary_subspace = first rotary_dim channels for block_diag, "
            "full_head = all head_dim channels for full; Pearson convention "
            "(3.0 = Gaussian); sensitivity rows are in-memory deterministic "
            "variants of the pristine fixture, primary injection is "
            f"{INJECT_N_CHANNELS} channels U({INJECT_SCALE_LO},{INJECT_SCALE_HI})",
        },
    )
    delta_block = float(
        rot_df[
            (rot_df.fixture.str.startswith("micro_kv_2k_outliers")) & (rot_df.scope == "block_diag")
        ].sqnr_delta_db.iloc[0]
    )
    sweep_df = needle_projection(delta_block)
    sweep_csv = DATA_DIR / "p10_hadamard_rescue.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    write_manifest(
        sweep_csv,
        "Modeled projection of needle accuracy under block-diagonal Hadamard rotation",
        {
            "delta_sqnr_db": round(delta_block, 3),
            "projection": "acc = acc_fp16 - (acc_fp16 - acc_k4v4) * 10**(-delta_db/10); "
            "delta_logp scaled identically; derived from a replay-adapter baseline, "
            "not from real-model forward passes",
            "config_id_note": "'k4v4_hadamard_projected' is a modeled projection, not a measured run",
            "ci_semantics": "ci95_low/ci95_high are Wilson intervals over the baseline trial "
            "counts; they quantify sampling noise of the replay baseline and do NOT "
            "cover projection-model error, which is unbounded by this CSV",
            "source_rotation_csv": rot_csv.as_posix(),
        },
    )
    print(rot_df.to_string(index=False))
    print(sweep_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
