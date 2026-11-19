#!/usr/bin/env python3
"""Phase 12: Gaussian QJL residual sketch MSE ablation (ADR-010 reference).

Emits ``bench/data/p12_qjl_mse_ablation.csv`` (+ ``.manifest.json``).

Conventions (fixed a priori, recorded in the manifest):
- Ground-truth logits are exact float32 inner products of the FP16 fixture
  values, attention-scaled: ell = <q, k> / sqrt(D).
- Two query regimes per key vector:
  * ``aligned``: q = k / ||k|| (attention-like self-query diagnostic). The
    symmetric INT2 shrinkage grid makes naive logits systematically biased;
    this is the regime the QJL correction is designed for.
  * ``independent``: q is a seeded unit-random vector. The naive residual
    is mean-zero in inner product here, so the QJL sketch's own variance
    dominates -- disclosed boundary condition, ratio ~ (pi/2) * D / M.
- Keys: 1000 vectors of D=128 formed by seeded pairing of two KV-head
  channel blocks of the committed micro_kv_2k.pt fixture (the fixture's
  head_dim is 64, so the mandated D=128 requires pairing two heads).
- QJL rows share projection seed 42; M ladder {64, 128, 256, 512, 1024, 2048}
  is pre-declared. The mandated packing uses M=64 (two uint32 words).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kv_quant.qjl_sketch import (  # noqa: E402
    estimate_qjl_logits_ref,
    naive_int2_logit,
    quantize_k_int2_qjl,
)
from tasks.base import SweepManifest  # noqa: E402

ACT_PATH = Path("bench/activations/micro_kv_2k.pt")
DATA_DIR = Path("bench/data")
D = 128
N_HEADS = 1000
SEED = 2026
QUERY_SEED = 99
PROJ_SEED = 42
M_LADDER = (64, 128, 256, 512, 1024, 2048)


def sample_head_pairs() -> torch.Tensor:
    """1000 D=128 vectors from seeded pairs of fixture KV-head channels."""
    fixture = torch.load(ACT_PATH, map_location="cpu", weights_only=True)
    k = fixture["k"].float()  # (layers, seq, heads, 64)
    layers, seq, heads, dim = k.shape
    cand = []
    for layer in range(layers):
        for h in range(0, heads - 1, 2):
            cand.append(torch.cat([k[layer, :, h, :], k[layer, :, h + 1, :]], dim=-1))
    pool = torch.cat(cand, dim=0)  # (layers * heads/2 * seq, 128)
    generator = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(pool.shape[0], generator=generator)[:N_HEADS]
    return pool[idx]


def _rademacher_estimate(payload, k, q, m, scale, seed):  # noqa: ANN001
    """Same estimator with a Rademacher (+-1) projection, for the ablation.

    Deliberately uses the SAME sqrt(pi/2)/M constant: the claim under test is
    that the constant is wrong for Rademacher because the half-normal law
    requires rotational invariance that +-1 rows do not possess.
    """
    from kv_quant.qjl_sketch import naive_int2_logit, pack_qjl_signs, unpack_qjl_signs
    from kv_quant.quantizer import dequantize_group

    generator = torch.Generator().manual_seed(seed)
    proj = (torch.randint(0, 2, (m, payload.dim), generator=generator) * 2 - 1).float()
    k_hat = dequantize_group(payload.packed_int2, payload.int2_params, 2, "sym")
    residual = k.to(torch.float32) - k_hat.to(torch.float32)
    signs = unpack_qjl_signs(
        pack_qjl_signs(torch.nn.functional.pad(((proj @ residual) > 0).to(torch.uint8), (0, 0))), m
    ).to(torch.float32)
    base = naive_int2_logit(q, payload)
    correction = (
        math.sqrt(math.pi / 2)
        / m
        * float(payload.residual_norm)
        * float(((proj @ q) * signs).sum())
    )
    return (base + correction) / scale


def main() -> int:
    keys = sample_head_pairs()
    scale = math.sqrt(D)
    q_gen = torch.Generator().manual_seed(QUERY_SEED)
    aligned_q = [k / k.norm() for k in keys]
    independent_q = []
    for _ in keys:
        v = torch.randn(D, generator=q_gen)
        independent_q.append(v / v.norm())

    rows: list[dict] = []
    for regime, queries in (("aligned", aligned_q), ("independent", independent_q)):
        true = [float((q @ k).item()) / scale for q, k in zip(queries, keys, strict=True)]
        err_naive, err_qjl, err_rademacher = [], {m: [] for m in M_LADDER}, []
        for k, q in zip(keys, queries, strict=True):
            true_logit = float((q @ k).item()) / scale
            payload = quantize_k_int2_qjl(k, None, seed=PROJ_SEED)
            err_naive.append(naive_int2_logit(q, payload) / scale - true_logit)
            err_rademacher.append(
                _rademacher_estimate(payload, k, q, 64, scale, PROJ_SEED) - true_logit
            )
            for m in M_LADDER:
                payload_m = (
                    payload if m == 64 else quantize_k_int2_qjl(k, None, seed=PROJ_SEED, m_bits=m)
                )
                est = estimate_qjl_logits_ref(q, payload_m, seed=PROJ_SEED)
                err_qjl[m].append(est / scale - true_logit)

        def stats(errs: list[float], ratio_to: float | None = None) -> dict:
            t = torch.tensor(errs)
            row = {
                "bias_mean": round(float(t.mean()), 6),
                "bias_abs": round(float(t.mean().abs()), 6),
                "mse": round(float(t.pow(2).mean()), 6),
            }
            if ratio_to is not None and ratio_to > 0:
                row["mse_ratio_vs_naive"] = round(row["mse"] / ratio_to, 4)
            return row

        mse_naive = float(torch.tensor(err_naive).pow(2).mean())
        rows.append(
            {"regime": regime, "config": "fp16_ref", "m_bits": 0, **stats([0.0] * len(true))}
        )
        rows.append({"regime": regime, "config": "int2_naive", "m_bits": 0, **stats(err_naive)})
        rows.append(
            {
                "regime": regime,
                "config": "int2_qjl_rademacher_m64",
                "m_bits": 64,
                **stats(err_rademacher, ratio_to=mse_naive),
            }
        )
        for m in M_LADDER:
            rows.append(
                {
                    "regime": regime,
                    "config": f"int2_qjl_m{m}",
                    "m_bits": m,
                    **stats(err_qjl[m], ratio_to=mse_naive),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["regime", "m_bits"], kind="mergesort").reset_index(drop=True)
    csv = DATA_DIR / "p12_qjl_mse_ablation.csv"
    df.to_csv(csv, index=False)
    manifest = SweepManifest.capture(
        model_id="synthetic/micro_kv_2k head-pairs (no model weights)",
        param_count_b=0.0,
        num_q_heads=8,
        num_kv_heads=8,
        rotary_dim=64,
        seed=PROJ_SEED,
    )
    manifest.extra = {
        "description": "Gaussian QJL residual sketch: bias and logit MSE vs FP16 ground truth",
        "construction": (
            f"{N_HEADS} key vectors of D={D} formed by seeded pairing (seed {SEED}) "
            "of two KV-head channel blocks of micro_kv_2k.pt (fixture head_dim is 64, "
            "so the mandated D=128 requires pairing); ground truth = exact float32 "
            "inner products attention-scaled by 1/sqrt(D)"
        ),
        "query_regimes": (
            "aligned: q = k/||k|| (attention-like; naive INT2 shrinkage bias is "
            "systematic); independent: seeded unit-random q (residual mean-zero; "
            "QJL sketch variance dominates, ratio ~ (pi/2)*D/M)"
        ),
        "projection_seed": PROJ_SEED,
        "m_ladder_predeclared": list(M_LADDER),
        "packing_note": "mandated packing is m_bits=64 (two uint32 words); larger M "
        "rows are stored as unpacked sign references, not the "
        "production packing",
        "theory": "QJL estimator variance ~ (pi/2)*||r||^2*||q||^2/M (Gaussian "
        "arcsine/half-normal law); independent-regime MSE ratio vs naive "
        "~ (pi/2)*D/M",
    }
    manifest.with_csv_hash(csv).write(csv.with_suffix(csv.suffix + ".manifest.json"))
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
