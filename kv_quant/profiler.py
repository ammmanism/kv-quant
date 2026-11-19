"""Retrieval-head vs streaming-head adaptive mixed-precision profiling.

Three metrics per (layer, kv_head): distant attention mass, attention entropy,
and INT4 quantization KL perturbation. A schedule compiler assigns per-head
bit-widths under an average-bit budget.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import pandas as pd
import torch


@dataclass(frozen=True)
class ProfileResult:
    """Kernel timing result."""

    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    iterations: int


@dataclass(frozen=True)
class HeadProfile:
    """Sensitivity profile for one KV head."""

    layer_idx: int
    kv_head_idx: int
    distant_attn_mass: float
    attn_entropy: float
    kl_sensitivity: float
    head_class: str = "contextual"


# Backwards-compatible kernel profiling API -----------------------------------


def profile_kernel(
    fn: Callable[[], None],
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cuda",
) -> ProfileResult:
    import time

    for _ in range(warmup):
        fn()
    times_us = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times_us.append((time.perf_counter() - start) * 1e6)
    t = torch.tensor(times_us)
    return ProfileResult(
        mean_us=t.mean().item(),
        std_us=t.std().item(),
        min_us=t.min().item(),
        max_us=t.max().item(),
        iterations=iterations,
    )


def profile_memory(fn: Callable[[], None], device: str = "cuda") -> dict[str, int]:
    allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    fn()
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return {"allocated_bytes": int(allocated), "reserved_bytes": 0, "peak_bytes": int(peak)}


def benchmark_throughput(
    fn: Callable[[], None],
    batch_size: int,
    warmup: int = 5,
    iterations: int = 20,
    device: str = "cuda",
) -> float:
    result = profile_kernel(fn, warmup, iterations, device)
    return batch_size / (result.mean_us * 1e-6) if result.mean_us > 0 else float("inf")


# Head sensitivity profiling ---------------------------------------------------


def _shannon_entropy(p: torch.Tensor) -> float:
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())


def profile_heads(
    attn_weights: torch.Tensor,
    layer_idx: int = 0,
    sink_tokens: int = 4,
    local_window: int = 128,
    quantize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> list[HeadProfile]:
    """Profile every KV head in a layer.

    Args:
        attn_weights: Tensor (num_heads, q_len, k_len) of attention probabilities.
        layer_idx: Layer index tag for the returned profiles.
        sink_tokens: Number of leading sink tokens excluded from "distant".
        local_window: Size of the trailing local window excluded from "distant".
        quantize_fn: Optional INT4 perturbation applied to logits for KL measurement.

    Returns:
        One HeadProfile per head.
    """
    num_heads = attn_weights.shape[0]
    k_len = attn_weights.shape[-1]
    boundary = min(k_len, sink_tokens + local_window)
    profiles: list[HeadProfile] = []

    for h in range(num_heads):
        p = attn_weights[h].to(torch.float32)
        # 1. Distant attention mass (mean probability outside sink+local window)
        n_queries = max(1, p.shape[0])
        distant = 0.0
        if k_len > boundary:
            distant = float(p[..., : max(0, k_len - local_window)].sum().item()) / n_queries
        # Protect against sink counting: exclude first sink_tokens
        if k_len > sink_tokens:
            distant = (
                float(p[..., sink_tokens : max(sink_tokens, k_len - local_window)].sum().item())
                / n_queries
            )

        # 2. Shannon entropy (mean over query positions)
        entropies = [_shannon_entropy(row) for row in p.reshape(-1, k_len)]
        entropy = sum(entropies) / max(1, len(entropies))

        # 3. KL sensitivity under INT4 logit perturbation
        logits = p.clamp_min(1e-12).log()
        if quantize_fn is None:
            # Deterministic synthetic INT4-style perturbation
            torch.manual_seed(layer_idx * 1000 + h)
            noise = (torch.randint(-2, 3, logits.shape, dtype=torch.int32).to(torch.float32)) * 0.15
            qlogits = logits + noise
        else:
            qlogits = quantize_fn(logits)
        q = torch.softmax(qlogits, dim=-1).clamp_min(1e-12)
        kl = float((p.clamp_min(1e-12) * (p.clamp_min(1e-12).log() - q.log())).sum().item())
        kl /= max(1, p.shape[0])

        profiles.append(
            HeadProfile(
                layer_idx=layer_idx,
                kv_head_idx=h,
                distant_attn_mass=distant,
                attn_entropy=entropy,
                kl_sensitivity=abs(kl),
            )
        )
    return profiles


# Schedule compilation ---------------------------------------------------------

HEAD_SCHEDULES = {
    "retrieval": {
        "assigned_k_bits": 8,
        "assigned_v_bits": 4,
        "use_hadamard": False,
        "use_qjl": False,
        "label": "K8V4",
    },
    "contextual": {
        "assigned_k_bits": 4,
        "assigned_v_bits": 4,
        "use_hadamard": True,
        "use_qjl": False,
        "label": "K4V4_Hadamard",
    },
    "streaming": {
        "assigned_k_bits": 2,
        "assigned_v_bits": 2,
        "use_hadamard": False,
        "use_qjl": True,
        "label": "K2_QJL_V2",
    },
}


def compile_head_bit_schedule(
    profiles: list[HeadProfile],
    target_avg_bits: float = 3.5,
) -> list[dict[str, object]]:
    """Assign mixed-precision schedules: top 20% retrieval, mid 50% contextual,
    bottom 30% streaming, ranked by (distant_mass + kl_sensitivity)."""
    if not profiles:
        return []

    ranked = sorted(
        profiles,
        key=lambda p: p.distant_attn_mass + p.kl_sensitivity,
        reverse=True,
    )
    n = len(ranked)
    n_ret = max(1, round(n * 0.2))
    n_ctx = max(1, round(n * 0.5))
    n_str = n - n_ret - n_ctx
    if n_str < 1:
        n_str = 1
        n_ctx = max(1, n - n_ret - 1)

    schedule: list[dict[str, object]] = []
    for rank, prof in enumerate(ranked):
        if rank < n_ret:
            cls = "retrieval"
        elif rank < n_ret + n_ctx:
            cls = "contextual"
        else:
            cls = "streaming"
        spec = HEAD_SCHEDULES[cls]
        schedule.append(
            {
                "layer_idx": prof.layer_idx,
                "kv_head_idx": prof.kv_head_idx,
                "head_class": spec["label"],
                "assigned_k_bits": spec["assigned_k_bits"],
                "assigned_v_bits": spec["assigned_v_bits"],
                "use_hadamard": spec["use_hadamard"],
                "use_qjl": spec["use_qjl"],
                "distant_attn_mass": prof.distant_attn_mass,
                "attn_entropy": prof.attn_entropy,
                "kl_sensitivity": prof.kl_sensitivity,
            }
        )
    return schedule


def head_profiles_to_rows(profiles: list[HeadProfile]) -> list[dict[str, object]]:
    """Serialize HeadProfile objects to CSV-compatible rows."""
    return [asdict(p) for p in profiles]


def profile_calibration(
    attn_logits: torch.Tensor,
    sink_tokens: int = 4,
    local_window: int = 128,
    quantize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> list[HeadProfile]:
    """Profile every (layer, kv_head) over a multi-sequence calibration set.

    Args:
        attn_logits: Raw attention logits of shape
            (num_layers, num_kv_heads, num_seqs, q_len, k_len).
        sink_tokens: Leading sink tokens excluded from "distant" mass.
        local_window: Trailing local window excluded from "distant" mass.
        quantize_fn: Optional INT4 perturbation applied to logits for the KL
            measurement; when None a deterministic INT4-style noise is used.

    Returns:
        One HeadProfile per (layer, kv_head), each metric averaged over the
        calibration sequences.

    Raises:
        ValueError: If ``attn_logits`` does not have 5 dimensions.
    """
    if attn_logits.dim() != 5:
        raise ValueError(
            f"expected (layers, kv_heads, seqs, q_len, k_len), got {tuple(attn_logits.shape)}"
        )
    num_layers, num_heads, num_seqs = (
        attn_logits.shape[0],
        attn_logits.shape[1],
        attn_logits.shape[2],
    )
    probs = torch.softmax(attn_logits.to(torch.float32), dim=-1)

    per_seq: dict[tuple[int, int], list[HeadProfile]] = {}
    for layer in range(num_layers):
        for seq in range(num_seqs):
            # profile_heads consumes (num_heads, q_len, k_len) probabilities.
            for prof in profile_heads(
                probs[layer, :, seq],
                layer_idx=layer,
                sink_tokens=sink_tokens,
                local_window=local_window,
                quantize_fn=quantize_fn,
            ):
                per_seq.setdefault((prof.layer_idx, prof.kv_head_idx), []).append(prof)

    profiles: list[HeadProfile] = []
    for layer in range(num_layers):
        for head in range(num_heads):
            seqs = per_seq[(layer, head)]
            n = max(1, len(seqs))
            profiles.append(
                HeadProfile(
                    layer_idx=layer,
                    kv_head_idx=head,
                    distant_attn_mass=sum(p.distant_attn_mass for p in seqs) / n,
                    attn_entropy=sum(p.attn_entropy for p in seqs) / n,
                    kl_sensitivity=sum(p.kl_sensitivity for p in seqs) / n,
                )
            )
    return profiles


def compile_head_schedule(
    df: pd.DataFrame,
    target_avg_bits: float = 4.5,
    retrieval_k_bits: int = 8,
    retrieval_v_bits: int = 4,
    streaming_k_bits: int = 4,
    streaming_v_bits: int = 4,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Assign per-head bit-widths under an average-bit budget (ADR-009).

    Heads are ranked by the retrieval score ``distant_attn_mass +
    kl_perturbation`` (column name ``kl_sensitivity`` is accepted as a
    legacy alias). The top fraction -- derived from the budget so that the
    schedule's mean of (K + V) / 2 bits equals ``target_avg_bits`` -- becomes
    a Retrieval/Induction head at ``(retrieval_k_bits, retrieval_v_bits)``
    (default K8V4, protecting long-range softmax routing); every other head
    becomes a Local/Streaming head at ``(streaming_k_bits, streaming_v_bits)``
    (default K4V4 + Hadamard). At the default budget of 4.5 bits the derived
    fraction is exactly 25%.

    Ties are broken by (layer_idx, kv_head_idx) so the assignment is
    deterministic for a fixed frame.

    Args:
        df: Frame with columns ``layer_idx``, ``kv_head_idx``,
            ``distant_attn_mass`` and ``kl_perturbation``.
        target_avg_bits: Budget for the schedule-wide mean of (K + V) / 2.
        retrieval_k_bits: K bits for retrieval heads.
        retrieval_v_bits: V bits for retrieval heads.
        streaming_k_bits: K bits for streaming heads.
        streaming_v_bits: V bits for streaming heads.

    Returns:
        Mapping ``(layer_idx, kv_head_idx) -> (k_bits, v_bits)``.

    Raises:
        ValueError: If required columns are missing, the frame is empty, or
            the budget is outside the reachable range between the two classes'
            average bits.
    """
    for col in ("layer_idx", "kv_head_idx", "distant_attn_mass", "kl_perturbation"):
        if col not in df.columns and not (
            col == "kl_perturbation" and "kl_sensitivity" in df.columns
        ):
            raise ValueError(f"missing required column: {col}")
    if len(df) == 0:
        raise ValueError("cannot compile a schedule from an empty frame")

    kl_col = "kl_perturbation" if "kl_perturbation" in df.columns else "kl_sensitivity"
    frame = df.copy()
    frame["score"] = frame["distant_attn_mass"] + frame[kl_col]
    ranked = frame.sort_values(
        ["score", "layer_idx", "kv_head_idx"], ascending=[False, True, True], kind="mergesort"
    )

    high_avg = (retrieval_k_bits + retrieval_v_bits) / 2
    low_avg = (streaming_k_bits + streaming_v_bits) / 2
    if not low_avg <= target_avg_bits <= high_avg:
        raise ValueError(
            f"target_avg_bits={target_avg_bits} outside reachable range "
            f"[{low_avg}, {high_avg}] for the two head classes"
        )
    fraction = (target_avg_bits - low_avg) / (high_avg - low_avg) if high_avg > low_avg else 0.0

    n = len(ranked)
    n_retrieval = min(n, max(1, round(n * fraction))) if fraction > 0 else 0
    schedule: dict[tuple[int, int], tuple[int, int]] = {}
    for rank, (_, row) in enumerate(ranked.iterrows()):
        is_retrieval = rank < n_retrieval
        schedule[(int(row["layer_idx"]), int(row["kv_head_idx"]))] = (
            (retrieval_k_bits, retrieval_v_bits)
            if is_retrieval
            else (streaming_k_bits, streaming_v_bits)
        )
    return schedule
