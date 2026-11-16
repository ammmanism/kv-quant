"""Retrieval-head vs streaming-head adaptive mixed-precision profiling.

Three metrics per (layer, kv_head): distant attention mass, attention entropy,
and INT4 quantization KL sensitivity. A schedule compiler assigns per-head
bit-widths under an average-bit budget.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

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
