"""Configuration dataclasses for kv-quant."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QuantConfig:
    """Global quantization configuration for K and V separately."""

    # Bit-width (16 = no quantization)
    k_bits: Literal[16, 8, 6, 4, 2] = 16
    v_bits: Literal[16, 8, 6, 4, 2] = 16

    # Quantization scheme
    k_scheme: Literal["sym", "asym"] = "sym"
    v_scheme: Literal["sym", "asym"] = "asym"

    # Encoding type for K
    k_encoding: Literal["int", "fp8_e4m3"] = "int"

    # Group size for block quantization
    group_size: Literal[32, 64, 128] = 64

    # Per-channel quantization for K
    k_per_channel: bool = False

    # Quantization stage relative to RoPE
    k_quant_stage: Literal["post_rope", "pre_rope"] = "post_rope"

    # Fast Walsh-Hadamard + Rademacher rotation on Q and K
    use_hadamard: bool = False

    # 1-bit QJL sign sketch on K residual
    use_qjl_residual: bool = False

    # Sink tokens (always FP16, never quantized)
    sink_tokens: int = 4


@dataclass(frozen=True)
class CacheConfig:
    """Hybrid cache configuration."""

    max_pages: int = 1024
    device: str = "cuda"
    pin_memory: bool = True


@dataclass(frozen=True)
class BenchConfig:
    """Benchmark configuration."""

    concurrency: int = 1
    isl: int = 128  # input sequence length
    osl: int = 32  # output sequence length
    warmup: int = 2
    runs: int = 5
    model: str = "meta-llama/Llama-2-7b-hf"
