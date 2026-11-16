"""Configuration dataclasses for kv-quant."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QuantConfig:
    """Global quantization configuration."""

    group_size: int = 128
    bits: int = 4
    hadamard: bool = True
    qjl_sketch: int = 0  # 0 = disabled
    sink_tokens: int = 4
    page_size: int = 256
    dtype: Literal["fp16", "bf16"] = "fp16"


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
    osl: int = 32   # output sequence length
    warmup: int = 2
    runs: int = 5
    model: str = "meta-llama/Llama-2-7b-hf"
