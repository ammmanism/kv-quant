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

    # Rotation scope: "none" (auto-resolve when use_hadamard=True),
    # "full" (entire head_dim) or "block_diag" (rotary subspace only)
    hadamard_scope: Literal["none", "full", "block_diag"] = "none"

    # Architecture rotary width; when < head_dim the rotation is block-diagonal
    rotary_dim: int | None = None

    # Seed for the Rademacher sign flips D = diag(+-1)
    hadamard_seed: int = 42

    # 1-bit QJL sign sketch on K residual
    use_qjl_residual: bool = False

    # Sink tokens (always FP16, never quantized)
    sink_tokens: int = 4

    def resolve_hadamard_scope(
        self, head_dim: int, rotary_dim: int | None = None
    ) -> Literal["none", "full", "block_diag"]:
        """Resolve the effective Hadamard scope for a model geometry.

        Precedence: an explicit ``hadamard_scope`` wins; otherwise a model
        with partial RoPE (``rotary_dim < head_dim``) resolves to
        ``"block_diag"`` and a full-RoPE model resolves to ``"full"``. When
        rotation is disabled the effective scope is ``"none"``.

        Args:
            head_dim: Per-head channel width (e.g. 128).
            rotary_dim: Architecture rotary width; overrides ``self.rotary_dim``
                when provided.

        Returns:
            One of ``"none"``, ``"full"``, ``"block_diag"``.

        Raises:
            ValueError: If rotation is enabled but no rotary width can be
                determined, or the width is outside ``(0, head_dim]``.
        """
        if not self.use_hadamard:
            return "none"
        if self.hadamard_scope != "none":
            return self.hadamard_scope
        rd = rotary_dim if rotary_dim is not None else self.rotary_dim
        if rd is None:
            raise ValueError(
                "Hadamard rotation enabled but rotary_dim is unknown; "
                "set QuantConfig.rotary_dim or pass rotary_dim explicitly"
            )
        if not 0 < rd <= head_dim:
            raise ValueError(f"rotary_dim={rd} must be in (0, head_dim={head_dim}]")
        return "full" if rd == head_dim else "block_diag"


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
