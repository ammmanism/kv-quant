"""Model wrapper for integrating quantized KV cache with HuggingFace models."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from kv_quant.cache import HybridPagedKVCache
from kv_quant.config import CacheConfig, QuantConfig


@dataclass
class WrappedModelOutput:
    """Output from wrapped model forward pass."""

    logits: torch.Tensor
    past_kv: HybridPagedKVCache | None = None


class KVQuantWrapper:
    """Wrapper that intercepts attention to use quantized KV cache."""

    def __init__(
        self,
        model: torch.nn.Module,
        quant_config: QuantConfig,
        cache_config: CacheConfig,
    ):
        self.model = model
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.cache: HybridPagedKVCache | None = None
        self._original_forward: dict[str, Callable] = {}

    def __enter__(self) -> "KVQuantWrapper":
        """Enable quantization."""
        raise NotImplementedError("__enter__ not implemented")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Disable quantization, restore original model."""
        raise NotImplementedError("__exit__ not implemented")

    def _patch_attention(self) -> None:
        """Patch attention modules to use quantized cache."""
        raise NotImplementedError("_patch_attention not implemented")

    def _unpatch_attention(self) -> None:
        """Restore original attention modules."""
        raise NotImplementedError("_unpatch_attention not implemented")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: HybridPagedKVCache | None = None,
        **kwargs,
    ) -> WrappedModelOutput:
        """Forward pass with quantized KV cache."""
        raise NotImplementedError("forward not implemented")

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        """Generate with quantized KV cache."""
        raise NotImplementedError("generate not implemented")
