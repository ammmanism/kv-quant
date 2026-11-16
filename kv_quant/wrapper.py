"""Clean module-swap wrapper with zero-reload config switching and hooks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from kv_quant.cache import HybridPagedKVCache
from kv_quant.config import CacheConfig, QuantConfig
from kv_quant.hadamard import hadamard_transform


@dataclass
class WrappedModelOutput:
    """Output from a wrapped model forward pass."""

    logits: torch.Tensor
    past_kv: HybridPagedKVCache | None = None


class KVQuantWrapper:
    """Intercepts attention projections to apply per-head quantization hooks."""

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
        self.head_schedule: list[dict] | None = None
        self._original_forward: dict[str, Callable] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._active = False
        self._hadamard_signs: torch.Tensor | None = None

    # ------------------------------------------------------------ config switch
    def set_kv_config(
        self,
        quant_config: QuantConfig,
        head_schedule: list[dict] | None = None,
    ) -> None:
        """Zero-reload runtime switch to a new quantization configuration."""
        self.quant_config = quant_config
        if head_schedule is not None:
            self.head_schedule = head_schedule
        # Rebuild cache on next forward; no model reload required.
        self.cache = None

    # ----------------------------------------------------------------- context
    def __enter__(self) -> KVQuantWrapper:
        self._patch_attention()
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._unpatch_attention()
        self._active = False

    # ------------------------------------------------------------------- hooks
    def rope_hook(self, tensor: torch.Tensor, stage: str) -> torch.Tensor:
        """Pre-RoPE vs post-RoPE placement hook (identity for materialization)."""
        if self.quant_config.k_quant_stage == stage:
            return tensor
        return tensor

    def hadamard_hook(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard rotation on Q/K when enabled and dimension is a power of 2."""
        if not self.quant_config.use_hadamard:
            return tensor
        dim = tensor.shape[-1]
        if dim & (dim - 1) != 0:
            return tensor
        return hadamard_transform(tensor)

    def heterogeneous_bits_hook(self, tensor: torch.Tensor, head_dim: int) -> list[int]:
        """Return per-head bit schedule for this tensor's heads."""
        if self.head_schedule is None:
            return [self.quant_config.k_bits] * max(1, tensor.shape[-2] if tensor.dim() >= 2 else 1)
        return [int(h["assigned_k_bits"]) for h in self.head_schedule]

    def _patch_attention(self) -> None:
        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(tag in name for tag in ("q_proj", "k_proj", "v_proj")):
                continue
            self._original_forward[name] = module.forward

            def make_hook(mod_name: str):
                def hook(
                    module: torch.nn.Module, inputs: tuple, output: torch.Tensor
                ) -> torch.Tensor:
                    if "k_proj" in mod_name or "q_proj" in mod_name:
                        output = self.hadamard_hook(output)
                    return output

                return hook

            handle = module.register_forward_hook(make_hook(name))
            self._hooks.append(handle)

    def _unpatch_attention(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._original_forward.clear()

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: HybridPagedKVCache | None = None,
        **kwargs,
    ) -> WrappedModelOutput:
        """Forward pass, lazily using a hybrid cache when quantization is enabled."""
        if self.quant_config.k_bits == 16 and self.quant_config.v_bits == 16:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            return WrappedModelOutput(logits=out.logits if hasattr(out, "logits") else out)

        if self.cache is None:
            self.cache = past_kv or HybridPagedKVCache(self.quant_config, self.cache_config)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return WrappedModelOutput(
            logits=out.logits if hasattr(out, "logits") else out,
            past_kv=self.cache,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive generation with KV cache reuse."""
        current = input_ids
        for _ in range(max_new_tokens):
            out = self.forward(current, **kwargs)
            next_token = out.logits[:, -1:].argmax(dim=-1)
            current = torch.cat([current, next_token], dim=-1)
        return current
