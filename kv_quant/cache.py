"""3-Tier Hybrid KV Cache with Copy-on-Write prefix hashing."""

from dataclasses import dataclass

import torch

from kv_quant.config import CacheConfig, QuantConfig
from kv_quant.layout import PageLayout, get_layout
from kv_quant.quantizer import QuantParams


@dataclass(frozen=True)
class PageRef:
    """Reference to a quantized page (copy-on-write)."""
    page_id: int
    prefix_hash: int  # Hash of prefix for CoW deduplication
    ref_count: int = 1


class HybridPagedKVCache:
    """3-Tier Hybrid KV Cache:
    1. Sink tokens (FP16, never quantized)
    2. Quantized pages (sealed, CoW with prefix hashing)
    3. Active ring buffer (FP16, < group_size tokens)
    """

    def __init__(
        self,
        quant_config: QuantConfig,
        cache_config: CacheConfig,
        layout: PageLayout | None = None,
    ):
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.layout = layout or get_layout()

        # Tier 1: Sink tokens (FP16)
        self.sink_k: torch.Tensor | None = None  # (sink_tokens, head_dim)
        self.sink_v: torch.Tensor | None = None

        # Tier 2: Quantized pages (sealed, CoW)
        self.pages_k: list[tuple[torch.Tensor, list[QuantParams], PageRef]] = []
        self.pages_v: list[tuple[torch.Tensor, list[QuantParams], PageRef]] = []

        # Tier 3: Active ring buffer (FP16, trailing < group_size)
        self.ring_k: torch.Tensor | None = None  # (group_size - 1, head_dim)
        self.ring_v: torch.Tensor | None = None
        self.ring_start: int = 0
        self.ring_len: int = 0

        # Page allocator
        self._next_page_id: int = 0
        self._page_pool: dict[int, PageRef] = {}

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append new KV tokens to the cache.

        Args:
            k: Key tensor of shape (seq_len, num_heads, head_dim) or (seq_len, head_dim).
            v: Value tensor of same shape as k.
        """
        raise NotImplementedError("append not implemented")

    def get(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve KV at given indices.

        Args:
            indices: Tensor of shape (num_indices,) with token indices.

        Returns:
            (k, v) tensors of shape (num_indices, num_heads, head_dim).
        """
        raise NotImplementedError("get not implemented")

    def seal_page(self) -> None:
        """Seal current ring buffer into a quantized page with CoW."""
        raise NotImplementedError("seal_page not implemented")

    def _compute_prefix_hash(self, page_data: torch.Tensor) -> int:
        """Compute prefix hash for CoW deduplication."""
        raise NotImplementedError("_compute_prefix_hash not implemented")

    def _get_or_create_page(self, prefix_hash: int) -> PageRef:
        """Get existing page with prefix_hash or create new (CoW)."""
        raise NotImplementedError("_get_or_create_page not implemented")

    @property
    def total_tokens(self) -> int:
        """Total tokens in cache (sink + pages + ring)."""
        sink = self.quant_config.sink_tokens if self.sink_k is not None else 0
        pages = len(self.pages_k) * self.layout.page_size
        ring = self.ring_len
        return sink + pages + ring

    @property
    def vram_bytes(self) -> int:
        """Estimated VRAM usage in bytes."""
        raise NotImplementedError("vram_bytes not implemented")
