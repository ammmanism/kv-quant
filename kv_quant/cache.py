"""3-Tier Hybrid Paged KV Cache with Copy-on-Write prefix sharing.

Tier 1: sink tokens pinned in FP16 (never quantized).
Tier 2: sealed, quantized physical pages with ref-counted content-addressed sharing.
Tier 3: rolling FP16 tail buffer that auto-seals into Tier 2 at group_size.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from kv_quant.config import CacheConfig, QuantConfig
from kv_quant.layout import PageLayout, get_layout
from kv_quant.quantizer import QuantParams, dequantize_group, quantize_group


@dataclass
class PhysicalPage:
    """A sealed quantized page stored once in the physical pool."""

    page_id: int
    prefix_hash: int
    k_payload: torch.Tensor  # (heads, dim, packed_len) uint8
    v_payload: torch.Tensor
    k_params: list[QuantParams]
    v_params: list[QuantParams]
    # Per-head K/V bit-widths the payloads were sealed with (ADR-009). Empty
    # means the whole page used the config defaults (pre-ADR-009 pages).
    k_bits: list[int] = field(default_factory=list)
    v_bits: list[int] = field(default_factory=list)
    ref_count: int = 1


@dataclass(frozen=True)
class PageRef:
    """Lightweight reference to a physical page (copy-on-write)."""

    page_id: int
    prefix_hash: int
    ref_count: int = 1


class HybridPagedKVCache:
    """3-tier hybrid cache with CoW prefix sharing across agent sessions."""

    def __init__(
        self,
        quant_config: QuantConfig,
        cache_config: CacheConfig,
        layout: PageLayout | None = None,
    ):
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.layout = layout or get_layout()
        self.group_size: int = 64  # sealed block size (per spec)

        # Tier 1: sink
        self.sink_k: torch.Tensor | None = None
        self.sink_v: torch.Tensor | None = None
        self.sink_len: int = 0

        # Tier 2: physical pool + logical page table
        self._pages: dict[int, PhysicalPage] = {}
        self._pages_by_hash: dict[int, int] = {}
        self.page_table: list[int] = []
        self._next_page_id: int = 0

        # Tier 3: ring tail
        self.ring_k: torch.Tensor | None = None
        self.ring_v: torch.Tensor | None = None
        self.ring_len: int = 0
        self._head_dim: int = 0
        self._num_heads: int = 0
        self._device: torch.device = torch.device("cpu")

        # ADR-009 per-head mixed precision: kv_head_idx -> (k_bits, v_bits).
        # None means every head uses the config defaults.
        self.head_schedule: dict[int, tuple[int, int]] | None = None

        # HBM accounting (naive path telemetry)
        self.hbm_read_bytes: int = 0
        self.hbm_write_bytes: int = 0

    def set_head_schedule(
        self,
        schedule: dict[tuple[int, int], tuple[int, int]],
        layer_idx: int | None = None,
    ) -> None:
        """Install an ADR-009 head schedule for this (per-layer) cache.

        Args:
            schedule: Mapping ``(layer_idx, kv_head_idx) -> (k_bits, v_bits)``
                as produced by ``profiler.compile_head_schedule``.
            layer_idx: The layer this cache instance stores. When omitted, the
                schedule must contain exactly one layer.

        Raises:
            ValueError: If ``layer_idx`` is omitted and the schedule spans
                multiple layers, or no head matches the requested layer.
        """
        if layer_idx is None:
            layers = {layer for layer, _ in schedule}
            if len(layers) != 1:
                raise ValueError("layer_idx is required when the schedule spans multiple layers")
            layer_idx = next(iter(layers))
        heads = {head: bits for (layer, head), bits in schedule.items() if layer == layer_idx}
        if not heads:
            raise ValueError(f"schedule contains no heads for layer_idx={layer_idx}")
        self.head_schedule = heads

    def _k_bits_for(self, head: int) -> int:
        if self.head_schedule is not None and head in self.head_schedule:
            return self.head_schedule[head][0]
        return self.quant_config.k_bits

    def _v_bits_for(self, head: int) -> int:
        if self.head_schedule is not None and head in self.head_schedule:
            return self.head_schedule[head][1]
        return self.quant_config.v_bits

    # ------------------------------------------------------------------ append
    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append KV tokens, filling sink -> ring -> sealed pages."""
        if k.shape != v.shape:
            raise ValueError("k and v must have identical shapes")
        if k.dim() == 2:
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
        seq_len, heads, dim = k.shape
        self._head_dim, self._num_heads = dim, heads
        self._device = k.device

        # Tier 1: fill sink first
        need = self.quant_config.sink_tokens - self.sink_len
        if need > 0:
            take = min(need, seq_len)
            sk = k[:take].to(torch.float16)
            sv = v[:take].to(torch.float16)
            self.sink_k = sk if self.sink_k is None else torch.cat([self.sink_k, sk], 0)
            self.sink_v = sv if self.sink_v is None else torch.cat([self.sink_v, sv], 0)
            self.sink_len += take
            k, v = k[take:], v[take:]
            seq_len -= take
            self.hbm_write_bytes += sk.numel() * 2 + sv.numel() * 2

        # Tier 3: stream into ring, sealing full groups
        for start in range(0, seq_len, self.group_size):
            chunk_k = k[start : start + self.group_size]
            chunk_v = v[start : start + self.group_size]
            for token in range(chunk_k.shape[0]):
                self._push_ring(chunk_k[token : token + 1], chunk_v[token : token + 1])
                if self.ring_len == self.group_size:
                    self.seal_page()

    def _push_ring(self, k: torch.Tensor, v: torch.Tensor) -> None:
        k16, v16 = k.to(torch.float16), v.to(torch.float16)
        if self.ring_k is None:
            self.ring_k = k16
            self.ring_v = v16
        else:
            assert self.ring_k is not None and self.ring_v is not None
            self.ring_k = torch.cat([self.ring_k, k16], 0)
            self.ring_v = torch.cat([self.ring_v, v16], 0)
        self.ring_len += k16.shape[0]
        self.hbm_write_bytes += k16.numel() * 2 + v16.numel() * 2

    # ------------------------------------------------------------------ sealing
    def seal_page(self) -> None:
        """Quantize the full ring into a sealed page with CoW dedup."""
        if self.ring_len == 0:
            return
        if self.ring_len < self.group_size:
            # Partial tail stays in ring (never seal incomplete blocks)
            return
        assert self.ring_k is not None and self.ring_v is not None

        k_bits = [self._k_bits_for(h) for h in range(self._num_heads)]
        v_bits = [self._v_bits_for(h) for h in range(self._num_heads)]
        k_payload, k_params = self._quantize_block(self.ring_k, k_bits, self.quant_config.k_scheme)
        v_payload, v_params = self._quantize_block(self.ring_v, v_bits, self.quant_config.v_scheme)
        prefix_hash = self._compute_prefix_hash(k_payload)
        ref = self._get_or_create_page(
            prefix_hash, k_payload, k_params, v_payload, v_params, k_bits, v_bits
        )
        self.page_table.append(ref.page_id)

        # Reset ring
        self.ring_k = None
        self.ring_v = None
        self.ring_len = 0

    def _quantize_block(
        self, block: torch.Tensor, bits: int | list[int], scheme: str
    ) -> tuple[torch.Tensor, list[QuantParams]]:
        """Quantize a (group_size, heads, dim) block per (head, dim) channel.

        ``bits`` is either one width for every head or a per-head list
        (ADR-009 mixed precision). Channels are padded to the widest packed
        length so the payload stays a single rectangular tensor; unpacking
        honors ``original_size`` so the padding is inert.
        """
        heads, dim = self._num_heads, self._head_dim
        packed_heads: list[torch.Tensor] = []
        params: list[QuantParams] = []
        for h in range(heads):
            head_bits = bits if isinstance(bits, int) else bits[h]
            for d in range(dim):
                channel = block[:, h, d]
                packed, qp = quantize_group(channel, head_bits, scheme)  # type: ignore[arg-type]
                packed_heads.append(packed)
                params.append(qp)
        max_len = max(p.shape[0] for p in packed_heads)
        padded = []
        for p in packed_heads:
            if p.shape[0] < max_len:
                pad = torch.zeros(max_len - p.shape[0], dtype=p.dtype, device=p.device)
                padded.append(torch.cat([p, pad]))
            else:
                padded.append(p)
        return torch.stack(padded), params  # (heads*dim, packed_len)

    def _compute_prefix_hash(self, page_data: torch.Tensor) -> int:
        """Content-addressable SHA256 hash of a page payload."""
        raw = bytes(page_data.flatten().tolist())
        digest = hashlib.sha256(raw).digest()
        return int.from_bytes(digest[:8], "big")

    def _get_or_create_page(
        self,
        prefix_hash: int,
        k_payload: torch.Tensor,
        k_params: list[QuantParams],
        v_payload: torch.Tensor,
        v_params: list[QuantParams],
        k_bits: list[int] | None = None,
        v_bits: list[int] | None = None,
    ) -> PageRef:
        """Return an existing physical page (incremented ref) or allocate a new one."""
        if prefix_hash in self._pages_by_hash:
            pid = self._pages_by_hash[prefix_hash]
            self._pages[pid].ref_count += 1
            return PageRef(
                page_id=pid, prefix_hash=prefix_hash, ref_count=self._pages[pid].ref_count
            )

        pid = self._next_page_id
        self._next_page_id += 1
        self._pages[pid] = PhysicalPage(
            page_id=pid,
            prefix_hash=prefix_hash,
            k_payload=k_payload,
            v_payload=v_payload,
            k_params=k_params,
            v_params=v_params,
            k_bits=list(k_bits or []),
            v_bits=list(v_bits or []),
            ref_count=1,
        )
        self._pages_by_hash[prefix_hash] = pid
        self.hbm_write_bytes += k_payload.numel() + v_payload.numel()
        return PageRef(page_id=pid, prefix_hash=prefix_hash, ref_count=1)

    # ------------------------------------------------------------------ CoW fork
    def fork_prefix(self) -> HybridPagedKVCache:
        """Fork a new cache sharing all sealed physical pages (zero duplication)."""
        child = HybridPagedKVCache(self.quant_config, self.cache_config, self.layout)
        child.group_size = self.group_size
        child._head_dim, child._num_heads = self._head_dim, self._num_heads
        child._device = self._device
        child.head_schedule = dict(self.head_schedule) if self.head_schedule is not None else None
        child.sink_k = self.sink_k
        child.sink_v = self.sink_v
        child.sink_len = self.sink_len
        child._pages = dict(self._pages)  # share PhysicalPage objects
        child._pages_by_hash = dict(self._pages_by_hash)
        child.page_table = list(self.page_table)
        child._next_page_id = self._next_page_id
        child.ring_k = self.ring_k
        child.ring_v = self.ring_v
        child.ring_len = self.ring_len
        for pid in child.page_table:
            self._pages[pid].ref_count += 1
        return child

    def copy_on_write(self, page_id: int) -> int:
        """Clone a shared page before an in-place write; returns the writable page id."""
        page = self._pages[page_id]
        if page.ref_count <= 1:
            return page_id
        page.ref_count -= 1
        new_pid = self._next_page_id
        self._next_page_id += 1
        clone = PhysicalPage(
            page_id=new_pid,
            prefix_hash=page.prefix_hash,
            k_payload=page.k_payload.clone(),
            v_payload=page.v_payload.clone(),
            k_params=list(page.k_params),
            v_params=list(page.v_params),
            k_bits=list(page.k_bits),
            v_bits=list(page.v_bits),
            ref_count=1,
        )
        self._pages[new_pid] = clone
        self.page_table = [new_pid if p == page_id else p for p in self.page_table]
        return new_pid

    # ------------------------------------------------------------------ retrieval
    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize FP16 K/V from Tiers 1 + 2 + 3."""
        pieces_k: list[torch.Tensor] = []
        pieces_v: list[torch.Tensor] = []
        if self.sink_k is not None:
            assert self.sink_v is not None
            pieces_k.append(self.sink_k)
            pieces_v.append(self.sink_v)
            self.hbm_read_bytes += self.sink_k.numel() * 2 + self.sink_v.numel() * 2

        for pid in self.page_table:
            page = self._pages[pid]
            k = self._dequantize_payload(
                page.k_payload,
                page.k_params,
                page.k_bits or self.quant_config.k_bits,
                self.quant_config.k_scheme,
            )
            v = self._dequantize_payload(
                page.v_payload,
                page.v_params,
                page.v_bits or self.quant_config.v_bits,
                self.quant_config.v_scheme,
            )
            pieces_k.append(k)
            pieces_v.append(v)
            self.hbm_read_bytes += page.k_payload.numel() + page.v_payload.numel()

        if self.ring_k is not None:
            assert self.ring_v is not None
            pieces_k.append(self.ring_k)
            pieces_v.append(self.ring_v)
            self.hbm_read_bytes += self.ring_k.numel() * 2 + self.ring_v.numel() * 2

        if not pieces_k:
            empty = torch.zeros(
                0, self._num_heads, self._head_dim, dtype=torch.float16, device=self._device
            )
            return empty, empty
        return torch.cat(pieces_k, 0), torch.cat(pieces_v, 0)

    def _dequantize_payload(
        self, payload: torch.Tensor, params: list[QuantParams], bits: int | list[int], scheme: str
    ) -> torch.Tensor:
        """Dequantize one payload; ``bits`` may be per-head (ADR-009).

        Channel ``i`` of the payload corresponds to head ``i // head_dim``,
        matching ``_quantize_block``'s (head, dim) append order.
        """
        heads_dim = self._num_heads * self._head_dim
        channels = []
        for i in range(heads_dim):
            packed = payload[i]
            qp = params[i]
            head = i // self._head_dim if self._head_dim > 0 else 0
            head_bits = bits if isinstance(bits, int) else bits[head]
            channels.append(dequantize_group(packed, qp, head_bits, scheme))  # type: ignore[arg-type]
        # channels: (heads*dim, group_size) -> transpose -> (group_size, heads*dim)
        block = torch.stack(channels, 0).t()
        return block.reshape(self.group_size, self._num_heads, self._head_dim).to(torch.float16)

    def get(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.materialize()
        return k[indices], v[indices]

    @property
    def total_tokens(self) -> int:
        return self.sink_len + len(self.page_table) * self.group_size + self.ring_len

    @property
    def vram_bytes(self) -> int:
        """Actual physical bytes (shared pages counted once)."""
        sink = 0
        if self.sink_k is not None:
            assert self.sink_v is not None
            sink = self.sink_k.numel() * 2 + self.sink_v.numel() * 2
        pages = sum(p.k_payload.numel() + p.v_payload.numel() for p in self._pages.values())
        ring = 0
        if self.ring_k is not None:
            assert self.ring_v is not None
            ring = self.ring_k.numel() * 2 + self.ring_v.numel() * 2
        return sink + pages + ring

    @property
    def shared_pages(self) -> int:
        return sum(1 for p in self._pages.values() if p.ref_count > 1)
