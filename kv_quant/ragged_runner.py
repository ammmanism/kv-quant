"""Ragged-batch PagedAttention runner with Copy-on-Write prefix sharing.

`RaggedPagedBatchRunner` orchestrates many concurrent agent sessions over the
existing :class:`kv_quant.cache.HybridPagedKVCache` without re-implementing any
cache, layout, or quantization logic:

- A **root** session seals a shared system/repository prefix into Tier-2 pages.
- Every **forked** session shares those physical pages (``ref_count = N``) and
  allocates private pages only for its divergent reasoning turns and FP16 ring
  tail -- the Copy-on-Write behaviour the cache already provides via
  ``fork_prefix`` / ``copy_on_write``.
- ``context_lens`` exposes the ragged ``[B]`` tensor the decode kernel consumes,
  so one batched launch serves heterogeneous sequence lengths.

Prefix identity is chained and deterministic:
``chain_prefix_hash(parent_hash, token_block, config_id)`` hashes
``(parent_hash, block bytes, config_id)`` with SHA256. This is the runner's
*logical* dedup keyspace; physical page identity remains the cache's
content-addressed payload hash (Invariant 3), which the runner never bypasses.

Capacity is enforced against ``max_physical_pages`` **before** an append seals
new pages. The estimate is conservative: it assumes every new block allocates,
even when content dedup would have shared one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from kv_quant.cache import HybridPagedKVCache, PhysicalPage
from kv_quant.config import CacheConfig, QuantConfig


class CapacityExceededError(RuntimeError):
    """Raised when an append would exceed the physical page budget."""


@dataclass
class SessionState:
    """One concurrent agent session over the shared page pool."""

    session_id: str
    cache: HybridPagedKVCache
    shared_prefix_blocks: int
    tokens: int = 0
    completed: bool = False


def chain_prefix_hash(parent_hash: int, token_block: torch.Tensor, config_id: str) -> int:
    """Deterministic 64-bit hash of ``(parent_hash, block, config_id)``.

    Args:
        parent_hash: Hash of the preceding block (``0`` for the first block).
        token_block: Token ids of this 64-token block (any integer dtype).
        config_id: Quantization configuration identifier.

    Returns:
        The first 64 bits of SHA256 over the canonical serialization.
    """
    payload = b"|".join(
        [
            str(int(parent_hash)).encode("ascii"),
            config_id.encode("utf-8"),
            token_block.detach().to(torch.int64).cpu().contiguous().numpy().tobytes(),
        ]
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class RaggedPagedBatchRunner:
    """Global physical page pool over heterogeneous concurrent sessions."""

    def __init__(
        self,
        quant_config: QuantConfig,
        cache_config: CacheConfig,
        max_physical_pages: int = 4096,
        group_size: int = 64,
    ):
        if max_physical_pages <= 0:
            raise ValueError("max_physical_pages must be positive")
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.max_physical_pages = max_physical_pages
        self.group_size = group_size
        self.sessions: dict[str, SessionState] = {}
        self.freed_pages = 0

    # ------------------------------------------------------------- registration
    def register_root(
        self, session_id: str, prefix_k: torch.Tensor, prefix_v: torch.Tensor
    ) -> SessionState:
        """Create the root session holding the shared prefix pages."""
        if session_id in self.sessions:
            raise ValueError(f"duplicate session_id: {session_id}")
        cache = HybridPagedKVCache(self.quant_config, self.cache_config)
        cache.group_size = self.group_size
        state = SessionState(session_id=session_id, cache=cache, shared_prefix_blocks=0)
        self.sessions[session_id] = state
        if prefix_k.numel():
            self.append(session_id, prefix_k, prefix_v)
        state.shared_prefix_blocks = state.tokens // self.group_size
        return state

    def register_session(self, session_id: str, parent_id: str | None = None) -> SessionState:
        """Register a session, forking ``parent_id``'s pages when given.

        Forking shares every sealed physical page (``ref_count += 1`` per
        page) and copies the parent's sink/ring state; subsequent appends land
        in the child's private dicts only.
        """
        if session_id in self.sessions:
            raise ValueError(f"duplicate session_id: {session_id}")
        if parent_id is None:
            cache = HybridPagedKVCache(self.quant_config, self.cache_config)
            cache.group_size = self.group_size
            state = SessionState(session_id=session_id, cache=cache, shared_prefix_blocks=0)
        else:
            parent = self._live_session(parent_id)
            state = SessionState(
                session_id=session_id,
                cache=parent.cache.fork_prefix(),
                shared_prefix_blocks=parent.shared_prefix_blocks,
                tokens=parent.tokens,
            )
        self.sessions[session_id] = state
        return state

    # ------------------------------------------------------------------ runtime
    def append(self, session_id: str, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append ragged tokens to one session under the page budget."""
        state = self._live_session(session_id)
        new_tokens = k.shape[0] if k.dim() >= 2 else 1
        blocks_before = state.tokens // self.group_size
        blocks_after = (state.tokens + new_tokens) // self.group_size
        needed = max(0, blocks_after - blocks_before)
        if self.live_pages() + needed > self.max_physical_pages:
            raise CapacityExceededError(
                f"session {session_id!r} needs {needed} new page(s); "
                f"{self.live_pages()}/{self.max_physical_pages} physical pages live"
            )
        state.cache.append(k, v)
        state.tokens += new_tokens

    def complete(self, session_id: str) -> None:
        """Release a session's references; free pages whose count hits zero."""
        state = self._live_session(session_id)
        cache = state.cache
        for pid in list(cache.page_table):
            page = cache._pages.get(pid)
            if page is None:
                continue
            page.ref_count -= 1
            if page.ref_count <= 0:
                cache._pages.pop(pid, None)
                for key, value in list(cache._pages_by_hash.items()):
                    if value == pid:
                        cache._pages_by_hash.pop(key, None)
                self.freed_pages += 1
        state.completed = True
        state.tokens = 0
        cache.page_table.clear()

    def context_lens(self) -> torch.Tensor:
        """Ragged ``[B]`` context lengths of live sessions, registration order."""
        return torch.tensor(
            [s.tokens for s in self.sessions.values() if not s.completed], dtype=torch.int64
        )

    # ------------------------------------------------------------------- stats
    def _live_page_objects(self) -> list[PhysicalPage]:
        seen: dict[int, PhysicalPage] = {}
        for state in self.sessions.values():
            if state.completed:
                continue
            for page in state.cache._pages.values():
                if page.ref_count > 0:
                    seen[id(page)] = page
        return list(seen.values())

    def live_pages(self) -> int:
        return len(self._live_page_objects())

    def pool_stats(self) -> dict[str, int]:
        """Pool accounting: live/shared/private pages and freed counter."""
        pages = self._live_page_objects()
        shared = sum(1 for p in pages if p.ref_count > 1)
        return {
            "live_pages": len(pages),
            "shared_pages": shared,
            "private_pages": len(pages) - shared,
            "freed_pages": self.freed_pages,
            "sessions": sum(1 for s in self.sessions.values() if not s.completed),
        }

    # ------------------------------------------------------------------ helpers
    def _live_session(self, session_id: str) -> SessionState:
        try:
            state = self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown session: {session_id}") from exc
        if state.completed:
            raise RuntimeError(f"session {session_id!r} is already completed")
        return state
