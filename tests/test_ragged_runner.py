"""Tests for the ragged-batch paged runner and CoW prefix sharing (ADR-011)."""

from __future__ import annotations

import pytest
import torch
from kv_quant.config import CacheConfig, QuantConfig
from kv_quant.kernels.dequant_fused import fused_splitk_attention_cpu
from kv_quant.ragged_runner import (
    CapacityExceededError,
    RaggedPagedBatchRunner,
    chain_prefix_hash,
)

from tests.test_fused_kernel import HEAD_DIM, NUM_KV_HEADS, build_pages


def make_runner(max_pages: int = 4096, sink_tokens: int = 4) -> RaggedPagedBatchRunner:
    cfg = QuantConfig(k_bits=4, v_bits=4, group_size=64, sink_tokens=sink_tokens)
    return RaggedPagedBatchRunner(cfg, CacheConfig(), max_physical_pages=max_pages, group_size=64)


def kv(tokens: int) -> torch.Tensor:
    return torch.randn(tokens, 4, 32, dtype=torch.float16)


class TestCopyOnWrite:
    def test_shared_prefix_refcount(self) -> None:
        runner = make_runner()
        runner.register_root("root", kv(256), kv(256))
        root_pages = list(runner.sessions["root"].cache._pages.values())
        # 4 sinks + 252 streamed -> 3 sealed pages of 64, 60-token ring tail.
        assert len(root_pages) == 3
        for i in range(15):
            runner.register_session(f"agent{i}", parent_id="root")
        assert all(p.ref_count == 16 for p in root_pages), "root + 15 forks must share pages"

    def test_divergence_allocates_private_pages_only(self) -> None:
        runner = make_runner()
        runner.register_root("root", kv(256), kv(256))
        for i in range(3):
            runner.register_session(f"agent{i}", parent_id="root")
        shared = list(runner.sessions["root"].cache._pages.values())
        runner.append("agent0", kv(256), kv(256))
        assert all(p.ref_count == 4 for p in shared), "shared pages must not be re-refcounted"
        stats = runner.pool_stats()
        assert stats["shared_pages"] == 3
        assert stats["private_pages"] >= 3, "agent0's divergent seals are private"
        # The shared prefix payload is unchanged (CoW, not in-place).
        root_after = list(runner.sessions["root"].cache._pages.values())
        assert [id(p) for p in root_after] == [id(p) for p in shared]

    def test_capacity_budget_enforced(self) -> None:
        runner = make_runner(max_pages=4)
        runner.register_root("root", kv(256), kv(256))  # 3 pages
        runner.register_session("a", parent_id="root")
        with pytest.raises(CapacityExceededError):
            runner.append("a", kv(128), kv(128))  # +2 private pages -> 5 > 4
        assert runner.live_pages() == 3, "a rejected append must not allocate"


class TestLeaks:
    def test_zero_leaks_after_500_completions(self) -> None:
        runner = make_runner(sink_tokens=0)
        runner.register_root("root", kv(64), kv(64))  # 1 shared page
        for i in range(500):
            sid = f"req{i}"
            runner.register_session(sid, parent_id="root")
            runner.append(sid, kv(64), kv(64))  # 1 private page
            runner.complete(sid)
        assert runner.pool_stats()["shared_pages"] == 0, "all forks released their refs"
        assert runner.live_pages() == 1, "only the root's page may remain"
        assert runner.pool_stats()["freed_pages"] == 500, "every private page was freed"
        runner.complete("root")
        stats = runner.pool_stats()
        assert stats["live_pages"] == 0
        assert stats["freed_pages"] == 501
        assert stats["sessions"] == 0

    def test_completed_session_rejects_further_use(self) -> None:
        runner = make_runner()
        runner.register_root("root", kv(64), kv(64))
        runner.complete("root")
        with pytest.raises(RuntimeError):
            runner.append("root", kv(64), kv(64))


class TestRaggedParity:
    def test_batch16_ragged_equals_batch1(self) -> None:
        """One batched ragged launch must equal 16 individual launches."""
        torch.manual_seed(7)
        seq = 600
        group = 64
        q_len = 4
        gqa = 2
        k = torch.randn(seq, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
        v = torch.randn(seq, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
        k_pages, v_pages, kp, vp, bt1 = build_pages(k, v, group, 4, 4, "sym", "asym")

        batch = 16
        lens = [600, 64, 0, 128, 512, 1, 65, 256, 599, 63, 320, 0, 128, 64, 512, 200]
        q = torch.randn(batch, NUM_KV_HEADS * gqa, q_len, HEAD_DIM, dtype=torch.float16)
        kv_len = torch.tensor(lens, dtype=torch.int64)
        bt = bt1.repeat(batch, 1)

        batched = fused_splitk_attention_cpu(
            q,
            k_pages,
            v_pages,
            kp,
            vp,
            bt,
            kv_len,
            num_kv_heads=NUM_KV_HEADS,
            k_bits=4,
            v_bits=4,
            k_scheme="sym",
            v_scheme="asym",
            group_size=group,
        )
        singles = [
            fused_splitk_attention_cpu(
                q[b : b + 1],
                k_pages,
                v_pages,
                kp,
                vp,
                bt1,
                torch.tensor([lens[b]], dtype=torch.int64),
                num_kv_heads=NUM_KV_HEADS,
                k_bits=4,
                v_bits=4,
                k_scheme="sym",
                v_scheme="asym",
                group_size=group,
            )
            for b in range(batch)
        ]
        stacked = torch.cat(singles, dim=0)
        assert torch.isfinite(batched).all(), "ragged batch must not produce NaN/Inf"
        assert torch.allclose(batched, stacked, atol=1e-3), (
            f"maxdiff={(batched.float() - stacked.float()).abs().max().item()}"
        )
        # Degenerate (zero-length) items must be exactly zero, not NaN.
        assert torch.equal(batched[2], torch.zeros_like(batched[2]))

    def test_context_lens_tensor_matches_sessions(self) -> None:
        runner = make_runner()
        runner.register_root("root", kv(128), kv(128))
        runner.register_session("short", parent_id="root")
        runner.register_session("long", parent_id="root")
        runner.append("long", kv(192), kv(192))
        assert runner.context_lens().tolist() == [128, 128, 320]


class TestPrefixHash:
    def test_deterministic_and_sensitive(self) -> None:
        block = torch.arange(64)
        a = chain_prefix_hash(0, block, "k4v4")
        assert a == chain_prefix_hash(0, block, "k4v4")
        assert a != chain_prefix_hash(1, block, "k4v4")
        assert a != chain_prefix_hash(0, block + 1, "k4v4")
        assert a != chain_prefix_hash(0, block, "k8v4")
