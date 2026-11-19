"""Tests for 3-tier hybrid cache, CoW, profiler and wrapper."""

import pytest
import torch
from kv_quant.cache import HybridPagedKVCache
from kv_quant.config import CacheConfig, QuantConfig
from kv_quant.kernels.dequant_naive import materialize_kv_naive
from kv_quant.profiler import HeadProfile, compile_head_bit_schedule, profile_heads
from kv_quant.wrapper import KVQuantWrapper


def make_cache(bits: int = 4) -> HybridPagedKVCache:
    cfg = QuantConfig(k_bits=bits, v_bits=bits, group_size=64, sink_tokens=4)
    return HybridPagedKVCache(cfg, CacheConfig())


def test_append_tier_partition():
    cache = make_cache()
    k = torch.randn(140, 2, 8, dtype=torch.float16)
    v = torch.randn(140, 2, 8, dtype=torch.float16)
    cache.append(k, v)
    assert cache.sink_len == 4
    assert len(cache.page_table) == 2  # 136 tokens -> 2 full 64-blocks
    assert cache.ring_len == 8
    assert cache.total_tokens == 140


def test_materialize_shape_and_error():
    cache = make_cache(bits=8)
    k = torch.randn(140, 2, 8, dtype=torch.float16)
    v = torch.randn(140, 2, 8, dtype=torch.float16)
    cache.append(k, v)
    kk, vv = cache.materialize()
    assert kk.shape == k.shape
    assert vv.shape == v.shape
    err = (kk.float() - k.float()).abs().max().item()
    assert err < 0.1  # 8-bit should be quite accurate


def test_cow_fork_zero_duplication():
    cache = make_cache()
    k = torch.randn(200, 2, 8, dtype=torch.float16)
    cache.append(k, k)
    fork = cache.fork_prefix()
    assert fork.page_table == cache.page_table
    assert fork.shared_pages == len(cache.page_table)
    assert fork.vram_bytes == cache.vram_bytes  # no duplication on fork
    assert all(page.ref_count == 2 for page in fork._pages.values())


def test_copy_on_write_clones_shared_page():
    cache = make_cache()
    cache.append(
        torch.randn(140, 2, 8, dtype=torch.float16), torch.randn(140, 2, 8, dtype=torch.float16)
    )
    fork = cache.fork_prefix()
    old_pid = fork.page_table[0]
    new_pid = fork.copy_on_write(old_pid)
    assert new_pid != old_pid
    assert fork.page_table[0] == new_pid
    assert fork._pages[new_pid].ref_count == 1


def test_content_addressable_dedup():
    cfg = QuantConfig(k_bits=4, v_bits=4, group_size=64, sink_tokens=0)
    cache = HybridPagedKVCache(cfg, CacheConfig())
    # 128 tokens -> 2 sealed pages
    k = torch.randn(128, 1, 8, dtype=torch.float16)
    v = torch.randn(128, 1, 8, dtype=torch.float16)
    cache.append(k, v)
    assert len(cache.page_table) == 2
    cache.append(k, v)  # identical prefix -> 2 more logical pages, same physical
    assert len(cache.page_table) == 4
    # Identical payloads share physical pages
    assert len(cache._pages) == 2
    assert all(page.ref_count == 2 for page in cache._pages.values())


def test_naive_materialize_hbm_stats():
    cache = make_cache()
    k = torch.randn(140, 2, 8, dtype=torch.float16)
    cache.append(k, k)
    kk, vv, stats = materialize_kv_naive(
        cache.sink_k,
        cache.sink_v,
        cache._pages,
        cache.page_table,
        cache.ring_k,
        cache.ring_v,
        cache.quant_config.k_bits,
        cache.quant_config.v_bits,
        cache.quant_config.k_scheme,
        cache.quant_config.v_scheme,
        num_heads=2,
        head_dim=8,
        group_size=64,
    )
    assert kk.shape == k.shape
    assert stats.pages_dequantized == 2
    assert stats.sink_tokens == 4
    assert stats.ring_tokens == 8
    assert stats.hbm_read_bytes > 0
    assert stats.hbm_write_bytes > 0


def test_profile_heads_three_metrics():
    torch.manual_seed(1)
    p = torch.softmax(torch.randn(16, 256), dim=-1)
    profiles = profile_heads(p.unsqueeze(0), layer_idx=0)
    assert len(profiles) == 1
    hp = profiles[0]
    assert 0 <= hp.distant_attn_mass <= 1
    assert hp.attn_entropy > 0
    assert hp.kl_sensitivity >= 0


def test_compile_head_bit_schedule():
    profiles = []
    for layer in range(4):
        for head in range(10):
            profiles.append(
                HeadProfile(
                    layer_idx=layer,
                    kv_head_idx=head,
                    distant_attn_mass=float(head) / 10,
                    attn_entropy=2.0 + head * 0.1,
                    kl_sensitivity=0.01 * head,
                )
            )
    schedule = compile_head_bit_schedule(profiles)
    assert len(schedule) == 40
    classes = {s["head_class"] for s in schedule}
    assert classes == {"K8V4", "K4V4_Hadamard", "K2_QJL_V2"}
    retrieval = [s for s in schedule if s["head_class"] == "K8V4"]
    assert all(s["assigned_k_bits"] == 8 and s["assigned_v_bits"] == 4 for s in retrieval)
    streaming = [s for s in schedule if s["head_class"] == "K2_QJL_V2"]
    assert all(s["assigned_k_bits"] == 2 for s in streaming)


def test_wrapper_zero_reload_switch():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8)
            self.k_proj = torch.nn.Linear(8, 8)
            self.v_proj = torch.nn.Linear(8, 8)

        def forward(self, input_ids, attention_mask=None):
            x = input_ids.float() if input_ids.dtype != torch.float32 else input_ids
            return type("Out", (), {"logits": self.q_proj(x)})()

    model = TinyModel()
    wrapper = KVQuantWrapper(model, QuantConfig(), CacheConfig())
    with wrapper:
        out = wrapper.forward(torch.randint(0, 16, (2, 4, 8)))
        assert out.logits.shape == (2, 4, 8)
    new_cfg = QuantConfig(k_bits=4, v_bits=4, use_hadamard=True)
    wrapper.set_kv_config(new_cfg)
    assert wrapper.quant_config.k_bits == 4
    assert wrapper.cache is None


def test_head_schedule_per_head_precision():
    """ADR-009: retrieval head at K8 must dequantize more accurately than a K4 head."""
    cfg = QuantConfig(k_bits=4, v_bits=4, group_size=64, sink_tokens=0)
    cache = HybridPagedKVCache(cfg, CacheConfig())
    cache.set_head_schedule({(0, 0): (8, 4), (0, 1): (4, 4)}, layer_idx=0)
    k = torch.randn(128, 2, 8, dtype=torch.float16)
    v = torch.randn(128, 2, 8, dtype=torch.float16)
    cache.append(k, v)
    kk, vv = cache.materialize()
    assert kk.shape == k.shape
    err_head0 = (kk[:, 0].float() - k[:, 0].float()).pow(2).mean().item()
    err_head1 = (kk[:, 1].float() - k[:, 1].float()).pow(2).mean().item()
    assert err_head0 < err_head1, (err_head0, err_head1)
    # The sealed page records the per-head widths it was quantized with.
    page = cache._pages[cache.page_table[0]]
    assert page.k_bits == [8, 4]


def test_head_schedule_validation():
    cache = make_cache()
    with pytest.raises(ValueError):
        cache.set_head_schedule({(0, 0): (8, 4), (1, 1): (4, 4)})  # multi-layer, no layer_idx
    with pytest.raises(ValueError):
        cache.set_head_schedule({(3, 0): (8, 4)}, layer_idx=0)  # no heads for this layer


def test_head_schedule_survives_fork_and_cow():
    cfg = QuantConfig(k_bits=4, v_bits=4, group_size=64, sink_tokens=0)
    cache = HybridPagedKVCache(cfg, CacheConfig())
    cache.set_head_schedule({(0, 0): (8, 4), (0, 1): (4, 4)})
    cache.append(
        torch.randn(128, 2, 8, dtype=torch.float16), torch.randn(128, 2, 8, dtype=torch.float16)
    )
    fork = cache.fork_prefix()
    assert fork.head_schedule == cache.head_schedule
    clone_id = fork.copy_on_write(fork.page_table[0])
    assert fork._pages[clone_id].k_bits == [8, 4]


def test_head_schedule_missing_head_uses_config_default():
    """Heads absent from the schedule fall back to the config bit-width."""
    cfg = QuantConfig(k_bits=8, v_bits=8, group_size=64, sink_tokens=0)
    cache = HybridPagedKVCache(cfg, CacheConfig())
    cache.set_head_schedule({(0, 0): (4, 4)})  # head 1 unspecified -> K8V8
    cache.append(
        torch.randn(128, 2, 8, dtype=torch.float16), torch.randn(128, 2, 8, dtype=torch.float16)
    )
    kk, _ = cache.materialize()
    assert cache._pages[cache.page_table[0]].k_bits == [4, 8]
