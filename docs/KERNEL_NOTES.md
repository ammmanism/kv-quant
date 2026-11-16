# Fused Split-K GQA Decode Kernel — Design Notes

## 1. Register Unpacking Pipeline

The fused kernel never materializes FP16 K/V to HBM. Instead:

1. **Load packed words**: `tl.load(K_PAGES + packed_index, eviction_policy="evict_first")`
   reads `uint8` payload bytes directly from Tier-2 pages via `block_table`.
2. **In-register INT4 unpack** (matching `kv_quant/layout.py` exactly):
   ```python
   lo = packed & INT4_LOW_MASK  # 0x0F
   hi = (packed >> INT4_HIGH_SHIFT) & INT4_LOW_MASK  # shift 4
   decoded = tl.interleave(lo, hi)
   ```
   `check_layout_drift.py` pins `INT4_LOW_MASK=0x0F`, `INT4_HIGH_SHIFT=4`, so
   any layout drift fails CI before the kernel can silently mis-decode.
3. **Dequantize in registers**: `decoded = (decoded - zp) * scale` using the
   per-group FP16 scale and uint8 zero-point loaded with `evict_last`.
4. **Fused dot**: `tl.dot(q, tl.trans(k))` against *all* `gqa_group_size`
   query heads that share this KV head, broadcast in SRAM.

### GQA Broadcast & Split-K Grid

Grid is `(batch * num_kv_heads, num_splits)`. Each program loads one KV-head
page shard once and dots it against `num_q_heads // num_kv_heads` query heads.
`num_splits` shards the key dimension; partial online-softmax statistics
`m_i`, `l_i`, `acc` are reduced by `_splitk_reduce_kernel`.

### 3-Tier Online Softmax

- **Tier 1 (sink)**: loaded with `evict_last` — always retained in L2.
- **Tier 2 (paged pool)**: streamed with `evict_first` — single-use.
- **Tier 3 (ring tail)**: concatenated into the shard, using the same online
  softmax recurrence, so the three tiers combine without a separate pass.

The recurrence per block:
```
m_new = max(m_i, max(scores))
alpha = exp(m_i - m_new)
l_i   = l_i * alpha + sum(exp(scores - m_new))
acc   = acc * alpha + dot(p, k)
```

## 2. HBM Traffic Reduction Derivation (8.11×)

Reference configuration (Llama-2-7B KV geometry):

| Quantity | Value |
|---|---|
| `head_dim` | 128 |
| `num_kv_heads` | 32 |
| channels (`head_dim × kv_heads`) | 4096 |
| tokens | 2048 |
| quantization | INT4 K/V, `group_size = 64` |

**Per-token element counts**

- FP16 materialized K+V: `channels × 2 (K,V) × 2 bytes = 4096 × 4 = 16384 B`
- INT4 K+V payload: `channels × 2 × 0.5 = 4096 B`
- K+V metadata per channel-group: `8 B` (K scale fp16 + K zp u8 + V scale fp16
  + V zp u8 + 2 B alignment). With 64 tokens/group this is
  `4096 × 8 / 64 = 512 B` per token.
- **Quantized bytes per token**: `4096 + 512 = 4608 B`

**Traffic**

```
naive  = read_quantized + write_fp16 + sdpa_read_fp16
       = 4608 + 16384 + 16384 = 37376 B/token
fused  = read_quantized (once, registers) = 4608 B/token
reduction = 37376 / 4608 = 8.11x
```

This is the value emitted by `bench/bandwidth.py --bits 4 --group-size 64`
and recorded in `bench/data/p3_hbm_traffic_model.csv`.

> The 8.11× figure assumes the FP16 intermediate is written *and* re-read.
> For 16-bit inputs the relative gain shrinks (no quantization payload reduction)
> but the fused path still removes the materialization round-trip.

## 3. GPU Architecture Compatibility Matrix

| Feature | sm80 (A100) | sm86 (RTX 3090) | sm89 (4090/L40S) | sm90 (H100) | sm100 (B200) |
|---|:---:|:---:|:---:|:---:|:---:|
| `int8_unpack` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `int4_nibble_unpack` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `fp8_e4m3_tensorcore` | ❌ | ❌ | ✅ | ✅ | ✅ |
| `l2_eviction_hints` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `splitk_gqa_broadcast` | ✅ | ✅ | ✅ | ✅ | ✅ |

FP8 MMA requires sm89+; all other features are available on Ampere onward.
The full machine-readable matrix is committed at
`bench/data/arch_compat_matrix.csv`.

## 4. CPU Reference Path

`fused_splitk_attention()` dispatches to `fused_splitk_attention_cpu()` whenever
CUDA is unavailable. The CPU path performs the same dequantize → GQA broadcast
→ online-softmax math in PyTorch, providing a single source of truth for
correctness on CPU-only CI. GPU hosts use the Triton kernels above.
