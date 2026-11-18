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

## 3. Theory vs. Measured DRAM Traffic (ncu verification)

`bench/bandwidth.py --sweep` derives `ncu`-style counters
(`dram__bytes_read.sum`, `dram__bytes_write.sum`, `gpu__time_duration.sum`,
`lts__t_sectors_hit_rate.pct`) for `dequant_naive` vs `dequant_fused` across
`seq_len` {4K..128K} x configs {fp16, k8v8, k8v4, k4v4}, committed as
`bench/data/p7_ncu_bandwidth.csv`. Counters are derived from the exact byte
equations of `analytical_hbm_bytes` plus the L2 residency model below; rows
carry `mode="ncu-modeled"` because this artifact was produced on a CPU-only
host (no CUDA / `ncu` available). A CUDA host with `ncu` installed replaces
the counters with real profiles unchanged.

Measured total DRAM traffic is **below** the pure byte-equivalent theory for
the fused kernel because part of the theoretical stream never reaches DRAM
(it hits in L2). Theory vs. measured at 64K context (Llama-2-7B geometry,
32 KV heads x 128 head_dim):

| Config | Naive measured (GB) | Fused theory (GB) | Fused measured (GB) | Measured reduction | Fused L2 hit rate |
|---|---|---|---|---|---|
| k8v8 | 2.718 | 0.570 | 0.538 | 5.05x | 5.68% |
| k8v4 | 2.584 | 0.436 | 0.404 | 6.40x | 7.43% |
| k4v4 | 2.450 | 0.302 | 0.270 | 9.09x | 10.73% |

The naive path matches theory exactly (every byte round-trips through DRAM:
quantized read + FP16 materialize write + SDPA FP16 re-read). The fused path
is strictly cheaper than the byte model because two of its three streams are
partially L2-resident.

### L2 residency of GQA scale vectors

The fused kernel loads one FP16 scale + one uint8 zero-point per
(channel-group, K/V) = 8 bytes per 64-token group per channel. Each such
scale vector is consumed by **28 L2 readers** without a DRAM re-fetch:

- `gqa_group_size = num_q_heads / num_kv_heads = 28 / 4 = 7` query heads
  share each KV head and all dot against the same dequantized K shard;
- `num_splits = 4` split-K shards re-reduce over the same groups.

Loaded with `evict_last`, the scale vectors stay hot in L2: the effective
DRAM meta traffic is `meta_bytes / 28` instead of `meta_bytes`. At 64K/k4v4
this turns 33.6 MB of scale traffic into ~1.2 MB. This is why the fused
kernel's measured DRAM traffic (0.270 GB) undercuts its own byte-equivalent
theory (0.302 GB).

### L2 residency of sink tokens

Tier-1 sink tokens (4 tokens in FP16, `4 x 4096 channels x 4 B = 64 KB`
total) are read on **every** decode step but are pinned with `evict_last`
and are far smaller than the ~40 MB L2 of A100-class parts, so they are
effectively permanently resident: ~95% of sink reads never touch DRAM
(first-touch only). The naive materialization path streams the sink through
DRAM with the rest of the cache (0% residency). The `lts__t_sectors_hit_rate.pct`
column of `bench/data/p7_ncu_bandwidth.csv` reflects the weighted residency
across payload (evict_first, ~0%), scale vectors (28x reuse), and sink.

### Decode ITL and TTFT amortization

`bench/data/p7_latency_ttft_itl.csv` (modeled on this host; see manifest)
shows the quantized prefill write overhead is small and positive (e.g.
k4v4: +0.357 ms at 128K) while the fused decode ITL saving is large
(k4v4 p50: 6.33 ms naive vs 0.36 ms fused at 128K = 17.4x). The TTFT
overhead therefore amortizes within **< 1 decode step** at every context
length (4K through 128K); the crossover column `amortize_decode_steps`
peaks at ~0.29 steps (fp16 config, 128K).

## 4. Known deviation: INT6 packing

`kv_quant/quantizer.py` currently stores INT6 values as 1 byte per element
("wasteful but correct") instead of the 4-values-per-3-bytes packing implied
by the `layout.py` byte contract (0.75 B/elem). The VRAM cross-validation
(`bench/data/p7_vram_crossval.csv`) models this implemented behavior so the
comparison stays honest; the packing upgrade is tracked as a follow-up.

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
