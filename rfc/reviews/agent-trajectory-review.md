# Adversarial Review — v1.5.0-agent-bench

Scope: `tasks/agent_trajectory.py`, `scripts/generate_phase14_artifacts.py`,
`tests/test_agent_trajectory.py`, `bench/data/p14_agent_trajectory_drift.csv`,
`docs/AGENT_FAILURE_MODES.md`, `docs/DECISIONS.md` (ADR-012),
`docs/figures/fig5_agent_amnesia_vs_syntax.png`.

Reviewer stance: assume the harness reports what its author wanted it to report,
and try to find the mechanism that makes that inevitable regardless of the data.

The first ten findings are defects in the harness itself, found by measuring the
fixture rather than reading it. Each would have produced a defensible-looking
figure with an indefensible cause.

## Findings

- [x] **Finding**: The trajectory script `target_of_turn = (t * 5) % 24` yields 20 *distinct* files for `t < 20` (gcd(5,24)=1), so the agent never revisits a file. K-amnesia is defined as re-reading a file already in context, so the metric was structurally zero and its reported zero would have been an artifact of the script, not a property of the quantizer. The docstring simultaneously claimed "early files recur", which was false.
- **Resolution**: RESOLVED - replaced with a 20-turn ReAct script containing six explicit backtracking revisits (file 0 at turns 0/3/8, file 5 at 1/5, file 10 at 2/7, file 2 at 16/19), and added `test_script_revisits_files`, which asserts `len(set(IDEAL_SCRIPT)) < TURNS`.

- [x] **Finding**: `TOKEN_JITTER = 0.05` was applied directly to `torch.randn(N, HEAD_DIM)`. Since `||randn(64)|| ~ 8`, each sampled token key sat ~0.4 away from its own file key — larger than the ~0.045 cosine margin that separates a file from its nearest sibling. The unquantized argmax was therefore itself a lottery, and the FP16 control reported 0.63% "amnesia" (1 incident in 160 turn-records) that could only be sampling noise.
- **Resolution**: RESOLVED - `TOKEN_JITTER` is now the L2 norm of the per-token offset (`jitter = TOKEN_JITTER / sqrt(HEAD_DIM)` before scaling `randn`), giving cos ~0.9987 between a token and its file key. FP16 now reports exactly 0, and `test_fp16_key_roundtrip_is_lossless` plus `test_fp16_payload_decodes_cleanly` pin that baseline.

- [x] **Finding**: `_key_pool` sampled cached files uniformly, so when many files were cached the target could be absent from the pool entirely. Retrieval then necessarily "failed", and that failure was charged to quantization. This is the same class of error as the jitter bug and was masked by it in the first sweep.
- **Resolution**: RESOLVED - the pool now reserves a target slot unconditionally, and `test_key_pool_always_represents_the_target` asserts the target is present for every turn of the script.

- [x] **Finding**: The first fix over-corrected: `TARGET_KEYS = 32` gave the target 32 draws against roughly 10 per distractor, a several-fold extreme-value advantage that biased the harness *against* amnesia — the opposite error, and equally invisible in the output.
- **Resolution**: RESOLVED - reduced to a single guaranteed slot (`TARGET_KEYS = 1`), which keeps FP16 clean without tilting the comparison. Documented in ADR-012 under rejected alternatives.

- [x] **Finding**: `retrieval_margin` was reported as `top1 - top2` of the pooled scores. With multiple target copies in the pool, both top keys belong to the target, so the column measured jitter noise (mean 0.0003) rather than the target-versus-competitor gap that governs routing. Any figure threshold drawn from it would have been meaningless.
- **Resolution**: RESOLVED - margin is now `max(score | target keys) - max(score | non-target keys)`, i.e. the gap that a distortion must actually cross. Turn 0 is excluded from the figure threshold because only one file is cached there.

- [x] **Finding**: `DIFF_TURNS = {4, 9, 14, 18, 19}` never coincided with a revisited file (the diffed files were 15, 1, 18, 2, 17, each visited once), so no file was ever both visited three times and corrupt. `loop_collapse_turn` was `-1` for every run by construction, and a flat `-1` would have been read as "no loops occur".
- **Resolution**: RESOLVED - diff turns now include the revisits (`{3, 7, 8, 14, 18}`), targeting files 0, 10, 0, 18, 2, so a corrupt edit on a revisited file can actually trigger the three-strike collapse.

- [x] **Finding**: `resolved` was set inline only on the final turn, and then copied onto every row of the run. A session whose earlier edits were corrupted could still "resolve" if the last turn happened to be clean, inflating the headline rate.
- **Resolution**: RESOLVED - resolution is computed once per run as "no corrupted payload on any edit turn and no loop collapse", then broadcast to the run's rows. A debugging session is all-or-nothing, not "mostly resolved".

- [x] **Finding**: The first fixture produced kurtosis 0.099 against p10's measured pre-Hadamard value of ~14.8, because the channel scales were being normalised away. On a near-Gaussian fixture INT4 keys look artificially harmless, so the K channel would have been understated by construction.
- **Resolution**: RESOLVED - per-channel log-normal scales are applied after the cluster term and before unit normalisation, giving measured kurtosis 14.63. `test_fixture_is_deterministic_and_heavy_tailed` asserts the population stays heavy-tailed (band 5-40, not an equality).

- [x] **Finding**: With an isotropic 4096-entry codebook, INT2 values still decoded perfectly at 45% relative error: in 64 dimensions, concentration of measure makes the correct codeword the nearest neighbour almost surely. V-schema corruption would have been exactly zero for every configuration, and the V channel — the one that actually fails — would have looked perfect.
- **Resolution**: RESOLVED - the codebook is clustered (64 centres, sibling cosine ~0.996) and decoding competes only over the target file's 32-symbol action domain, on the argument that a real tool schema constrains what an argument may be. Measured decode mismatch now separates INT8 (0), INT4 (a few percent) and INT2 (~16-18%).

- [x] **Finding**: Per-vector value quantization collapses the INT2/INT4 distinction (measured: both decode cleanly per vector), which would have hidden the effect the cache layout actually causes and made the V2 configuration look free.
- **Resolution**: RESOLVED - values are quantized per channel over a 64-token block, matching `HybridPagedKVCache`'s sealing granularity. This is the mechanism, not a convenience: the choice is recorded in ADR-012.

- [x] **Finding**: The K channel reports zero amnesia at both contexts. A reviewer is entitled to suspect the metric is broken again rather than that routing is robust.
- **Resolution**: RESOLVED (reported, not tuned) - the null is explained quantitatively: measured key distortion tops out well below the minimum measured retrieval margin at every turn, so no argmax flip is reachable. The claim is deliberately narrowed to V-schema being the channel that fails; the fixture margin was *not* moved to manufacture a K effect, and the reasoning is recorded in ADR-012 and `docs/AGENT_FAILURE_MODES.md`.

- [x] **Finding**: K-amnesia at 32K is structurally zero because no revisit at that context exceeds the 20K-token forgotten-span window (the deepest revisit is ~10.3K tokens). Reporting a 32K/64K "comparison" would imply an effect that the design cannot produce.
- **Resolution**: RESOLVED (disclosed) - the 32K arm is documented as a control rather than an effect, in the manifest, ADR-012 and the failure-mode doc.

- [x] **Finding**: `dequantize_k` raises `NotImplementedError`, so the harness quantizes keys one vector at a time through `quantize_group`. At ~0.68 ms per call this is the dominant cost of the sweep and caps how many seeds are affordable.
- **Resolution**: Won't do — implementing `dequantize_k` is a kernel-scope change (`kv_quant/quantizer.py`), out of scope for a harness-only milestone under the one-core-file-per-pass rule, and the per-vector path is the honest reference the fused kernel is gated against. Recorded here rather than patched silently.

- [x] **Finding**: `python tasks/agent_trajectory.py` fails with `ModuleNotFoundError: No module named 'tasks'` because `tasks/` has no `__init__.py`; only `python -m` works.
- **Resolution**: RESOLVED - documented in the reproduce section of `docs/AGENT_FAILURE_MODES.md`, and the figure generator bootstraps the repo root onto `sys.path` explicitly rather than relying on invocation style.

- [x] **Finding**: The parallel sweep (`--jobs`) ran each worker at ~42% of a core. Torch spawns a full-size intra-op thread pool in every child process, so N workers oversubscribed N cores by N and the aggregate reached only ~2.1 of 4 logical cores — the wall-clock saving was half of what the core count implied, and the same defect would have hit any Colab rerun.
- **Resolution**: RESOLVED — `_run_one` now calls `torch.set_num_threads(1)` before simulating, giving each worker one thread. Serial and parallel sweeps were compared row-for-row on the same seeds (`serial=120 parallel=120 identical: True`), so the change affects throughput only.

- [x] **Finding**: The figure's routing threshold was drawn at the *minimum* `retrieval_margin`, a tie-contaminated statistic that included turn 0 (where only one file is cached and target-versus-other is undefined) and reported a negative value even for FP16. Read as a threshold, it implied the control had breached it, which would have been an artifact of the statistic rather than a routing event.
- **Resolution**: RESOLVED — the threshold is now the 1st percentile over turns > 0 (`0.010`), and the flips it was standing in for are counted directly: `6/6,400` records, all onto files read within `12,073` tokens, so none is a forgotten-span re-read. The null is reported with the flip census rather than a misleading minimum.

## Notes for the reviewer

- Every rate in this milestone is a **simulation** number and is labelled as such
  in the CSV, the manifest and the docs. No claim is made about a specific
  model's task accuracy.
- Mis-retrievals are recorded but not propagated (first-order isolation), which
  makes multi-turn degradation a lower bound rather than an estimate.
- The distractor pool subsamples the token population, truncating the extreme
  value tail, which also biases every rate downward.
- Both biases point the same way (understating failure). If the reported V-schema
  effect is real despite them, the direction of the conclusion is safe.
