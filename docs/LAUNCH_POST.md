# kv-quant v2.0.0 — Launch Post

Three ready-to-post variants of the same story. Every number resolves to a
committed, SHA256-verified CSV via the repo's citation format
(`[#file.csv:column]`, enforced by `scripts/check_claims.py` in CI) — keep the
citations when posting to the blog, trim them for X/LinkedIn character limits
but never trim the honesty labels (*projected* / *modeled* / *simulated*).

---

## Show HN

**Show HN: kv-quant — we measured exactly where INT4 KV-cache quantization breaks your long-context agent (and where it doesn't)**

We quantized the KV cache of Qwen2.5-7B-Instruct (GQA 7:1, 32K→128K contexts) and kept hitting the same confusing claim in the literature: "INT4 KV is basically free." It isn't — but the failure is narrower and stranger than we expected, and the interesting part is *which* half of the cache breaks.

Three findings, all with committed CSV artifacts and Wilson-95% intervals:

**1. The cliff is real, position-resolved, and key-sided.** Naive INT4-K (post-RoPE) collapses needle accuracy to `0.8455` at 32K vs `0.9924` for FP16 — a 14.8% drop — while INT4-*V* barely moves ([#p4_needle_depth_matrix.csv:acc_mean]). Keys govern softmax routing; values ride along. The cliff is depth-dependent: it starts at ~70% insertion depth ([#p4_needle_depth_matrix.csv:depth_pct]).

**2. A 16-line Hadamard rotation rescues it.** The cause is post-RoPE channel outliers (kurtosis `14.842`). A block-diagonal Walsh–Hadamard rotation of just the rotary subspace drops kurtosis to `2.011` and buys `+7.349 dB` SQNR on our outlier fixture ([#p10_hadamard_rotation.csv:sqnr_delta_db]). Combined with a retrieval-head profiler that spends K8V4 only on the ~25% of heads that actually retrieve ([#p11_head_sensitivity_schedule.csv:assigned_k_bits]), the 128K gap vs FP16 shrinks to `2.0–2.6%` under a 4.5-bit budget ([#p10_hadamard_rescue.csv:acc_mean], [#p11_head_rescue_curve.csv:acc_mean]). Honest caveat: those rescue curves are projections from declared synthetic fixtures via a replay adapter — not model forward passes.

**3. For multi-turn agents, everyone worries about the wrong channel.** We built a *simulated* 20-turn ReAct harness (no LLM forward pass — failures follow measured quantizer error on synthetic clustered codebooks, ADR-012) and separated the two failure modes people conflate. "K-amnesia" (the agent forgets a file and re-reads it) turned out to be a *measured null*: `0` incidents in 6,400 turn-records; routing flipped only 6 times, all within a 12,073-token window ([#p14_agent_trajectory_drift.csv:k_amnesia_incident]). Meanwhile INT2 *values* corrupted `15.94%` of edit payloads and collapsed `78.1%` of runs into retry loops at turn 8 ([#p14_agent_trajectory_drift.csv:v_schema_incident], [#p14_agent_trajectory_drift.csv:loop_collapse_turn]). Same bit budget, opposite outcome: K8V4 resolves `71.9%` of sessions where K4V8 resolves `100%` ([#p14_agent_trajectory_drift.csv:resolved]). If you serve agents: protect the V channel first.

Fleet math (analytical byte model, not GPU runs): copy-on-write paged prefix sharing alone fits `3.73×` more concurrent sessions (41 vs 11 at 24 GiB/32K); quant + CoW gets you to `14.64×` (161 vs 11) ([#p13_ragged_concurrency.csv:max_concurrent_sessions]).

Everything is explorable in a zero-dependency interactive dashboard (docs/explorer.html) that is *byte-gated* in CI — it rebuilds from the SHA256-verified CSVs and fails if the committed HTML drifts by one byte. Greyed-out cells in the UI are honest "not measured" states; every panel carries its epistemic badge.

CPU-only repo, Apache-2.0, 9 CI gates including a claims-provenance checker that refuses any number without a committed artifact. Ask us anything — especially if you think the replay-adapter methodology is cheating; we wrote ADR-012 about exactly that.

---

## X thread (7 posts)

1/ We quantized KV caches and found out *which half* breaks your agent. Spoiler: it's not the one everyone blames. 🧵

2/ Naive INT4 keys collapse needle accuracy 14.8% at 32K. INT4 values: barely a scratch. Keys do routing; values do payload. That asymmetry is the whole story. [#p4_needle_depth_matrix.csv:acc_mean]

3/ The culprit: post-RoPE channel outliers (kurtosis 14.8). A block-diagonal Hadamard rotation of the rotary subspace flattens it to 2.0 and adds +7.35 dB SQNR. [#p10_hadamard_rotation.csv:sqnr_delta_db]

4/ Then a retrieval-head profiler spends K8V4 on the ~25% of heads that actually retrieve, K4V4 elsewhere: 128K gap to FP16 shrinks to 2.0–2.6% at a 4.5-bit budget. (Projections from synthetic fixtures — labeled as such.) [#p11_head_rescue_curve.csv:acc_mean]

5/ Now the agent part. 20-turn ReAct harness (SIMULATED — measured quantizer error, no LLM). 6,400 turn-records. "K-amnesia" — agent forgets a file, re-reads it? ZERO incidents. Routing flipped 6 times total, all within 12K tokens. [#p14_agent_trajectory_drift.csv:k_amnesia_incident]

6/ But INT2 values corrupted 15.9% of edit payloads and sent 78% of runs into retry loops at turn 8. Same bit budget: K8V4 resolves 71.9% of sessions, K4V8 resolves 100%. Protect the V channel. [#p14_agent_trajectory_drift.csv:v_schema_incident]

7/ Fleet (MODELED byte model): CoW prefix sharing fits 3.7× more concurrent agent sessions (11→41 on 24 GiB); +INT4 quant gets 14.6× (161 sessions). Explore it all interactively — byte-gated against committed CSVs. Apache-2.0, CPU-only. [#p13_ragged_concurrency.csv:max_concurrent_sessions]

---

## LinkedIn post

**Where does KV-cache quantization actually break? We committed the answer as CSVs.**

Most INT4 KV-cache benchmarks report a single accuracy number and hide where the loss lives. We built the evaluation spine we wished existed: position-resolved needle sweeps, per-channel kurtosis diagnostics, a retrieval-head profiler, a 20-turn agentic stress harness, and a fleet concurrency model — every number in a committed CSV with a SHA256 manifest, and a CI gate that deletes any claim without an artifact.

What the data says:

• INT4 keys, not values, are the cliff: −14.8% needle accuracy at 32K, onset at deep insertion positions ([#p4_needle_depth_matrix.csv:acc_mean]).
• The fix is cheap: a block-diagonal Walsh–Hadamard rotation of the rotary subspace (+7.35 dB SQNR on our outlier fixture) plus K8V4 on the ~25% retrieval-biased heads brings the 128K residual to 2.0–2.6% at a 4.5-bit budget ([#p10_hadamard_rotation.csv:sqnr_delta_db], [#p11_head_rescue_curve.csv:acc_mean]) — projections, honestly labeled.
• For multi-turn agents, the feared failure ("amnesia" re-reads) never fired in 6,400 turn-records of a **simulated** harness driven by measured quantizer error; corrupted *values* collapsed 78% of INT2 runs into retry loops instead ([#p14_agent_trajectory_drift.csv:k_amnesia_incident], [#p14_agent_trajectory_drift.csv:loop_collapse_turn]).
• Serving economics (analytical **modeled** byte model, not GPU runs): CoW prefix sharing fits 3.7× more concurrent sessions at 24 GiB; quant on top reaches 14.6× ([#p13_ragged_concurrency.csv:max_concurrent_sessions]).

v2.0.0 ships an interactive, zero-dependency explorer (docs/explorer.html) rebuilt and byte-compared from the committed evidence on every CI run — greyed cells mean "not measured," and every panel is badged measured / projected / modeled / simulated.

Apache-2.0, CPU-only, 9 CI gates. If you run long-context agents in production, the decision matrix is in the README — start with "protect the V channel."

---

*Posting checklist: keep the `[#file.csv:column]` citations on the blog/HN; the
repo's claims gate (Invariant 1) fails CI if any cited number loses its CSV.
Numbers that are projections or models must keep their labels — that is the
entire point of this repo.*
