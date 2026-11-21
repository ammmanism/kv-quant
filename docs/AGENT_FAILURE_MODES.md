# Agent Failure Modes Under KV-Cache Quantization

How a quantized KV cache breaks a *multi-turn agent*, and which channel breaks
first. Scope, method and limitations are recorded in ADR-012; every number below
resolves to a committed CSV via `scripts/check_claims.py`.

## Two channels, not one

A long-horizon agent fails in two mechanistically distinct ways, and conflating
them produces the wrong fix.

**K channel — routing failure ("K-amnesia").** Keys are what attention uses to
*find* a span. If quantization distorts the key, the query can be routed to a
span the agent already holds. The agent then re-reads a file it read twenty
thousand tokens ago: a re-read of a forgotten span. The cost is latency and
context blow-up, not necessarily a wrong answer — an agent that re-reads is slow
but can still be correct.

**V channel — payload corruption ("V-schema drift").** Values are what attention
uses to *return* content. If quantization distorts the value, the agent reads
back a file whose bytes were mangled — an identifier off by one symbol, a line
number that does not exist. A tool call built from that payload is malformed in a
way that is not syntactically obvious. The cost is a failed edit, and an agent
that does not notice will loop.

The cache's actual layout decides which channel dominates: keys are quantized
per token over `head_dim`, values per channel over a 64-token block. A
per-vector value roundtrip hides this entirely — measured on this fixture, INT2
and INT4 values both decode cleanly when quantized per vector, because
concentration of measure makes a 64-dimensional code robust to large relative
error. Per-channel over a block, the same INT2 values do not decode cleanly.
The layout is the mechanism, so the harness uses the layout.

## Why the agent is simulated

There is no LLM on this host (ADR-006). Rather than hand-assign plausible agent
behaviours, the harness measures two error terms by pushing fixture tensors
through the repo's own `quantize_group`/`dequantize_group`, and lets the
trajectory follow from those measurements:

- `eps_k` — mean key distortion in the scoring basis, between the key the
  retriever held and the key it has after the roundtrip.
- `eps_v` — value reconstruction error plus nearest-neighbour decode mismatch
  over the target file's action domain (its identifiers and line-number bins).

Nothing else decides whether a turn fails. This is a deliberate constraint: it
means a configuration can only look bad here if its tensors actually are bad.

## The four metrics

Definitions are mirrored verbatim in the CSV manifest.

- `k_amnesia_reread_rate` — fraction of turns where quantized retrieval selects a
  file other than the scripted target *and* that file was last read more than the
  forgotten-span window ago.
- `v_schema_corruption_rate` — fraction of turns where any emitted payload slot
  decodes to the wrong symbol.
- `loop_collapse_turn` — the first turn at which a file has been visited three
  times and at least one of those visits returned an unparseable payload; `-1` if
  the run never collapsed.
- `task_resolved_rate` — fraction of runs with no corrupted payload on any edit
  turn and no loop collapse. All-or-nothing per run, because a debugging session
  is not "mostly resolved".

## The trajectory

A twenty-turn ReAct debug session over 24 files: sweep the call path, return to
the entry point, widen across modules, then follow the data flow with one
backtrack. Six revisits, so the agent genuinely returns to files it already
read — without them a re-read is impossible and the K metric measures nothing.

## Results

Sweep: 2 contexts × 5 configurations × 32 seeds × 20 turns = 6,400 turn
records. All rates below are at `64K` unless a `32K` control is named
([#p14_agent_trajectory_drift.csv:context_len]).

| Config | K-amnesia | V-schema | Resolved | Loop collapse |
| :--- | :--- | :--- | :--- | :--- |
| FP16 | `0` | `0` | `100%` | `0` |
| K4V8 post-RoPE | `0` | `0` | `100%` | `0` |
| K4V4 + Hadamard | `0` | `0.16%` | `96.9%` | `0` |
| K8V4 | `0` | `1.87%` | `71.9%` | `9.4%` |
| K8V2 asym | `0` | `15.94%` | `0%` | `78.1%` |

Evidence: K-amnesia [#p14_agent_trajectory_drift.csv:k_amnesia_incident],
V-schema [#p14_agent_trajectory_drift.csv:v_schema_incident], resolved
[#p14_agent_trajectory_drift.csv:resolved], collapse
[#p14_agent_trajectory_drift.csv:loop_collapse_turn].

### The K channel does not fire — and that is measured, not assumed

No configuration produces a single amnesia incident at either context
([#p14_agent_trajectory_drift.csv:k_amnesia_incident]). Two measured reasons:

1. **Routing rarely flips.** Comparing the best target key against the best key
   from any other cached file, a flip (margin < 0) occurs in `6` of `6,400`
   turn-records overall — at most `0.16%` within any single configuration,
   FP16 included ([#p14_agent_trajectory_drift.csv:retrieval_margin]).
2. **When it flips, it lands close.** The oldest file any flip selected had been
   read `12,073` tokens earlier, inside the `20,000`-token forgotten-span
   window, so it is not amnesia ([#p14_agent_trajectory_drift.csv:context_tokens]).

Mean key distortion at `64K` is `0.01485` for K4V8 post-RoPE, `0.00450` for
K4V4 + Hadamard, and `6.7e-5` for both INT8-key configurations
([#p14_agent_trajectory_drift.csv:key_distortion]), against a first-percentile
retrieval margin of `0.010` ([#p14_agent_trajectory_drift.csv:retrieval_margin]).
Only post-RoPE K4 reaches that band, and one flip per configuration in 1,280
turn-records is not a failure mode.

### The V channel is where sessions die

At `64K`, K8V2 corrupts `15.94%` of turn payloads, K8V4 `1.87%`, K4V4 +
Hadamard `0.16%`, and both FP16 and K4V8 post-RoPE `0%`
([#p14_agent_trajectory_drift.csv:v_schema_incident]). Loop collapse follows:
`78.1%` of K8V2 runs and `9.4%` of K8V4 runs hit the three-strike abort at
turn `8`; no other configuration collapses
([#p14_agent_trajectory_drift.csv:loop_collapse_turn]). The `32K` control shows
the same ordering with lower rates — K8V2 `13.13%` V-schema and `78.1%`
collapse, K8V4 `1.09%` and `9.4%`
([#p14_agent_trajectory_drift.csv:v_schema_incident],
[#p14_agent_trajectory_drift.csv:context_len]).

### Same bit budget, opposite outcome

K8V4 and K4V8 post-RoPE allocate the same total bits per element, yet resolve
`71.9%` and `100%` of 20-turn sessions at `64K` (`78.1%` and `100%` at `32K`)
([#p14_agent_trajectory_drift.csv:resolved]). Spending the four bits on V
(K8V4) loses sessions; spending them on K (K4V8) does not, because the keys
were never the constraint. This is the cleanest available contrast: same
nominal budget, opposite failure channel.

## How to read the result

If a configuration shows K-amnesia, its keys are the problem and the fix is
key-side: raise K bits, or rotate before quantizing (Hadamard) so the heavy-tail
channels are shared rather than individually clipped.

If a configuration shows V-schema corruption instead, its keys are fine and the
fix is value-side: raise V bits, or stop spending V precision on a routing
problem that does not exist. Re-reads are not the failure; failed edits are.

A configuration can be excellent at one and poor at the other. Because K8V4 and
K4V8 occupy the same memory, the pair is the cleanest contrast available: same
budget, opposite failure channel.

## Limitations

- **Simulated, not measured on a model.** The harness isolates a mechanism; it
  does not predict a specific model's task accuracy.
- **First-order isolation.** A mis-retrieval is recorded but not propagated: the
  next turn proceeds as if the agent had read the right file. Real agents
  compound errors, so multi-turn degradation here is a lower bound.
- **Subsampled distractors.** The pool samples a small fraction of the token
  population per turn. That truncates the extreme-value tail, so every rate here
  is conservative.
- **The 32K arm is a control, not an effect.** No revisit at 32K exceeds the
  forgotten-span window, so K-amnesia is structurally zero there by
  construction. Only the 64K arm can inform the K channel.
- **Loop collapse is a V-channel symptom.** It is conditioned on an unparseable
  payload, so it must not be read as independent evidence about keys.

## Reproduce

```bash
uv run python -m tasks.agent_trajectory          # writes the CSV + manifest
uv run python scripts/generate_phase14_artifacts.py   # renders Figure 5
```

`python tasks/agent_trajectory.py` will not work: `tasks/` has no `__init__.py`,
so the repo root must be on `sys.path`, which `-m` guarantees and a script
invocation does not.
