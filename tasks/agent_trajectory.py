"""Multi-turn agentic trajectory harness (ADR-012).

Simulates a 20-turn ReAct repo-debugging session (context 8K -> 64K) and
measures how KV-cache quantization degrades the agent. This is a simulation:
there is no LLM on this CPU-only host, and nothing behavioural is hard-coded.

Two error terms are measured by pushing real tensors through the repo's own
``quantize_group``/``dequantize_group``: ``eps_k`` (key distortion -> routing
failures -> K-amnesia rereads) and ``eps_v`` (value distortion -> malformed
payload arguments -> V-schema corruption). The fixture is heavy-tailed on
purpose, calibrated to p10's measured pre-Hadamard kurtosis (~14.8); on a
Gaussian fixture INT4 looks artificially harmless.
"""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from kv_quant.hadamard import compute_kurtosis, fast_walsh_hadamard, inverse_fast_walsh_hadamard
from kv_quant.quantizer import dequantize_group, quantize_group
from tasks.base import SweepManifest

HEAD_DIM = 64
CODEBOOK_SIZE = 4096
CODEBOOK_CLUSTERS = 64
CODEBOOK_SPREAD = 0.06  # sibling cosine ~0.996: NN decode error-sensitive
SLOT_VOCAB = 32  # per-file action domain (identifiers + line-number bins)
PAYLOAD_LEN = 24
N_FILES = 24
N_MODULES = 4
KURTOSIS_POPULATION = 2048
TURNS = 20
START_TOKENS = 8192
CONTEXTS = (32768, 65536)
AMNESIA_AGE_TOKENS = 20000  # a span older than this counts as forgotten, not cached
# Realistic 20-turn ReAct debug script (index into the 24 files). The agent
# sweeps the call path, returns to the entry point, widens across modules, then
# follows the data flow with one backtrack. Six revisits total.
IDEAL_SCRIPT: tuple[int, ...] = (
    0,
    5,
    10,
    0,  # entry -> config loader -> parser -> back to entry (revisit)
    15,
    5,  # schema -> loader (revisit)
    20,
    10,
    0,  # tests -> parser (revisit) -> entry (revisit)
    1,
    6,
    11,
    16,
    21,  # widen across the remaining modules
    2,
    7,
    12,
    2,
    17,
    22,  # data flow, with one backtrack (revisit)
)
assert len(IDEAL_SCRIPT) == TURNS
DISTRACTOR_SAMPLE = 256
TARGET_KEYS = 1  # guaranteed presence only; a larger reservation biases against amnesia
CHANNEL_SIGMA = 1.25  # log-normal channel scale; kurtosis ~= p10's 14.8
CLUSTER_SPREAD = 0.05
TOKEN_JITTER = 0.05  # L2 norm of a token's offset from its file key (cos ~= 0.9987)
QUERY_NOISE = 0.015  # base query ambiguity; per-turn scale drawn in the sim
HADAMARD_SEED = 42
ROPE_BASE = 10000.0


@dataclass(frozen=True)
class ConfigSpec:
    """One KV-cache configuration under test."""

    name: str
    k_bits: int
    v_bits: int
    k_scheme: Literal["sym", "asym"] = "sym"
    v_scheme: Literal["sym", "asym"] = "asym"
    hadamard: bool = False
    post_rope: bool = False


CONFIGS: tuple[ConfigSpec, ...] = (
    ConfigSpec("fp16", 16, 16),
    ConfigSpec("k8v4", 8, 4),
    ConfigSpec("k4v8_postrope", 4, 8, post_rope=True),
    ConfigSpec("k8v2_asym", 8, 2),
    ConfigSpec("k4v4_hadamard", 4, 4, hadamard=True),
)


@dataclass
class Fixture:
    """Seeded trajectory fixture. Tensors are float32 before quantization."""

    file_keys: torch.Tensor
    file_module: list[int]
    slot_ids: torch.Tensor
    payload: torch.Tensor
    codebook: torch.Tensor
    queries: torch.Tensor
    positions: torch.Tensor
    target_of_turn: list[int]
    tokens_at_turn: list[int]
    kurtosis: float


def _unit(x: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return out


def apply_rope(vecs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Standard rotate-half RoPE on the last dimension (orthogonal)."""
    half = HEAD_DIM // 2
    freqs = 1.0 / (ROPE_BASE ** (torch.arange(half, dtype=torch.float32) / half))
    ang = positions.to(torch.float32)[:, None] * freqs[None, :]
    x1, x2 = vecs[:, :half], vecs[:, half:]
    return torch.cat([x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()], dim=-1)


def build_fixture(seed: int = 2026, context_len: int = 65536) -> Fixture:
    """Build the seeded heavy-tailed trajectory fixture."""
    g = torch.Generator().manual_seed(seed)
    centers = _unit(torch.randn(N_MODULES, HEAD_DIM, generator=g))
    scales = torch.exp(CHANNEL_SIGMA * torch.randn(HEAD_DIM, generator=g))
    modules = torch.arange(KURTOSIS_POPULATION) % N_MODULES
    tokens = centers[modules] + CLUSTER_SPREAD * torch.randn(
        KURTOSIS_POPULATION, HEAD_DIM, generator=g
    )
    token_keys = _unit(tokens * scales)

    # Clustered vocabulary: siblings are ~0.996 cosine apart, so payload
    # decoding is genuinely error-sensitive (an isotropic codebook is not).
    cb_centers = _unit(torch.randn(CODEBOOK_CLUSTERS, HEAD_DIM, generator=g))
    cb = cb_centers[torch.arange(CODEBOOK_SIZE) // CODEBOOK_CLUSTERS]
    cb = cb + (CODEBOOK_SPREAD / math.sqrt(HEAD_DIM)) * torch.randn(
        CODEBOOK_SIZE, HEAD_DIM, generator=g
    )
    codebook = _unit(cb * scales)
    slot_ids = torch.stack(
        [torch.randperm(CODEBOOK_SIZE, generator=g)[:SLOT_VOCAB] for _ in range(N_FILES)]
    )
    payload = torch.randint(0, SLOT_VOCAB, (N_FILES, PAYLOAD_LEN), generator=g)
    file_keys = token_keys[:N_FILES]
    queries = _unit(file_keys + QUERY_NOISE * torch.randn(N_FILES, HEAD_DIM, generator=g))

    # A realistic ReAct debug script over 24 files: 20 turns with 6 backtracking
    # revisits, so a file read early is returned to later (after the context has
    # grown). ``(t * 5) % N_FILES`` would give 20 *distinct* files and a re-read
    # could never happen at all.
    target_of_turn = list(IDEAL_SCRIPT)
    step = (context_len - START_TOKENS) / (TURNS - 1)
    return Fixture(
        file_keys=file_keys,
        file_module=modules[:N_FILES].tolist(),
        slot_ids=slot_ids,
        payload=payload,
        codebook=codebook,
        queries=queries,
        positions=torch.arange(N_FILES, dtype=torch.float32) * 64.0,
        target_of_turn=target_of_turn,
        tokens_at_turn=[int(START_TOKENS + t * step) for t in range(TURNS)],
        kurtosis=float(compute_kurtosis(token_keys, dim=-1).mean()),
    )


def _roundtrip_key(vec: torch.Tensor, spec: ConfigSpec) -> torch.Tensor:
    """Per-token key quantization over head_dim (group_size = head_dim)."""
    x = vec.to(torch.float16)
    if spec.hadamard:
        x = fast_walsh_hadamard(x, scope="full", seed=HADAMARD_SEED)
    packed, params = quantize_group(x, spec.k_bits, spec.k_scheme)
    out = dequantize_group(packed, params, spec.k_bits, spec.k_scheme)
    if spec.hadamard:
        out = inverse_fast_walsh_hadamard(out, scope="full", seed=HADAMARD_SEED)
    return out.to(torch.float32)


def apply_rope_inverse(vecs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Inverse (transpose) of :func:`apply_rope`."""
    half = HEAD_DIM // 2
    freqs = 1.0 / (ROPE_BASE ** (torch.arange(half, dtype=torch.float32) / half))
    ang = positions.to(torch.float32)[:, None] * freqs[None, :]
    x1, x2 = vecs[:, :half], vecs[:, half:]
    return torch.cat([x1 * ang.cos() + x2 * ang.sin(), -x1 * ang.sin() + x2 * ang.cos()], dim=-1)


def roundtrip_keys(keys: torch.Tensor, spec: ConfigSpec, positions: torch.Tensor) -> torch.Tensor:
    """Reconstruct keys in the *unrotated* basis used for scoring.

    RoPE is orthogonal, so un-rotating after reconstruction is exact. The only
    difference between pre- and post-RoPE configs is the basis in which the
    quantizer operates -- which is precisely the effect under study.
    """
    if spec.post_rope:
        rotated = apply_rope(keys, positions)
        recon = torch.stack([_roundtrip_key(r, spec) for r in rotated])
        return _unit(apply_rope_inverse(recon, positions))
    recon = torch.stack([_roundtrip_key(r, spec) for r in keys])
    return _unit(recon)


def roundtrip_values(block: torch.Tensor, spec: ConfigSpec) -> torch.Tensor:
    """Per-channel V quantization over a 64-token block (the cache layout).

    ``block`` is (64, HEAD_DIM). Hadamard rotates each token across head_dim
    before per-channel quantization, exactly like ``rotate_qk``.
    """
    x = block.float()
    if spec.hadamard:
        x = fast_walsh_hadamard(x, scope="full", seed=HADAMARD_SEED)
    out = torch.empty_like(x)
    for d in range(HEAD_DIM):
        channel = x[:, d].to(torch.float16)
        packed, params = quantize_group(channel, spec.v_bits, spec.v_scheme)
        out[:, d] = dequantize_group(packed, params, spec.v_bits, spec.v_scheme).float()
    if spec.hadamard:
        out = inverse_fast_walsh_hadamard(out, scope="full", seed=HADAMARD_SEED)
    return out


def _file_payload_values(fx: Fixture, file_id: int) -> torch.Tensor:
    """(PAYLOAD_LEN, HEAD_DIM) value vectors emitted by this file's diff."""
    return fx.codebook[fx.slot_ids[file_id][fx.payload[file_id]]]


def epsilon_key(fx: Fixture, spec: ConfigSpec) -> float:
    """Mean key cosine distortion in the scoring basis."""
    recon = roundtrip_keys(fx.file_keys, spec, fx.positions)
    return float((1.0 - (_unit(fx.file_keys) * recon).sum(dim=-1)).mean())


def epsilon_value(fx: Fixture, spec: ConfigSpec) -> tuple[float, float]:
    """Relative value error and payload decode-mismatch rate.

    Decoding competes only over the target file's ``SLOT_VOCAB`` action domain
    (identifiers + line-number bins), not the full 4096-token vocabulary: a
    real tool schema constrains what an argument may be.
    """
    rels: list[float] = []
    mismatches = 0
    for f in range(N_FILES):
        values = _file_payload_values(fx, f)
        block = torch.zeros(64, HEAD_DIM)
        block[:PAYLOAD_LEN] = values
        recon = roundtrip_values(block, spec)[:PAYLOAD_LEN]
        rels.append(
            float(((recon - values).norm(dim=-1) / values.norm(dim=-1).clamp_min(1e-9)).mean())
        )
        vocab = _unit(fx.codebook[fx.slot_ids[f]])
        decoded = (_unit(recon) @ vocab.T).argmax(dim=-1)
        mismatches += int((decoded != fx.payload[f]).sum())
    return float(sum(rels) / len(rels)), mismatches / (N_FILES * PAYLOAD_LEN)


# ─── Trajectory simulation (pass 2) ───

SEEDS: tuple[int, ...] = tuple(range(32))
EMIT_POSITIONS = 6
DIFF_TURNS = frozenset({3, 7, 8, 14, 18})  # includes revisits, so retry loops are reachable
TOOL_CYCLE = ("read_file", "search_symbol", "run_pytest")


@dataclass
class TurnRecord:
    """One row of ``bench/data/p14_agent_trajectory_drift.csv``."""

    context_len: int
    config: str
    seed: int
    turn: int
    tool: str
    context_tokens: int
    target_file: int
    retrieved_file: int
    key_distortion: float  # mean (1 - cos) of the pooled keys after quantization
    retrieval_margin: float  # post-quantization top1 - top2 score gap
    k_amnesia_incident: int
    v_schema_incident: int
    loop_collapse_turn: int
    resolved: int


def corrupt_positions(fx: Fixture, spec: ConfigSpec) -> torch.Tensor:
    """(N_FILES, PAYLOAD_LEN) per-slot decode-corruption flags."""
    flags = torch.zeros(N_FILES, PAYLOAD_LEN, dtype=torch.bool)
    for f in range(N_FILES):
        values = _file_payload_values(fx, f)
        block = torch.zeros(64, HEAD_DIM)
        block[:PAYLOAD_LEN] = values
        recon = roundtrip_values(block, spec)[:PAYLOAD_LEN]
        vocab = _unit(fx.codebook[fx.slot_ids[f]])
        decoded = (_unit(recon) @ vocab.T).argmax(dim=-1)
        flags[f] = decoded != fx.payload[f]
    return flags


def _key_pool(
    fx: Fixture, cached: set[int], target: int, n_keys: int, rng: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample token keys from the cached files, with the target guaranteed a slot.

    A uniform draw can miss the target entirely when many files are cached, and
    then an FP16 run reports "amnesia" that is pure sampling noise. Exactly
    ``TARGET_KEYS`` slot goes to the target and the rest are uniform over the
    cached files, so the target gets no more draws than a distractor and the
    retrieval comparison stays fair.

    ``TOKEN_JITTER`` is the L2 norm of the per-token offset, so a sampled key
    sits ~0.9987 cosine from its file key -- well inside the ~0.045 sibling
    margin. Scaling ``randn`` directly instead would put tokens further from
    their own file than neighbouring files are from each other.
    """
    pick = torch.full((n_keys,), target, dtype=torch.long)
    pool_files = sorted(cached - {target})
    n_target = min(TARGET_KEYS, n_keys)
    if pool_files and n_keys > n_target:
        others = torch.tensor(pool_files)
        pick[n_target:] = others[torch.randint(0, len(others), (n_keys - n_target,), generator=rng)]
    jitter = TOKEN_JITTER / math.sqrt(HEAD_DIM)
    keys = _unit(fx.file_keys[pick] + jitter * torch.randn(n_keys, HEAD_DIM, generator=rng))
    return keys, pick, fx.positions[pick]


def simulate_run(context_len: int, spec: ConfigSpec, seed: int) -> list[TurnRecord]:
    """Roll out one 20-turn trajectory and record per-turn diagnostics."""
    fx = build_fixture(seed=seed, context_len=context_len)
    corrupt = corrupt_positions(fx, spec)
    cfg_idx = [c.name for c in CONFIGS].index(spec.name)
    rng = torch.Generator().manual_seed(seed * 977 + cfg_idx * 13 + context_len)
    cached: set[int] = set()
    last_read: dict[int, int] = {}
    corrupt_ever: set[int] = set()
    records: list[TurnRecord] = []

    for t in range(TURNS):
        target = fx.target_of_turn[t]
        cached.add(target)
        tokens = fx.tokens_at_turn[t]
        n_keys = min(DISTRACTOR_SAMPLE, max(64, tokens // 64))
        keys, pick, pos = _key_pool(fx, cached, target, n_keys, rng)
        query = _unit(fx.queries[target])
        recon = roundtrip_keys(keys, spec, pos)
        scores = recon @ query
        retrieved_q = int(pick[int(scores.argmax())])
        # The margin that governs routing is target vs a *different* file; the
        # top-2 gap would compare two copies of the target and measure jitter.
        is_target = pick == target
        best_target = float(scores[is_target].max())
        best_other = float(scores[~is_target].max()) if bool((~is_target).any()) else best_target
        margin = best_target - best_other
        distortion = float((1.0 - (keys * recon).sum(dim=-1)).mean())

        age = tokens - last_read.get(retrieved_q, tokens)
        amnesia = int(retrieved_q != target and age > AMNESIA_AGE_TOKENS)
        if t in DIFF_TURNS:
            tool = "apply_diff"
            chosen = torch.randint(0, PAYLOAD_LEN, (EMIT_POSITIONS,), generator=rng)
            v_incident = int(bool(corrupt[target][chosen].any()))
        else:
            tool = TOOL_CYCLE[t % len(TOOL_CYCLE)]
            v_incident = 0
        last_read[target] = tokens
        if amnesia:
            last_read[retrieved_q] = tokens  # the agent just re-read it
        if v_incident:
            corrupt_ever.add(target)
        records.append(
            TurnRecord(
                context_len,
                spec.name,
                seed,
                t,
                tool,
                tokens,
                target,
                retrieved_q,
                distortion,
                margin,
                amnesia,
                v_incident,
                -1,
                0,
            )
        )

    # Loop collapse: the agent returned to the same file three times and at least
    # one of those attempts yielded an unparseable payload -- the classic thrash
    # of re-read, fail to parse, re-read again.
    loop_turn = -1
    counts: dict[int, int] = {}
    for record in records:
        counts[record.target_file] = counts.get(record.target_file, 0) + 1
        if loop_turn < 0 and counts[record.target_file] >= 3 and record.target_file in corrupt_ever:
            loop_turn = record.turn
    resolved = int(loop_turn < 0 and not any(r.v_schema_incident for r in records))
    for record in records:
        record.loop_collapse_turn = loop_turn
        record.resolved = resolved
    return records


def _run_one(task: tuple[int, str, int]) -> list[TurnRecord]:
    """Worker entry point for the process pool (module-level so it pickles)."""
    context_len, config_name, seed = task
    # One intra-op thread per worker. Torch otherwise spawns a full-size thread
    # pool in every child, so N workers oversubscribe N cores by N and each ends
    # up throttled to ~40% of a core.
    torch.set_num_threads(1)
    spec = next(c for c in CONFIGS if c.name == config_name)
    return simulate_run(context_len, spec, seed)


def run_sweep(
    contexts: tuple[int, ...] = CONTEXTS,
    specs: tuple[ConfigSpec, ...] = CONFIGS,
    seeds: tuple[int, ...] = SEEDS,
    jobs: int = 1,
) -> list[TurnRecord]:
    """Execute the full 2 context x 5 config x N seed sweep.

    Runs are independent and individually seeded, so ``jobs > 1`` changes
    throughput only, never results: the row order is fixed explicitly below.
    On Windows the caller must sit behind an ``if __name__ == "__main__"`` guard.
    """
    tasks = [(ctx, spec.name, seed) for ctx in contexts for spec in specs for seed in seeds]
    if jobs <= 1:
        chunks = [_run_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            chunks = list(pool.map(_run_one, tasks))
    records = [record for chunk in chunks for record in chunk]
    order = {spec.name: i for i, spec in enumerate(CONFIGS)}
    records.sort(key=lambda r: (r.context_len, order[r.config], r.seed, r.turn))
    return records


def write_artifacts(
    csv_path: str = "bench/data/p14_agent_trajectory_drift.csv", jobs: int = 1
) -> str:
    """Run the sweep, write the CSV, and pin it in a manifest."""
    records = run_sweep(jobs=jobs)
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TurnRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    manifest = SweepManifest.capture(model_id="Qwen/Qwen2.5-7B-Instruct", seed=42)
    manifest.extra = {
        "description": "20-turn agentic trajectory: K-amnesia vs V-schema drift",
        "mode": "simulated (no LLM/GPU on host); behaviour driven by measured quantizer error",
        "turns": TURNS,
        "seeds": len(SEEDS),
        "rows": len(records),
        "fixture_kurtosis": round(build_fixture(context_len=CONTEXTS[0]).kurtosis, 3),
        "channel_sigma": CHANNEL_SIGMA,
        "codebook_spread": CODEBOOK_SPREAD,
        "slot_vocab": SLOT_VOCAB,
        "distractor_sample": DISTRACTOR_SAMPLE,
        "target_keys_reserved": TARGET_KEYS,
        "token_jitter_norm": TOKEN_JITTER,
        "amnesia_age_tokens": AMNESIA_AGE_TOKENS,
        "k_quant_granularity": "per-token over head_dim",
        "v_quant_granularity": "per-channel over a 64-token block (cache layout)",
        "isolation": "first-order: a mis-retrieval is recorded but not propagated to later turns",
        "trajectory": f"{TURNS}-turn ReAct script over {N_FILES} files with 6 backtracking revisits",
        "tools": ["read_file", "search_symbol", "run_pytest", "apply_diff"],
        "metric_definitions": {
            "k_amnesia_reread_rate": (
                "fraction of turns where quantized retrieval selects a file other than the "
                f"scripted target and the selected file was last read >{AMNESIA_AGE_TOKENS} "
                "tokens earlier (a re-read of a forgotten span)"
            ),
            "v_schema_corruption_rate": (
                "fraction of turns where any emitted payload slot decodes to the wrong symbol "
                f"({EMIT_POSITIONS} slots sampled on the {len(DIFF_TURNS)} apply_diff turns)"
            ),
            "loop_collapse_turn": (
                "first turn at which a file has been visited 3 times and at least one of those "
                "visits produced an unparseable payload; -1 if the run never collapsed"
            ),
            "task_resolved_rate": (
                "fraction of runs with no corrupted payload on any apply_diff turn and no loop "
                "collapse"
            ),
        },
        "subsample": f"{DISTRACTOR_SAMPLE} sampled token keys/turn (1:{max(CONTEXTS) // DISTRACTOR_SAMPLE} of 64K); tail understated, so rates are conservative",
    }
    manifest.with_csv_hash(path).write(path.with_suffix(".csv.manifest.json"))
    return str(path)


def main(argv: list[str] | None = None) -> int:
    """Emit p14 CSV + manifest and print the headline table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel worker processes (results are identical for any value)",
    )
    args = parser.parse_args(argv)
    path = write_artifacts(jobs=args.jobs)
    records = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    print(f"wrote {path}: {len(records)} rows")
    print(f"{'ctx':>6}{'config':>16}{'amnesia':>10}{'v_schema':>10}{'resolved':>10}")
    for ctx in CONTEXTS:
        for spec in CONFIGS:
            rows = [r for r in records if int(r["context_len"]) == ctx and r["config"] == spec.name]
            amn = sum(int(r["k_amnesia_incident"]) for r in rows) / len(rows)
            vsc = sum(int(r["v_schema_incident"]) for r in rows) / len(rows)
            res = sum(int(r["resolved"]) for r in rows) / (TURNS * len(SEEDS))
            print(f"{ctx:>6}{spec.name:>16}{amn:>10.4f}{vsc:>10.4f}{res:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
