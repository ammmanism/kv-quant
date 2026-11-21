"""Tests for the simulated agent-trajectory harness (ADR-012).

These guard the fixture invariants that make the reported metrics meaningful.
They are deliberately cheap: no full rollout, which takes minutes.
"""

import torch
from tasks.agent_trajectory import (
    CONFIGS,
    IDEAL_SCRIPT,
    TARGET_KEYS,
    TURNS,
    _key_pool,
    build_fixture,
    corrupt_positions,
    epsilon_key,
)


def test_fixture_is_deterministic_and_heavy_tailed() -> None:
    """Same seed must give identical keys; the tail must stay non-Gaussian."""
    a = build_fixture(seed=7, context_len=65536)
    b = build_fixture(seed=7, context_len=65536)
    assert torch.equal(a.file_keys, b.file_keys)
    # The fixture is calibrated to p10's pre-Hadamard kurtosis (~14.8). A band,
    # not an equality: the point is that it stays heavy-tailed, not Gaussian.
    assert 5.0 < a.kurtosis < 40.0


def test_script_revisits_files() -> None:
    """A re-read is only possible if the script returns to an earlier file."""
    assert len(IDEAL_SCRIPT) == TURNS
    assert len(set(IDEAL_SCRIPT)) < TURNS, "20 distinct files cannot produce a re-read"


def test_fp16_key_roundtrip_is_lossless() -> None:
    """The FP16 control must not lose key fidelity, else amnesia is unmeasurable."""
    fx = build_fixture(seed=0, context_len=65536)
    assert abs(epsilon_key(fx, CONFIGS[0])) < 1e-6


def test_fp16_payload_decodes_cleanly() -> None:
    """The FP16 control must produce no V-schema corruption at all."""
    fx = build_fixture(seed=0, context_len=65536)
    assert not corrupt_positions(fx, CONFIGS[0]).any()


def test_key_pool_always_represents_the_target() -> None:
    """A uniform draw can miss the target; the reserved slot must prevent that."""
    fx = build_fixture(seed=0, context_len=65536)
    rng = torch.Generator().manual_seed(0)
    for t in range(TURNS):
        target = fx.target_of_turn[t]
        cached = set(fx.target_of_turn[: t + 1])
        _, pick, _ = _key_pool(fx, cached, target, 64, rng)
        assert int((pick == target).sum()) >= TARGET_KEYS
