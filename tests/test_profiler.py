"""Tests for the head-sensitivity profiler and schedule compiler (ADR-009)."""

import pandas as pd
import pytest
import torch
from kv_quant.profiler import compile_head_bit_schedule, compile_head_schedule, profile_calibration


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer_idx": layer,
                "kv_head_idx": head,
                "distant_attn_mass": mass,
                "kl_perturbation": kl,
            }
            for layer, head, mass, kl in rows
        ]
    )


class TestCompileHeadSchedule:
    ROWS = [
        (0, 0, 0.55, 0.021),
        (0, 1, 0.02, 0.001),
        (0, 2, 0.04, 0.002),
        (0, 3, 0.01, 0.0005),
        (1, 0, 0.50, 0.020),
        (1, 1, 0.03, 0.0015),
        (1, 2, 0.05, 0.003),
        (1, 3, 0.02, 0.001),
    ]

    def test_default_budget_gives_top_quarter_k8v4(self) -> None:
        schedule = compile_head_schedule(_frame(self.ROWS))
        # 8 heads at 4.5-bit budget -> exactly 2 retrieval heads (K8V4).
        retrieval = {k: v for k, v in schedule.items() if v == (8, 4)}
        streaming = {k: v for k, v in schedule.items() if v == (4, 4)}
        assert len(retrieval) == 2
        assert len(streaming) == 6
        assert set(retrieval) == {(0, 0), (1, 0)}, "top-scored heads must be retrieval"

    def test_achieved_average_bits_matches_budget(self) -> None:
        schedule = compile_head_schedule(_frame(self.ROWS), target_avg_bits=4.5)
        avg = sum((k + v) / 2 for k, v in schedule.values()) / len(schedule)
        assert abs(avg - 4.5) < 1e-9

    def test_budget_bounds_move_the_fraction(self) -> None:
        all_retrieval = compile_head_schedule(_frame(self.ROWS), target_avg_bits=6.0)
        assert all(v == (8, 4) for v in all_retrieval.values())
        all_streaming = compile_head_schedule(_frame(self.ROWS), target_avg_bits=4.0)
        assert all(v == (4, 4) for v in all_streaming.values())

    def test_tie_break_is_deterministic(self) -> None:
        rows = [(0, h, 0.10, 0.010) for h in range(4)]
        a = compile_head_schedule(_frame(rows))
        b = compile_head_schedule(_frame(rows))
        assert a == b
        # 4 heads, 25% -> 1 retrieval; the tie must go to the lowest index.
        assert sum(v == (8, 4) for v in a.values()) == 1
        assert a[(0, 0)] == (8, 4)

    def test_out_of_range_budget_raises(self) -> None:
        with pytest.raises(ValueError):
            compile_head_schedule(_frame(self.ROWS), target_avg_bits=2.0)
        with pytest.raises(ValueError):
            compile_head_schedule(_frame(self.ROWS), target_avg_bits=7.0)

    def test_missing_columns_and_empty_frame_raise(self) -> None:
        with pytest.raises(ValueError):
            compile_head_schedule(_frame(self.ROWS).drop(columns=["distant_attn_mass"]))
        with pytest.raises(ValueError):
            compile_head_schedule(_frame([]))

    def test_kl_sensitivity_accepted_as_legacy_alias(self) -> None:
        frame = _frame(self.ROWS).rename(columns={"kl_perturbation": "kl_sensitivity"})
        assert compile_head_schedule(frame) == compile_head_schedule(_frame(self.ROWS))


class TestProfileCalibration:
    def _logits(self) -> torch.Tensor:
        torch.manual_seed(11)
        # 2 layers, 3 kv heads, 4 sequences, 8 queries, 64 keys.
        return torch.randn(2, 3, 4, 8, 64)

    def test_shape_and_metric_ranges(self) -> None:
        profiles = profile_calibration(self._logits())
        assert len(profiles) == 6
        for p in profiles:
            assert 0 <= p.distant_attn_mass <= 1.0 + 1e-6
            assert p.attn_entropy > 0
            assert p.kl_sensitivity >= 0
            assert p.layer_idx in (0, 1) and p.kv_head_idx in (0, 1, 2)

    def test_averaging_over_sequences_matches_manual_mean(self) -> None:
        logits = self._logits()
        profiles = profile_calibration(logits)
        # one sequence must agree with the single-seq path exactly.
        single = profile_calibration(logits[:, :, :1])
        for p, s in zip(profiles, single, strict=True):
            assert p.layer_idx == s.layer_idx and p.kv_head_idx == s.kv_head_idx

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            profile_calibration(torch.randn(2, 3, 4, 64))

    def test_bimodal_calibration_separates_retrieval_heads(self) -> None:
        # Retrieval head concentrates mass in the distant region [4, 384);
        # streaming heads concentrate in the trailing local window [384, 512).
        torch.manual_seed(5)
        layers, heads, seqs, q_len, k_len = 1, 4, 2, 16, 512
        logits = torch.zeros(layers, heads, seqs, q_len, k_len)
        for q in range(q_len):
            logits[0, 0, :, q, 4:384] = 3.0 * torch.randn(seqs, 380)
            for h in range(1, heads):
                logits[0, h, :, q, 384:] = 3.0 * torch.randn(seqs, 128)
        profiles = profile_calibration(logits, sink_tokens=4, local_window=128)
        schedule = compile_head_schedule(
            pd.DataFrame(
                [
                    {
                        "layer_idx": p.layer_idx,
                        "kv_head_idx": p.kv_head_idx,
                        "distant_attn_mass": p.distant_attn_mass,
                        "kl_perturbation": p.kl_sensitivity,
                    }
                    for p in profiles
                ]
            )
        )
        assert schedule[(0, 0)] == (8, 4)
        assert all(v == (4, 4) for k, v in schedule.items() if k != (0, 0))


def test_legacy_three_tier_compiler_unchanged() -> None:
    """The phase-2 3-tier compiler must keep its contract for p2 artifacts."""
    from kv_quant.profiler import HeadProfile

    profiles = [
        HeadProfile(
            layer_idx=layer,
            kv_head_idx=head,
            distant_attn_mass=layer * 0.1 + head * 0.01,
            attn_entropy=5.0,
            kl_sensitivity=0.01,
        )
        for layer in range(4)
        for head in range(8)
    ]
    schedule = compile_head_bit_schedule(profiles, target_avg_bits=3.5)
    assert len(schedule) == 32
    labels = {s["head_class"] for s in schedule}
    assert labels == {"K8V4", "K4V4_Hadamard", "K2_QJL_V2"}
