"""Kernel Parity Tests (Invariant 6).

dequant_fused.py must match dequant_naive.py within 1e-3 on synthetic
and real captured activation fixtures.
"""

import pytest
import torch
from kv_quant.kernels.dequant_naive import dequant_naive
from kv_quant.layout import PageLayout, get_layout
from kv_quant.quantizer import QuantParams

try:
    from kv_quant.kernels.dequant_fused import dequant_fused
    HAS_TRITON = True
except ImportError:
    dequant_fused = None
    HAS_TRITON = False


@pytest.fixture
def layout() -> PageLayout:
    return get_layout()


@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_test_page(layout: PageLayout, device: str) -> tuple[torch.Tensor, list[QuantParams]]:
    """Create a synthetic test page with known values."""
    page_size = layout.page_size
    sink_tokens = layout.sink_tokens
    group_size = layout.group_size
    groups_per_page = (page_size - sink_tokens) // group_size

    # Create packed page data
    total_bytes = layout.total_bytes
    packed = torch.zeros(total_bytes, dtype=torch.uint8, device=device)

    # Fill sink with known FP16 values
    sink_bytes = sink_tokens * 2
    if sink_bytes > 0:
        sink_values = torch.arange(sink_tokens, dtype=torch.float16, device=device) * 0.1
        packed[:sink_bytes] = sink_values.view(torch.uint8)

    # Fill quantized groups with known INT4 values
    params_list = []
    quant_offset = sink_bytes
    quant_bytes_per_group = group_size // 2

    for g in range(groups_per_page):
        # Create test pattern: alternating low/high nibbles
        group_data = torch.zeros(quant_bytes_per_group, dtype=torch.uint8, device=device)
        for i in range(quant_bytes_per_group):
            low = (g * 16 + i * 2) % 16
            high = (g * 16 + i * 2 + 1) % 16
            group_data[i] = low | (high << 4)

        packed[quant_offset:quant_offset + quant_bytes_per_group] = group_data
        quant_offset += quant_bytes_per_group

        # Params for this group
        scale = torch.tensor(0.05, dtype=torch.float16, device=device)
        zero_point = torch.tensor(8, dtype=torch.uint8, device=device)
        params_list.append(QuantParams(scale=scale, zero_point=zero_point))

    # Fill metadata (scales and zero_points) - not used by dequant functions directly
    # but part of the packed format
    meta_bytes_per_group = 3  # fp16 + uint8
    for g in range(groups_per_page):
        # Scale (fp16)
        scale_bytes = torch.tensor(0.05, dtype=torch.float16, device=device).view(torch.uint8)
        packed[quant_offset:quant_offset + 2] = scale_bytes
        quant_offset += 2
        # Zero point (uint8)
        packed[quant_offset] = 8
        quant_offset += 1

    return packed, params_list


def test_dequant_parity_synthetic(layout: PageLayout, device: str):
    """Test fused vs naive dequant on synthetic data."""
    if not HAS_TRITON:
        pytest.skip("Triton not available")
    packed, params_list = make_test_page(layout, device)

    # Run both implementations
    naive_out = dequant_naive(packed, params_list, layout)
    fused_out = dequant_fused(packed, params_list, layout)

    # Compare
    max_diff = (naive_out - fused_out).abs().max().item()
    print(f"Max difference: {max_diff}")

    assert max_diff < 1e-3, f"Parity failed: max diff = {max_diff} >= 1e-3"


def test_dequant_parity_random(layout: PageLayout, device: str):
    """Test fused vs naive dequant on random data."""
    if not HAS_TRITON:
        pytest.skip("Triton not available")
    torch.manual_seed(42)

    page_size = layout.page_size
    sink_tokens = layout.sink_tokens
    group_size = layout.group_size
    groups_per_page = (page_size - sink_tokens) // group_size

    total_bytes = layout.total_bytes
    packed = torch.randint(0, 256, (total_bytes,), dtype=torch.uint8, device=device)

    params_list = []
    for g in range(groups_per_page):
        scale = torch.tensor(torch.rand(1).item() * 0.1 + 0.01, dtype=torch.float16, device=device)
        zero_point = torch.randint(0, 16, (1,), dtype=torch.uint8, device=device).item()
        params_list.append(QuantParams(scale=scale, zero_point=torch.tensor(zero_point, dtype=torch.uint8, device=device)))

    naive_out = dequant_naive(packed, params_list, layout)
    fused_out = dequant_fused(packed, params_list, layout)

    max_diff = (naive_out - fused_out).abs().max().item()
    print(f"Max difference (random): {max_diff}")

    assert max_diff < 1e-3, f"Parity failed on random data: max diff = {max_diff} >= 1e-3"


@pytest.mark.gpu
def test_dequant_parity_gpu(layout: PageLayout):
    """GPU-specific parity test."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not HAS_TRITON:
        pytest.skip("Triton not available")
    test_dequant_parity_synthetic(layout, "cuda")
    test_dequant_parity_random(layout, "cuda")
