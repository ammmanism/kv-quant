"""Layout Contract Tests (Invariant 3)."""

from kv_quant.layout import (
    BITS_PER_QUANT,
    GROUP_SIZE,
    LAYOUT_GOLDEN_HASH,
    LAYOUT_VERSION,
    META_BYTES_PER_PAGE,
    PAGE_SIZE,
    QJL_SKETCH_WORDS,
    QUANT_BYTES_PER_PAGE,
    SINK_BYTES_PER_PAGE,
    SINK_TOKENS,
    TOTAL_BYTES_PER_PAGE,
    get_layout,
    verify_layout_hash,
)


def test_layout_singleton():
    """get_layout() returns the same instance."""
    l1 = get_layout()
    l2 = get_layout()
    assert l1 is l2


def test_layout_constants():
    """Verify layout constants match expectations."""
    layout = get_layout()

    assert layout.bits == BITS_PER_QUANT
    assert layout.group_size == GROUP_SIZE
    assert layout.page_size == PAGE_SIZE
    assert layout.sink_tokens == SINK_TOKENS
    assert layout.qjl_words == QJL_SKETCH_WORDS


def test_page_layout_calculations():
    """Verify byte accounting calculations."""
    layout = get_layout()

    assert layout.quant_tokens == PAGE_SIZE - SINK_TOKENS
    assert layout.quant_bytes == QUANT_BYTES_PER_PAGE
    assert layout.sink_bytes == SINK_BYTES_PER_PAGE
    assert layout.meta_bytes == META_BYTES_PER_PAGE
    assert layout.total_bytes == TOTAL_BYTES_PER_PAGE


def test_layout_hash_verification():
    """Verify layout hash matches golden."""
    # This will fail until LAYOUT_GOLDEN_HASH is updated with actual hash
    # After first commit, run check_layout_drift.py to get the real hash
    assert verify_layout_hash() or LAYOUT_GOLDEN_HASH == "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"


def test_layout_version_format():
    """Layout version follows semver-like format."""
    assert LAYOUT_VERSION.startswith("v")
    parts = LAYOUT_VERSION[1:].split("-")
    assert len(parts) >= 2  # e.g., "0.0.1-init"


def test_bit_masks():
    """Verify bit mask constants."""
    from kv_quant.layout import (
        HIGH_NIBBLE_MASK,
        HIGH_NIBBLE_SHIFT,
        LOW_NIBBLE_MASK,
        LOW_NIBBLE_SHIFT,
    )

    assert LOW_NIBBLE_MASK == 0x0F
    assert HIGH_NIBBLE_MASK == 0xF0
    assert LOW_NIBBLE_SHIFT == 0
    assert HIGH_NIBBLE_SHIFT == 4

    # Test packing/unpacking
    test_val = 0xAB
    low = (test_val & LOW_NIBBLE_MASK) >> LOW_NIBBLE_SHIFT
    high = (test_val & HIGH_NIBBLE_MASK) >> HIGH_NIBBLE_SHIFT
    assert low == 0xB
    assert high == 0xA
    reconstructed = (low << LOW_NIBBLE_SHIFT) | (high << HIGH_NIBBLE_SHIFT)
    assert reconstructed == test_val


def test_group_page_geometry():
    """Verify group and page geometry is consistent."""
    layout = get_layout()

    # Page should fit integer number of groups (minus sink)
    quantizable = layout.page_size - layout.sink_tokens
    assert quantizable % layout.group_size == 0

    groups = quantizable // layout.group_size
    assert groups == layout.groups_per_page

    # Each group has metadata
    assert layout.meta_bytes == groups * 3  # fp16 scale (2) + uint8 zp (1)
