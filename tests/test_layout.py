"""Layout Contract Tests (Invariant 3)."""

from kv_quant.layout import (
    GROUP_SIZE,
    INT2_MASKS,
    INT2_SHIFTS,
    INT2_VALUES_PER_BYTE,
    INT4_HIGH_MASK,
    INT4_HIGH_SHIFT,
    INT4_LOW_MASK,
    INT4_LOW_SHIFT,
    INT4_VALUES_PER_BYTE,
    LAYOUT_GOLDEN_HASH,
    LAYOUT_VERSION,
    PAGE_SIZE,
    QJL_BITS_PER_WORD,
    QJL_BYTES_PER_WORD,
    QJL_SKETCH_WORDS,
    SINK_TOKENS,
    bytes_per_element,
    get_layout,
    page_bytes,
    verify_layout_hash,
)


def test_layout_singleton():
    """get_layout() returns a valid PageLayout."""
    l1 = get_layout()
    l2 = get_layout()
    assert l1 == l2
    assert l1.group_size == GROUP_SIZE
    assert l1.page_size == PAGE_SIZE
    assert l1.sink_tokens == SINK_TOKENS
    assert l1.qjl_words == QJL_SKETCH_WORDS


def test_int4_packing_spec():
    """INT4: 2 nibbles/byte, masks 0x0F/0xF0, shifts 0/4."""
    assert INT4_VALUES_PER_BYTE == 2
    assert INT4_LOW_MASK == 0x0F
    assert INT4_HIGH_MASK == 0xF0
    assert INT4_LOW_SHIFT == 0
    assert INT4_HIGH_SHIFT == 4

    test_val = 0xAB
    low = (test_val & INT4_LOW_MASK) >> INT4_LOW_SHIFT
    high = (test_val & INT4_HIGH_MASK) >> INT4_HIGH_SHIFT
    assert low == 0xB
    assert high == 0xA
    reconstructed = (low << INT4_LOW_SHIFT) | (high << INT4_HIGH_SHIFT)
    assert reconstructed == test_val


def test_int2_packing_spec():
    """INT2: 4 crumbs/byte, masks 0x03/0x0C/0x30/0xC0, shifts 0/2/4/6."""
    assert INT2_VALUES_PER_BYTE == 4
    assert INT2_MASKS == (0x03, 0x0C, 0x30, 0xC0)
    assert INT2_SHIFTS == (0, 2, 4, 6)


def test_qjl_packing_spec():
    """QJL: 32 sign bits per uint32 word."""
    assert QJL_BITS_PER_WORD == 32
    assert QJL_BYTES_PER_WORD == 4


def test_byte_accounting_formula():
    """Verify the byte accounting formula: (b+I_qjl)/8 + 2(1+I_asym+I_qjl)/G."""
    b, g = 4, 64
    # sym, no qjl: 4/8 + 2/64 = 0.5 + 0.03125
    assert abs(bytes_per_element(b, g, False, False) - (0.5 + 2 / 64)) < 1e-9
    # asym: + 1 zero_point byte per group
    assert abs(bytes_per_element(b, g, True, False) - (0.5 + 4 / 64)) < 1e-9
    # asym + qjl: (4+1)/8 + 2*3/64
    assert abs(bytes_per_element(b, g, True, True) - (5 / 8 + 6 / 64)) < 1e-9


def test_page_bytes():
    """Verify page byte accounting is consistent."""
    layout = get_layout()
    pb = page_bytes(4, layout.group_size, False, False,
                    layout.page_size, layout.sink_tokens)
    # Should be positive and larger than pure quant payload
    assert pb > 0
    assert pb >= layout.quant_tokens * 4 // 8


def test_layout_hash_verification():
    """Verify layout hash matches golden."""
    assert verify_layout_hash()
    assert LAYOUT_GOLDEN_HASH != ""


def test_layout_version_format():
    """Layout version follows semver-like format."""
    assert LAYOUT_VERSION.startswith("v")
    parts = LAYOUT_VERSION[1:].split("-")
    assert len(parts) >= 2  # e.g., "0.1.0-quantizer"


def test_group_page_geometry():
    """Verify group and page geometry is consistent."""
    layout = get_layout()

    # Page should fit integer number of groups (minus sink)
    quantizable = layout.page_size - layout.sink_tokens
    assert quantizable % layout.group_size == 0

    groups = quantizable // layout.group_size
    assert groups == layout.groups_per_page
