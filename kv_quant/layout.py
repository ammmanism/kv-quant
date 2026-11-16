"""Single Layout Contract: bit masks, shifts, QJL sketch words, byte-accounting.

This module is the ONLY authority on the physical memory layout of quantized KV pages.
Any change here MUST be reflected in the golden layout hash (checked by check_layout_drift.py).

Byte accounting formula:
    Bytes_per_elem(b, G, asym, qjl) = (b + I_qjl)/8 + 2*(1 + I_asym + I_qjl)/G
where I_x is 1 if feature x is enabled else 0.
"""

from dataclasses import dataclass
from typing import Final


# ─── Bit-width constants ───
# INT4: 2 nibbles per byte
INT4_VALUES_PER_BYTE: Final[int] = 2
INT4_LOW_MASK: Final[int] = 0x0F
INT4_HIGH_MASK: Final[int] = 0xF0
INT4_LOW_SHIFT: Final[int] = 0
INT4_HIGH_SHIFT: Final[int] = 4

# INT2: 4 crumbs per byte
INT2_VALUES_PER_BYTE: Final[int] = 4
INT2_MASKS: Final[tuple[int, ...]] = (0x03, 0x0C, 0x30, 0xC0)
INT2_SHIFTS: Final[tuple[int, ...]] = (0, 2, 4, 6)

# QJL: 32 sign bits per uint32 word
QJL_BITS_PER_WORD: Final[int] = 32
QJL_BYTES_PER_WORD: Final[int] = 4

# ─── Group & page geometry ───
GROUP_SIZE: Final[int] = 128            # tokens per quantization group (max)
PAGE_SIZE: Final[int] = 260             # tokens per page (2 groups + 4 sink)
SINK_TOKENS: Final[int] = 4             # FP16 sink tokens (never quantized)

# ─── QJL sketch ───
QJL_SKETCH_WORDS: Final[int] = 0        # 0 = disabled; >0 enables QJL projection
QJL_SEED: Final[int] = 0xC0FFEE         # fixed seed for reproducibility

# ─── Byte accounting helpers ───
def bytes_per_element(
    bits: int,
    group_size: int,
    asymmetric: bool = False,
    qjl_enabled: bool = False,
) -> float:
    """
    Compute exact bytes per element for a given configuration.

    Formula:
        Bytes_per_elem(b, G, asym, qjl) = (b + I_qjl)/8 + 2*(1 + I_asym + I_qjl)/G

    Args:
        bits: Quantization bit-width (2, 4, 6, 8, 16)
        group_size: Tokens per quantization group
        asymmetric: Whether asymmetric quantization is used (adds zero_point)
        qjl_enabled: Whether QJL sketch is enabled (adds 1 bit + norm)

    Returns:
        Bytes per element (float, may be fractional)
    """
    qjl_bit = 1 if qjl_enabled else 0
    asym_flag = 1 if asymmetric else 0
    qjl_flag = 1 if qjl_enabled else 0

    # Payload bits per element (including QJL sign bit if enabled)
    payload_bytes = (bits + qjl_bit) / 8.0

    # Metadata: scale (fp16 = 2 bytes) + zero_point (uint8 = 1 byte if asym)
    # + QJL norm (fp16 = 2 bytes if qjl)
    # Per group, so divide by group_size
    meta_bytes_per_elem = 2.0 * (1 + asym_flag + qjl_flag) / group_size

    return payload_bytes + meta_bytes_per_elem


def page_bytes(
    bits: int,
    group_size: int,
    asymmetric: bool = False,
    qjl_enabled: bool = False,
    page_size: int = PAGE_SIZE,
    sink_tokens: int = SINK_TOKENS,
) -> int:
    """
    Compute total bytes for a full page.

    Args:
        bits: Quantization bit-width
        group_size: Tokens per quantization group
        asymmetric: Whether asymmetric quantization is used
        qjl_enabled: Whether QJL sketch is enabled
        page_size: Total tokens per page
        sink_tokens: Sink tokens (FP16)

    Returns:
        Total bytes per page (integer)
    """
    quant_tokens = page_size - sink_tokens
    groups_per_page = quant_tokens // group_size

    # Sink bytes (FP16 = 2 bytes per token)
    sink_bytes = sink_tokens * 2

    # Quantized payload bytes
    qjl_bit = 1 if qjl_enabled else 0
    payload_bits_total = quant_tokens * (bits + qjl_bit)
    payload_bytes = (payload_bits_total + 7) // 8  # round up to byte

    # Metadata bytes per group
    meta_per_group = 2 + (1 if asymmetric else 0) + (2 if qjl_enabled else 0)
    meta_bytes = groups_per_page * meta_per_group

    return sink_bytes + payload_bytes + meta_bytes


# ─── Layout version hash (updated when layout changes) ───
# Computed as: sha256(f"{constants in deterministic order}")
LAYOUT_VERSION: Final[str] = "v0.1.0-quantizer"
LAYOUT_GOLDEN_HASH: Final[str] = (
    "913cd248603f77d8bb1b3cffe94fb876f016bd88c3f9b2f69ca4899dfee350b1"
)


@dataclass(frozen=True)
class PageLayout:
    """Immutable description of a quantized KV page layout."""

    sink_tokens: int = SINK_TOKENS
    page_size: int = PAGE_SIZE
    group_size: int = GROUP_SIZE
    bits: int = 4
    asymmetric: bool = False
    qjl_enabled: bool = False
    qjl_words: int = QJL_SKETCH_WORDS

    @property
    def quant_tokens(self) -> int:
        return self.page_size - self.sink_tokens

    @property
    def groups_per_page(self) -> int:
        return self.quant_tokens // self.group_size

    @property
    def bytes_per_elem(self) -> float:
        return bytes_per_element(self.bits, self.group_size, self.asymmetric, self.qjl_enabled)

    @property
    def total_bytes(self) -> int:
        return page_bytes(self.bits, self.group_size, self.asymmetric, self.qjl_enabled,
                          self.page_size, self.sink_tokens)


def get_layout() -> PageLayout:
    """Return the canonical page layout singleton."""
    return PageLayout()


def verify_layout_hash() -> bool:
    """Verify current layout matches golden hash."""
    import hashlib

    # Deterministic serialization of all layout constants
    parts = [
        str(INT4_VALUES_PER_BYTE),
        str(INT4_LOW_MASK),
        str(INT4_HIGH_MASK),
        str(INT4_LOW_SHIFT),
        str(INT4_HIGH_SHIFT),
        str(INT2_VALUES_PER_BYTE),
        str(INT2_MASKS),
        str(INT2_SHIFTS),
        str(QJL_BITS_PER_WORD),
        str(QJL_BYTES_PER_WORD),
        str(GROUP_SIZE),
        str(PAGE_SIZE),
        str(SINK_TOKENS),
        str(QJL_SKETCH_WORDS),
        str(QJL_SEED),
    ]
    data = "|".join(parts).encode()
    current_hash = hashlib.sha256(data).hexdigest()
    return current_hash == LAYOUT_GOLDEN_HASH