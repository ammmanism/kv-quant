"""Single Layout Contract: bit masks, shifts, QJL sketch words, byte-accounting.

This module is the ONLY authority on the physical memory layout of quantized KV pages.
Any change here MUST be reflected in the golden layout hash (checked by check_layout_drift.py).
"""

from dataclasses import dataclass
from typing import Final

# ─── Bit-width constants ───
BITS_PER_QUANT: Final[int] = 4          # 4-bit quantization
VALUES_PER_BYTE: Final[int] = 2         # two 4-bit values per byte
PACK_FACTOR: Final[int] = VALUES_PER_BYTE

# ─── Group & page geometry ───
GROUP_SIZE: Final[int] = 128            # tokens per quantization group
PAGE_SIZE: Final[int] = 260             # tokens per page (2 groups + 4 sink)
SINK_TOKENS: Final[int] = 4             # FP16 sink tokens (never quantized)

# ─── QJL sketch ───
QJL_SKETCH_WORDS: Final[int] = 0        # 0 = disabled; >0 enables QJL projection
QJL_SEED: Final[int] = 0xC0FFEE         # fixed seed for reproducibility

# ─── Bit masks & shifts (little-endian packing) ───
# Each byte holds two 4-bit values: [low_nibble | high_nibble << 4]
LOW_NIBBLE_MASK: Final[int] = 0x0F
HIGH_NIBBLE_MASK: Final[int] = 0xF0
LOW_NIBBLE_SHIFT: Final[int] = 0
HIGH_NIBBLE_SHIFT: Final[int] = 4

# ─── Byte accounting ───
# Quantized page payload (excluding sink & metadata)
QUANT_VALUES_PER_PAGE: Final[int] = PAGE_SIZE - SINK_TOKENS
QUANT_BYTES_PER_PAGE: Final[int] = QUANT_VALUES_PER_PAGE // VALUES_PER_BYTE

# FP16 sink bytes (2 bytes per token)
SINK_BYTES_PER_PAGE: Final[int] = SINK_TOKENS * 2

# Metadata: scale (fp16) + zero_point (uint8) per group
GROUPS_PER_PAGE: Final[int] = (PAGE_SIZE - SINK_TOKENS) // GROUP_SIZE
META_BYTES_PER_GROUP: Final[int] = 2 + 1   # fp16 scale + uint8 zero_point
META_BYTES_PER_PAGE: Final[int] = GROUPS_PER_PAGE * META_BYTES_PER_GROUP

# Total bytes per page
TOTAL_BYTES_PER_PAGE: Final[int] = (
    SINK_BYTES_PER_PAGE + QUANT_BYTES_PER_PAGE + META_BYTES_PER_PAGE
)

# ─── Layout version hash (updated when layout changes) ───
# Computed as: sha256(f"{constants in deterministic order}")
LAYOUT_VERSION: Final[str] = "v0.0.1-init"
LAYOUT_GOLDEN_HASH: Final[str] = (
    "df4d41cc3e964e3a956b40fc5c90b8dfd8c1a737b6e70a12f6d7f4d91795fc3a"
)


@dataclass(frozen=True)
class PageLayout:
    """Immutable description of a quantized KV page layout."""

    sink_tokens: int = SINK_TOKENS
    page_size: int = PAGE_SIZE
    group_size: int = GROUP_SIZE
    bits: int = BITS_PER_QUANT
    qjl_words: int = QJL_SKETCH_WORDS

    @property
    def quant_tokens(self) -> int:
        return self.page_size - self.sink_tokens

    @property
    def quant_bytes(self) -> int:
        return self.quant_tokens // VALUES_PER_BYTE

    @property
    def sink_bytes(self) -> int:
        return self.sink_tokens * 2

    @property
    def groups_per_page(self) -> int:
        return self.quant_tokens // self.group_size

    @property
    def meta_bytes(self) -> int:
        return self.groups_per_page * META_BYTES_PER_GROUP

    @property
    def total_bytes(self) -> int:
        return self.sink_bytes + self.quant_bytes + self.meta_bytes


_LAYOUT_SINGLETON: PageLayout | None = None


def get_layout() -> PageLayout:
    """Return the canonical page layout singleton."""
    global _LAYOUT_SINGLETON
    if _LAYOUT_SINGLETON is None:
        _LAYOUT_SINGLETON = PageLayout()
    return _LAYOUT_SINGLETON


def verify_layout_hash() -> bool:
    """Verify current layout matches golden hash.

    Returns:
        True if layout matches, False otherwise.
    """
    import hashlib

    # Deterministic serialization of all layout constants
    parts = [
        str(BITS_PER_QUANT),
        str(GROUP_SIZE),
        str(PAGE_SIZE),
        str(SINK_TOKENS),
        str(QJL_SKETCH_WORDS),
        str(QJL_SEED),
        str(LOW_NIBBLE_MASK),
        str(HIGH_NIBBLE_MASK),
        str(LOW_NIBBLE_SHIFT),
        str(HIGH_NIBBLE_SHIFT),
    ]
    data = "|".join(parts).encode()
    current_hash = hashlib.sha256(data).hexdigest()
    return current_hash == LAYOUT_GOLDEN_HASH
