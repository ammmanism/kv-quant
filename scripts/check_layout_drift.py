#!/usr/bin/env python3
"""Invariant 3: Single Layout Contract.

kv_quant/layout.py is the single source of truth for bit masks, shifts,
QJL sketch words, and byte-accounting equations.

This script verifies the current layout matches the committed golden hash.
"""

import hashlib
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kv_quant.layout import (
    LAYOUT_GOLDEN_HASH,
    LAYOUT_VERSION,
)


def compute_current_hash() -> str:
    """Compute hash of current layout constants."""
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
        PAGE_SIZE,
        QJL_BITS_PER_WORD,
        QJL_BYTES_PER_WORD,
        QJL_SEED,
        QJL_SKETCH_WORDS,
        SINK_TOKENS,
    )

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
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    print(f"Layout version: {LAYOUT_VERSION}")
    print(f"Golden hash:    {LAYOUT_GOLDEN_HASH}")

    current_hash = compute_current_hash()
    print(f"Current hash:   {current_hash}")

    if current_hash == LAYOUT_GOLDEN_HASH:
        print("\n[OK] Layout matches golden hash - no drift detected")
        return 0
    else:
        print("\n[FAIL] LAYOUT DRIFT DETECTED!")
        print("   The layout constants in kv_quant/layout.py have changed.")
        print("   If intentional, update LAYOUT_GOLDEN_HASH and LAYOUT_VERSION.")
        print("   If accidental, revert the changes to layout.py.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
