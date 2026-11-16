#!/usr/bin/env python3
"""Invariant 1: Evidence Traceability.

Every numeric claim in README.md or docs/*.md MUST cite a committed CSV via ``.
"""

import re
import sys
from pathlib import Path

import pandas as pd

CLAIM_PATTERN = re.compile(r"\[#([^\]]+)\]")

# Patterns to skip (template/examples, not real claims)
SKIP_PATTERNS = [
    r"\.\.\.",  # ellipsis placeholder
    r"file\.csv:column",  # example format
    r"filename\.csv:",  # example format
]
CSV_DIR = Path("bench/data")


def find_claims_in_file(path: Path) -> list[tuple[int, str]]:
    """Find all [#...] claims in a markdown file."""
    claims = []
    content = path.read_text(encoding="utf-8")
    for i, line in enumerate(content.splitlines(), 1):
        for match in CLAIM_PATTERN.finditer(line):
            ref = match.group(1)
            # Skip template/example patterns
            if any(re.search(p, ref) for p in SKIP_PATTERNS):
                continue
            claims.append((i, ref))
    return claims


def resolve_claim(ref: str) -> tuple[Path | None, str | None]:
    """Resolve a claim reference to a CSV file and column."""
    # Format: filename.csv:column_name or filename.csv:row,col
    if ":" not in ref:
        return None, f"Invalid reference format: {ref} (expected file.csv:column)"

    csv_name, spec = ref.split(":", 1)
    csv_path = CSV_DIR / csv_name

    if not csv_path.exists():
        return None, f"CSV not found: {csv_path}"

    return csv_path, None


def verify_claim(csv_path: Path, spec: str) -> tuple[bool, str]:
    """Verify a claim exists in the CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"Failed to read CSV {csv_path}: {e}"

    # Spec can be: column_name, or row,col (0-indexed)
    if "," in spec:
        try:
            row_str, col_str = spec.split(",", 1)
            row = int(row_str)
            col = int(col_str)
            if row >= len(df) or col >= len(df.columns):
                return False, f"Index out of bounds: row={row}, col={col} (shape={df.shape})"
            val = df.iloc[row, col]
            return True, f"Value: {val}"
        except ValueError:
            return False, f"Invalid row,col spec: {spec}"
    else:
        # Column name
        if spec not in df.columns:
            return False, f"Column not found: {spec} (available: {list(df.columns)})"
        return True, f"Column '{spec}' exists with {len(df)} rows"


def main() -> int:
    """Check all markdown files for unverified claims."""
    md_files = list(Path().rglob("*.md"))
    # Exclude rfc/reviews (adversary gate handles those)
    md_files = [f for f in md_files if not str(f).startswith("rfc/reviews/")]

    all_ok = True
    total_claims = 0

    for md_file in md_files:
        claims = find_claims_in_file(md_file)
        if not claims:
            continue

        print(f"\n[FILE] {md_file}: {len(claims)} claim(s)")
        for line_num, ref in claims:
            total_claims += 1
            csv_path, error = resolve_claim(ref)
            if error:
                print(f"  [FAIL] Line {line_num}: [{ref}] -> {error}")
                all_ok = False
                continue

            ok, msg = verify_claim(csv_path, spec=ref.split(":", 1)[1])
            if ok:
                print(f"  [OK] Line {line_num}: [{ref}] -> {msg}")
            else:
                print(f"  [FAIL] Line {line_num}: [{ref}] -> {msg}")
                all_ok = False

    print(f"\n{'=' * 50}")
    print(f"Total claims checked: {total_claims}")
    if all_ok:
        print("[OK] All claims verified against committed CSVs")
        return 0
    else:
        print("[FAIL] Some claims could not be verified")
        return 1


if __name__ == "__main__":
    sys.exit(main())
