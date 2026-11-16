#!/usr/bin/env python3
"""Invariant 2: Visual Provenance.

Every PNG in docs/figures/*.png MUST have a companion
docs/figures/<name>.png.meta.json recording:
- Generating script path
- Source CSV paths
- Source CSV SHA256 hashes
- Image dimensions (>=300 DPI)
"""

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

FIGURES_DIR = Path("docs/figures")
REQUIRED_META_FIELDS = [
    "generating_script",
    "source_csvs",
    "csv_hashes",
    "width_px",
    "height_px",
    "dpi",
]


def compute_sha256(path: Path) -> str:
    """Compute SHA256 with normalized line endings (cross-platform stable).

    Git may check out CRLF on Windows but LF on Linux; normalizing ensures the
    committed hash matches on every platform.
    """
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def check_figure(png_path: Path) -> tuple[bool, list[str]]:
    """Check a single figure for provenance compliance."""
    errors = []

    # Check meta file exists
    meta_path = png_path.with_suffix(png_path.suffix + ".meta.json")
    if not meta_path.exists():
        errors.append(f"Missing meta file: {meta_path}")
        return False, errors

    # Load and validate meta
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in meta file: {e}")
        return False, errors

    # Check required fields
    for field in REQUIRED_META_FIELDS:
        if field not in meta:
            errors.append(f"Missing required field in meta: {field}")

    if errors:
        return False, errors

    # Verify generating script exists (normalize separators for cross-platform)
    script_path = Path(meta["generating_script"].replace("\\", "/"))
    if not script_path.exists():
        errors.append(f"Generating script not found: {script_path}")

    # Verify source CSVs exist and hashes match (normalize separators)
    for csv_rel, expected_hash in zip(meta["source_csvs"], meta["csv_hashes"]):
        csv_path = Path(csv_rel.replace("\\", "/"))
        if not csv_path.exists():
            errors.append(f"Source CSV not found: {csv_path}")
        else:
            actual_hash = compute_sha256(csv_path)
            if actual_hash != expected_hash:
                errors.append(
                    f"CSV hash mismatch for {csv_path}: "
                    f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                )

    # Verify image dimensions and DPI
    try:
        with Image.open(png_path) as img:
            width, height = img.size
            dpi = img.info.get("dpi", (72, 72))[0]  # default 72 if not set
            dpi = round(dpi)  # PIL stores DPI as float with rounding error

            if width != meta["width_px"]:
                errors.append(f"Width mismatch: meta={meta['width_px']}, actual={width}")
            if height != meta["height_px"]:
                errors.append(f"Height mismatch: meta={meta['height_px']}, actual={height}")
            if dpi < 300:
                errors.append(f"DPI too low: {dpi} (minimum 300)")
    except Exception as e:
        errors.append(f"Failed to read image: {e}")

    return len(errors) == 0, errors


def main() -> int:
    if not FIGURES_DIR.exists():
        print(f"Figures directory not found: {FIGURES_DIR}")
        print("[OK] No figures to check (directory doesn't exist)")
        return 0

    png_files = list(FIGURES_DIR.glob("*.png"))
    if not png_files:
        print("[OK] No PNG figures found in docs/figures/")
        return 0

    print(f"Checking {len(png_files)} figure(s) in {FIGURES_DIR}...\n")

    all_ok = True
    for png in png_files:
        ok, errors = check_figure(png)
        if ok:
            print(f"[OK] {png.name}")
        else:
            print(f"[FAIL] {png.name}")
            for err in errors:
                print(f"   - {err}")
            all_ok = False

    print(f"\n{'=' * 50}")
    if all_ok:
        print("[OK] All figures have valid provenance metadata")
        return 0
    else:
        print("[FAIL] Some figures failed provenance checks")
        return 1


if __name__ == "__main__":
    sys.exit(main())
