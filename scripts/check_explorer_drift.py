#!/usr/bin/env python3
"""CI Gate: docs/explorer.html can never drift from bench/data/*.csv.

Rebuilds the explorer in memory (which re-verifies every manifest SHA256)
and byte-compares against the committed file; asserts zero external
resource loads and a sane size budget.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_interactive_explorer import OUT_PATH, build_html  # noqa: E402

MAX_BYTES = 600 * 1024
EXTERNAL_PATTERNS = [
    r"<script[^>]+src=",  # external scripts
    r"<link[^>]+href=",  # stylesheets / icons
    r"<img[^>]+src=",  # images
    r"<iframe\b",
    r"<object\b",
    r"<embed\b",
    r"<audio\b",
    r"<video\b",
    r"<source\b",
    r"@import",  # CSS imports
    r"url\(",  # CSS resource loads
    r"fetch\(",  # network calls from JS
    r"XMLHttpRequest",
    r"sendBeacon",
    r"WebSocket",
    r"EventSource",
]
EXPECTED_ARTIFACTS = 19  # tracked CSVs minus the gitignored CI stub


def main() -> int:
    if not OUT_PATH.exists():
        print(f"[FAIL] {OUT_PATH} does not exist; run scripts/build_interactive_explorer.py")
        return 1
    committed = OUT_PATH.read_bytes()

    # 1. Rebuild (this re-verifies every manifest hash and row count).
    rebuilt = build_html().encode("utf-8")

    ok = True
    if committed != rebuilt:
        ok = False
        n = min(len(committed), len(rebuilt))
        first = next((i for i in range(n) if committed[i] != rebuilt[i]), n)
        print(
            "[FAIL] docs/explorer.html has drifted from bench/data/*.csv "
            f"(committed {len(committed)} B vs rebuilt {len(rebuilt)} B; "
            f"first diff at byte {first})."
        )
        print("       Regenerate with: uv run python scripts/build_interactive_explorer.py")
    else:
        print(
            f"[OK] byte-identical rebuild ({len(committed)} B) — explorer matches all {EXPECTED_ARTIFACTS} CSVs"
        )

    # 2. Zero external resources (self-contained, works offline from file://).
    text = committed.decode("utf-8", errors="replace")
    for pattern in EXTERNAL_PATTERNS:
        hits = re.findall(pattern, text, flags=re.IGNORECASE)
        if hits:
            ok = False
            print(f"[FAIL] external resource reference detected: {pattern} ({len(hits)} hits)")
    if all(not re.findall(p, text, flags=re.IGNORECASE) for p in EXTERNAL_PATTERNS):
        print("[OK] zero external resource loads (fully offline)")

    # 3. Provenance completeness: every tracked artifact appears with its hash.
    import json  # noqa: PLC0415  (deferred: only needed once the rebuild passed)

    marker = "window.KQ_DATA = "
    start = text.find(marker)
    if start < 0:
        print("[FAIL] embedded payload not found")
        return 1
    payload = json.loads(
        text[start + len(marker) : text.find("</script>", start)].rstrip(" \t\r\n;")
    )
    artifacts = payload["meta"]["artifacts"]
    if len(artifacts) != EXPECTED_ARTIFACTS:
        ok = False
        print(f"[FAIL] payload lists {len(artifacts)} artifacts, expected {EXPECTED_ARTIFACTS}")
    else:
        print(f"[OK] provenance block lists all {len(artifacts)} artifacts with SHA256 + mode")
    for art in artifacts:
        if len(art["sha256"]) != 64 or not art["mode"]:
            ok = False
            print(f"[FAIL] malformed provenance row: {art['file']}")

    # 4. Size budget (keeps the single file sane to review and commit).
    if len(committed) > MAX_BYTES:
        ok = False
        print(f"[FAIL] explorer is {len(committed)} B > {MAX_BYTES} B budget")
    else:
        print(f"[OK] size {len(committed) / 1024:.1f} KiB within {MAX_BYTES / 1024:.0f} KiB budget")

    print("[PASS: explorer-drift]" if ok else "[FAIL: explorer-drift]")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
