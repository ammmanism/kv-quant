#!/usr/bin/env python3
"""Build docs/explorer.html from the SHA256-verified bench/data CSVs.

Deterministic: no timestamps, no absolute paths, no randomness. The drift
gate (scripts/check_explorer_drift.py) rebuilds this document in memory and
byte-compares it against the committed file, so the explorer can never
silently diverge from the committed evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from explorer_data import build_payload  # noqa: E402
from explorer_template import render_html  # noqa: E402

OUT_PATH = Path("docs/explorer.html")


def build_html() -> str:
    """Verify manifests, aggregate, and render the full document."""
    payload = build_payload()
    artifacts = payload["meta"]["artifacts"]
    assert isinstance(artifacts, list)
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    # Escape "<" so a payload string can never terminate the <script> block.
    return render_html(payload_json.replace("<", "\\u003c"), artifacts)


def main() -> int:
    html = build_html()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote {OUT_PATH} ({len(html.encode()) / 1024:.1f} KiB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
