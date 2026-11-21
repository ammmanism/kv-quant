#!/usr/bin/env python3
"""Invariant 9: Adversarial Gate.

Verifies that PR has an adversarial review file with all findings resolved.
"""

import re
import sys
from pathlib import Path

REVIEWS_DIR = Path("rfc/reviews")
RESOLUTION_PATTERNS = [
    r"Fixed in `[a-f0-9]{7,40}`",
    r"Fixed in [a-f0-9]{7,40}",
    # A marker must be followed by an actual justification, not just the word.
    r"(?i)^\s*(?:RESOLVED|Fixed)\b[\s\u2014:-]+\S",
    r"(?i)^\s*Won't do\b[\s(\u2014:-]+\S",
    # Legacy audits in this repo use these two documented outcome markers.
    r"(?i)^\s*Investigated\b[\s\u2014:-]+\S",
    r"(?i)user-approved[^\n]*resolution applied",
    r"^Question\b",
]
# A resolution that still says PENDING / not fixed / TBD is NOT resolved.
PENDING_PATTERN = re.compile(r"(?i)\b(?:PENDING|not fixed|TBD|TODO)\b")


def check_review_file(review_path: Path) -> tuple[bool, list[str]]:
    """Check a review file for resolved findings."""
    errors = []
    content = review_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Find all findings (lines starting with "- [ ]" or "- [x]")
    # Format: "- [x] **Finding**: ..." followed by "- **Resolution**: ..."
    findings = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r"^-\s*\[([ x])\]\s*\*\*Finding\*\*:\s*(.+)$", line)
        if match:
            checked = match.group(1)
            finding = match.group(2)
            # Look for resolution on next line(s)
            resolution = ""
            j = i + 1
            while j < len(lines):
                res_match = re.match(r"^-\s*\*\*Resolution\*\*:\s*(.+)$", lines[j])
                if res_match:
                    # Resolutions may span multiple lines; join the whole
                    # block so a marker on any line counts, not just the first.
                    parts = [res_match.group(1)]
                    k = j + 1
                    while k < len(lines):
                        if (
                            re.match(r"^-\s*\[[ x]\]\s*\*\*Finding\*\*:", lines[k])
                            or re.match(r"^-\s*\*\*Resolution\*\*:", lines[k])
                            or re.match(r"^### ", lines[k])
                            or not lines[k].strip()
                        ):
                            break
                        parts.append(lines[k].strip())
                        k += 1
                    resolution = " ".join(parts)
                    break
                # Stop if we hit another finding or section
                if re.match(r"^-\s*\[[ x]\]\s*\*\*Finding\*\*:", lines[j]):
                    break
                if re.match(r"^### ", lines[j]):
                    break
                j += 1
            findings.append((checked, finding, resolution))
        i += 1

    if not findings:
        errors.append(
            "No findings found in review file (expected format: '- [ ] **Finding**: ...' followed by '- **Resolution**: ...')"
        )
        return False, errors

    unresolved = 0
    for checked, finding, resolution in findings:
        if checked != "x":
            unresolved += 1
            errors.append(f"Unresolved finding: {finding}")
        # Verify resolution marker exists in resolution text
        elif PENDING_PATTERN.search(resolution):
            errors.append(f"Resolution is still pending: {finding} | Resolution: {resolution}")
        elif not any(re.search(p, resolution) for p in RESOLUTION_PATTERNS):
            errors.append(
                f"Resolved finding missing resolution marker: {finding} | Resolution: {resolution}"
            )

    return unresolved == 0 and len(errors) == 0, errors


def main() -> int:
    # Review files contain typographic characters (≥, —, ✓); a Windows console
    # defaulting to cp1252 raises UnicodeEncodeError mid-report, masking the
    # actual verdict. Force UTF-8 output wherever the stream is reconfigurable.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    # In CI, we'd determine the PR scope from the branch name or changed files
    # For now, check all review files
    if not REVIEWS_DIR.exists():
        print(f"[FAIL] Reviews directory not found: {REVIEWS_DIR}")
        return 1

    review_files = list(REVIEWS_DIR.glob("*-review.md"))
    if not review_files:
        print("[FAIL] No adversarial review files found in rfc/reviews/")
        print("   Every PR must commit rfc/reviews/<scope>-review.md")
        return 1

    all_ok = True
    for review_file in review_files:
        print(f"Checking {review_file.name}...")
        ok, errors = check_review_file(review_file)
        if ok:
            print("  [OK] All findings resolved")
        else:
            print("  [FAIL] Issues found:")
            for err in errors:
                print(f"     - {err}")
            all_ok = False

    if all_ok:
        print("\n[OK] Adversarial gate passed - all reviews have resolved findings")
        return 0
    else:
        print("\n[FAIL] Adversarial gate failed - unresolved findings in review(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
