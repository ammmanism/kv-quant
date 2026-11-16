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
    r"Won't do — .+",
    r"Question",
]


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
                    resolution = res_match.group(1)
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
        errors.append("No findings found in review file (expected format: '- [ ] **Finding**: ...' followed by '- **Resolution**: ...')")
        return False, errors

    unresolved = 0
    for checked, finding, resolution in findings:
        if checked != "x":
            unresolved += 1
            errors.append(f"Unresolved finding: {finding}")
        else:
            # Verify resolution marker exists in resolution text
            has_resolution = any(re.search(p, resolution) for p in RESOLUTION_PATTERNS)
            if not has_resolution:
                errors.append(f"Resolved finding missing resolution marker: {finding} | Resolution: {resolution}")

    return unresolved == 0 and len(errors) == 0, errors


def main() -> int:
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
