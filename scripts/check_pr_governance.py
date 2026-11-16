"""CI Gate: Enforce Conventional Commit PR titles and resolved Adversary reviews."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# Summary length widened to 100 so milestone titles (e.g. the 3-tier cache PR)
# fit the Conventional Commits shape without truncation.
CONVENTIONAL_PR_RE = re.compile(
    r"^(feat|fix|docs|chore|bench|refactor|test|ci|perf)\([a-z0-9\-]+\):\s+.{10,100}$"
)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    pr_title = os.environ.get("PR_TITLE", "").strip()
    pr_body = os.environ.get("PR_BODY", "").strip()

    errors: list[str] = []

    # 1. Validate PR Title format if running inside a GitHub PR event
    if pr_title:
        if not CONVENTIONAL_PR_RE.match(pr_title):
            errors.append(
                f"PR title '{pr_title}' must match '<type>(<scope>): <imperative summary>' "
                "(10-100 chars, lowercase)."
            )
        if len(pr_body) < 40:
            errors.append("PR body is too short; fill out .github/PULL_REQUEST_TEMPLATE.md.")

    # 2. Verify at least one review file exists in rfc/reviews/ and has zero unresolved BLOCKING tags
    review_dir = root / "rfc" / "reviews"
    reviews = sorted(review_dir.glob("*-review.md")) if review_dir.exists() else []
    if not reviews:
        errors.append("No adversarial review file found in rfc/reviews/*-review.md.")
    else:
        for rev in reviews:
            text = rev.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if "[BLOCKING]" in line and "RESOLVED" not in line:
                    errors.append(
                        f"{rev.name}:{line_no} has an unresolved [BLOCKING] adversary finding."
                    )

    # 3. Verify branch changed files include an updated or new review in PR context
    base_ref = os.environ.get("GITHUB_BASE_REF", "")
    if base_ref:
        diff = subprocess.check_output(
            ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"],
            text=True,
        ).splitlines()
        if not any(p.startswith("rfc/reviews/") and p.endswith("-review.md") for p in diff):
            errors.append(
                "This PR did not add or update an adversary review in rfc/reviews/<scope>-review.md."
            )

    if errors:
        for err in errors:
            print(f"[FAIL: pr-governance] {err}", file=sys.stderr)
        return 1

    print("[PASS: pr-governance] Conventional title and adversarial review gate verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
