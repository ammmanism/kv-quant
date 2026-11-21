"""Explorer gates: manifest verification, build determinism, drift, snippet.

The snippet emitted by the interactive explorer is executed against the
real QuantConfig dataclass so the one-click output can never stop compiling.
The headless UI drive runs via node (scripts/explorer_smoke.js) when
available and is skipped otherwise.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_interactive_explorer import build_html  # noqa: E402
from check_explorer_drift import EXTERNAL_PATTERNS, MAX_BYTES  # noqa: E402
from explorer_data import verified_artifacts  # noqa: E402

ROOT = Path(__file__).parent.parent
EXPLORER = ROOT / "docs" / "explorer.html"


def test_all_tracked_csvs_verify_against_manifests() -> None:
    artifacts = verified_artifacts()
    assert len(artifacts) == 19
    for art in artifacts:
        assert len(art["sha256"]) == 64  # type: ignore[arg-type]
        assert art["mode"], art
        assert art["rows"], art


def test_tampered_csv_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    import explorer_data

    real = explorer_data.csv_sha256

    def forged(path: Path) -> str:
        return "0" * 64 if path.name.endswith("p13_ragged_concurrency.csv") else real(path)

    monkeypatch.setattr(explorer_data, "csv_sha256", forged)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        explorer_data.verified_artifacts()


def test_build_is_deterministic() -> None:
    assert build_html() == build_html()


def test_committed_explorer_matches_rebuild() -> None:
    assert EXPLORER.exists(), "docs/explorer.html missing; run build_interactive_explorer.py"
    committed = EXPLORER.read_text(encoding="utf-8")
    assert committed == build_html(), "explorer drifted from bench/data/*.csv; regenerate it"


def test_no_external_resources_and_size_budget() -> None:
    import re

    text = EXPLORER.read_text(encoding="utf-8")
    for pattern in EXTERNAL_PATTERNS:
        assert not re.findall(pattern, text, flags=re.IGNORECASE), f"external resource: {pattern}"
    assert len(text.encode()) <= MAX_BYTES


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_all_snippet_branches_compile_against_real_config() -> None:
    import re

    smoke = subprocess.run(
        ["node", str(ROOT / "scripts" / "explorer_smoke.js")],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert "SMOKE OK" in smoke.stdout
    snippets = dict(re.findall(r"SNIPPET-BEGIN (\S+)\n(.*?)\nSNIPPET-END", smoke.stdout, re.S))
    assert set(snippets) == {"v1", "v2-density", "v2-quality", "agent"}

    expected = {
        "v1": (8, 4, "none"),
        "v2-density": (4, 4, "block_diag"),
        "v2-quality": (8, 4, "block_diag"),
        "agent": (4, 8, "none"),
    }
    for name, snippet in snippets.items():
        namespace: dict[str, object] = {}
        exec(snippet, namespace)  # noqa: S102 - the point of the test
        cfg = namespace["cfg"]
        k_bits, v_bits, scope = expected[name]
        assert (cfg.k_bits, cfg.v_bits, cfg.hadamard_scope) == (  # type: ignore[attr-defined]
            k_bits,
            v_bits,
            scope,
        ), name
        assert cfg.sink_tokens == 4 and cfg.group_size == 64  # type: ignore[attr-defined]
        # Fleet sizing is a serving-side concern and must stay OUT of the kwargs.
        assert "CacheConfig(device=" in snippet
    assert "p13: k8v4_cow fits 109 sessions" in snippets["v1"]
    assert "resolve_hadamard_scope() reports 'none'" in snippets["v2-density"]
