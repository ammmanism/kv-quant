$ErrorActionPreference = "Stop"

Write-Host "=== [1/8] test-cpu (pytest -m 'not gpu') ===" -ForegroundColor Cyan
uv run pytest -m "not gpu"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [2/8] lint & format (ruff) ===" -ForegroundColor Cyan
uv run ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
uv run ruff format --check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [3/8] typecheck (mypy --strict) ===" -ForegroundColor Cyan
uv run mypy kv_quant scripts tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [4/8] drift-check ===" -ForegroundColor Cyan
uv run python scripts/check_layout_drift.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [5/8] no-claims-without-artifact ===" -ForegroundColor Cyan
uv run python scripts/check_claims.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [6/8] figure-provenance ===" -ForegroundColor Cyan
uv run python scripts/check_figure_provenance.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [7/8] pr-governance & adversary check ===" -ForegroundColor Cyan
uv run python scripts/check_pr_governance.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [8/8] security SAST (bandit) ===" -ForegroundColor Cyan
uvx bandit -r kv_quant scripts -ll -ii
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[ALL 8 LOCAL GATES GREEN] Ready to ship PR." -ForegroundColor Green
