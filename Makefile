# kv-quant Makefile
# Targets for CI gates and development workflow

.PHONY: verify test lint typecheck drift-check figure-check kernel-parity golden-compare no-claims-without-artifact charts clean help

# Default target
help:
	@echo "kv-quant Makefile targets:"
	@echo "  verify                    - Run all CI gates"
	@echo "  test                      - Run unit tests (pytest)"
	@echo "  lint                      - Run ruff linter"
	@echo "  typecheck                 - Run mypy type checker"
	@echo "  drift-check               - Check layout drift"
	@echo "  figure-check              - Check figure provenance"
	@echo "  kernel-parity             - Run kernel parity tests"
	@echo "  golden-compare            - Compare against golden baseline"
	@echo "  no-claims-without-artifact - Check claims have CSV artifacts"
	@echo "  charts                    - Render documentation charts"
	@echo "  clean                     - Clean build artifacts"

# Run all CI verification gates
verify: test lint typecheck drift-check figure-check kernel-parity golden-compare no-claims-without-artifact
	@echo "✅ All verification gates passed"

# Unit tests
test:
	uv run pytest tests/ -x -q

# Linting
lint:
	uv run ruff check .

# Type checking
typecheck:
	uv run mypy .

# Layout drift check (Invariant 3)
drift-check:
	uv run python scripts/check_layout_drift.py

# Figure provenance check (Invariant 2)
figure-check:
	uv run python scripts/check_figure_provenance.py

# Kernel parity (Invariant 6)
kernel-parity:
	uv run pytest tests/test_parity.py -x

# Golden curve compare (Invariant 7)
golden-compare:
	uv run python scripts/check_golden_curve.py

# Claims check (Invariant 1) + Adversarial review (Invariant 9)
no-claims-without-artifact:
	uv run python scripts/check_claims.py
	uv run python scripts/check_adversarial_review.py

# Render charts for documentation
charts:
	uv run python scripts/render_charts.py

# Clean build artifacts
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf kv_quant/__pycache__ kv_quant/kernels/__pycache__
	rm -rf tests/__pycache__ bench/__pycache__ scripts/__pycache__
	rm -rf bench/data/tmp
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Install dependencies
install:
	uv sync --dev

# Update dependencies
update:
	uv lock --upgrade

# Format code
format:
	uv run ruff format .
	uv run ruff check . --fix