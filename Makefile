SHELL := /bin/bash

.PHONY: setup play arena zip gate

setup:
	uv sync

play:
	uv run python -m harness.play --white submissions/v44-volatility-pvs-candidate --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --agent submissions/v44-volatility-pvs-candidate --opponent baselines/greedy --games 20

zip:
	uv run python -c "from pathlib import Path; from harness.package import build; print(build(Path('submissions/v44-volatility-pvs-candidate'), Path('submission.zip'), ()))"

gate:
	uv run ruff check submissions/v44-volatility-pvs-candidate harness baselines
	uv run mypy
	uv run python -m harness.arena --agent submissions/v44-volatility-pvs-candidate --opponent baselines/random --games 2 --base-ms 5000
