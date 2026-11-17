.PHONY: setup download test lint

setup:
	uv sync

download:
	uv run python -m src.download

test:
	uv run pytest

lint:
	uv run ruff check .
