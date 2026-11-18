.PHONY: setup download inspect test lint

setup:
	uv sync

download:
	uv run python -m src.download

inspect:
	uv run python -m src.inspect_data

test:
	uv run pytest

lint:
	uv run ruff check .
