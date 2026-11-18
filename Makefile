.PHONY: setup download inspect data test lint

setup:
	uv sync

download:
	uv run python -m src.download

inspect:
	uv run python -m src.inspect_data

data:
	uv run python -m src.load_scada
	uv run python -m src.events
	uv run python -m src.clean

test:
	uv run pytest

lint:
	uv run ruff check .
