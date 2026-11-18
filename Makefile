.PHONY: setup download inspect data ontology test lint

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

ontology:
	uv run python -m src.build_ontology

test:
	uv run pytest

lint:
	uv run ruff check .
