.PHONY: setup download inspect data ontology train test lint

setup:
	uv sync

download:
	uv run python -m src.data.download

inspect:
	uv run python -m src.data.inspect_data

data:
	uv run python -m src.data.load_scada
	uv run python -m src.data.events
	uv run python -m src.data.clean

ontology:
	uv run python -m src.graph.build_ontology

train:
	uv run python -m src.models.power_baseline
	uv run python -m src.models.physics_power
	uv run python -m src.models.pitch
	uv run python -m src.models.yaw

test:
	uv run pytest

lint:
	uv run ruff check .
