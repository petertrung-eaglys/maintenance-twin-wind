.PHONY: setup download inspect data ontology train test lint

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

train:
	uv run python -m src.power_baseline
	uv run python -m src.physics_power
	uv run python -m src.pitch
	uv run python -m src.yaw

test:
	uv run pytest

lint:
	uv run ruff check .
