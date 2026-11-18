.PHONY: setup download inspect data ontology train alerts evaluate test lint

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

# Training runs at low CPU priority (nice) so the desktop stays usable;
# threads are capped by compute.threads in config.yaml.
NICE = nice -n 10

train:
	$(NICE) uv run python -m src.models.power_baseline
	$(NICE) uv run python -m src.models.physics_power
	$(NICE) uv run python -m src.models.pitch
	$(NICE) uv run python -m src.models.yaw
	$(NICE) uv run python -m src.models.thermal.train

alerts:
	$(NICE) uv run python -m src.health.alerts

evaluate:
	$(NICE) uv run python -m src.health.evaluate

test:
	uv run pytest

lint:
	uv run ruff check .
