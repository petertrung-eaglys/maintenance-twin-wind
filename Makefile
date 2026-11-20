.PHONY: setup download inspect data ontology train wake alerts evaluate curves cross-site report app test lint

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
	$(NICE) uv run python -m src.models.wake
	$(NICE) uv run python -m src.models.pitch
	$(NICE) uv run python -m src.models.yaw
	$(NICE) uv run python -m src.models.thermal.train

# Wake-aware power alone (§7.13, also part of train); needs the physics
# power curves
wake:
	$(NICE) uv run python -m src.models.wake

alerts:
	$(NICE) uv run python -m src.health.alerts

evaluate:
	$(NICE) uv run python -m src.health.evaluate

curves:
	$(NICE) uv run python -m src.health.curves

cross-site:
	$(NICE) uv run python -m src.experiments.cross_site

report:
	$(NICE) uv run python -m src.report.build

app:
	uv run streamlit run app/app.py

test:
	uv run pytest

lint:
	uv run ruff check .
