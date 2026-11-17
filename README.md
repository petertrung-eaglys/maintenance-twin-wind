# Wind Farm Maintenance Digital Twin (PoC)

A proof of concept of a maintenance digital twin for two UK wind farms,
Kelmarsh (6 × Senvion MM92) and Penmanshiel (14 × Senvion MM82), using a
hybrid physics-ML method: a physics model states what a healthy turbine
should do, machine learning learns what the physics misses, and the gap
between the healthy twin and the measurement reveals component degradation
before the turbine raises a fault. See [docs/spec.md](docs/spec.md) for the
full specification.

**Status:** milestone §7.1 (scaffold) done. Later milestones add their
pipeline steps and Makefile targets.

Code is derived from the Kelmarsh power-performance twin in
`~/digital-twin-wind-farm`, tag **`v1.0-kelmarsh`** (copied and adapted, not
a dependency).

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```bash
make setup        # uv sync
```

All configuration (paths, download URLs and MD5s, column names, model
settings, thresholds) lives in [config.yaml](config.yaml).

## Run

Every pipeline script runs standalone with `uv run python -m src.<name>`
and has a Makefile target (added per milestone).

## Testing and linting

```bash
make test         # uv run pytest
make lint         # uv run ruff check .
```

## Attribution

- Kelmarsh wind farm data by Cubico Sustainable Investments Ltd, licensed
  under CC-BY-4.0. DOI:
  [10.5281/zenodo.16807551](https://doi.org/10.5281/zenodo.16807551).
- Penmanshiel wind farm data by Cubico Sustainable Investments Ltd, licensed
  under CC-BY-4.0. DOI:
  [10.5281/zenodo.16807304](https://doi.org/10.5281/zenodo.16807304).

## Deviations from spec

- `config.yaml` has no `downloads.files` level: each farm's `downloads` is
  the list itself (as in the spec §6 shape). SCADA archives are the entries
  with `extract_to`; several Penmanshiel archives share one `scada_<year>`
  folder.
- Makefile targets and dependencies are added in the milestone that first
  needs them, so no target points at a script that does not exist yet.
