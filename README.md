# Wind Farm Maintenance Digital Twin (PoC)

A proof of concept of a maintenance digital twin for two UK wind farms,
Kelmarsh (6 × Senvion MM92) and Penmanshiel (14 × Senvion MM82), using a
hybrid physics-ML method: a physics model states what a healthy turbine
should do, machine learning learns what the physics misses, and the gap
between the healthy twin and the measurement reveals component degradation
before the turbine raises a fault. See [docs/spec.md](docs/spec.md) for the
full specification.

**Status:** milestones §7.1 (scaffold) and §7.2 (download) done. Later
milestones add their pipeline steps and Makefile targets.

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

```bash
make download     # fetch into data/raw/<farm>/ (MD5-verified, skips), unzip
```

The download is about 3.7 GB (Kelmarsh) plus 8.4 GB (Penmanshiel) and
unzips to about 43 GB plus 62 GB. Each farm's SCADA archives unzip into
`data/raw/<farm>/scada_<year>/`. Re-running downloads and extracts nothing
(about 10 s for the MD5 checks).

To reuse raw files from the Kelmarsh repo instead of downloading them again,
hard-link them before the first run (same filesystem, no extra space):

```bash
mkdir -p data/raw && cp -al ~/digital-twin-wind-farm/data/raw data/raw/kelmarsh
```

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
- Kelmarsh raw files were hard-linked from the Kelmarsh repo
  (`cp -al`) rather than downloaded; `download.py` verifies their MD5s and
  skips them like any valid file.
- Penmanshiel archive layout (§3.2 open points), confirmed after download:
  - Every zip unzips flat (no subfolders); no file name appears in two
    zips of the same year, so the shared `scada_<year>/` folders are safe.
  - Turbine numbers are zero-padded in file names
    (`Turbine_Data_Penmanshiel_01_…`, `Status_Penmanshiel_01_…`), unlike
    Kelmarsh (`Kelmarsh_1`).
  - The WT01–10 zips hold 9 turbines (no T03): 9 data and 9 status files.
  - 2023, WT01–10: four period files per turbine (Feb, Mar, Q2, Q3–Q4), so
    `scada_2023/` holds 41 turbine data files (9 × 4 + 5).
  - 2024 (WT11–15 only): about 3.8 GB per turbine file, which suggests
    repeated years as in Kelmarsh 2023/2024 (to be confirmed in §7.3).
  - Unzipped size is about 62 GB, not the ~100 GB estimated in §3.1.
