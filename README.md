# Wind Farm Maintenance Digital Twin (PoC)

A proof of concept of a maintenance digital twin for two UK wind farms,
Kelmarsh (6 × Senvion MM92) and Penmanshiel (14 × Senvion MM82), using a
hybrid physics-ML method: a physics model states what a healthy turbine
should do, machine learning learns what the physics misses, and the gap
between the healthy twin and the measurement reveals component degradation
before the turbine raises a fault. See [docs/spec.md](docs/spec.md) for the
full specification.

**Status:** milestones §7.1 (scaffold), §7.2 (download) and §7.3
(inspection) done. Later milestones add their pipeline steps and Makefile
targets.

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

```bash
make inspect      # print raw file structure, column coverage and events
```

`make inspect` takes about a minute (it reads the configured columns of one
turbine file per year, including the multi-GB 2023/2024 exports).

Every pipeline script runs standalone with `uv run python -m src.<name>`
and has a Makefile target (added per milestone).

## Data inspection decisions (§7.3)

From `make inspect` on both farms:

1. **Column names:** the 7 core and 8 component columns guessed in spec
   §3.4 exist with exactly these names in all 54 Kelmarsh and 144
   Penmanshiel turbine data files.
2. **Component signals:** all 8 exist in both farms, so thermal models
   (§7.9) can cover every component in both:
   - main bearing: `front_bearing_temp`, `rear_bearing_temp` (manufacturer
     title "Axle bearing temperature 1/2");
   - gearbox: `gear_oil_temp`, `gear_oil_inlet_temp`;
   - generator: `generator_bearing_front_temp`,
     `generator_bearing_rear_temp`, `stator_temp`;
   - reference: `nacelle_temp`.

   Filled share of timestamps (first turbine file per year): Kelmarsh
   95–100 % from 2017 (66 % in 2016, before commissioning on 15 April).
   Penmanshiel T01 has **no values for pitch, main-bearing and gear-oil
   temperatures in 2016–2017 and 63 % in 2018**; everything else is
   94–100 %. Penmanshiel pitch, main-bearing and gearbox models therefore
   train on mid-2018 to 2021 only.
3. **Fault labels:** `config.yaml` `fault_labels` maps 35 exact
   Stop/Warning messages to `gearbox`, `generator`, `pitch` and `yaw`
   (sensor and communication defects, routine operation, generator fan
   overloads and the very frequent "High yaw motor current" are left out).
   No message concerns the main bearing, so main-bearing alerts can only be
   shown as case studies, not validated. Distinct mapped events:

   | Farm | Component | Train (2016–2021) | Test (2022–2024) | Turbines with a test event |
   |---|---|---|---|---|
   | Kelmarsh | gearbox | 30 | 27 | 4 |
   | Kelmarsh | generator | 2 | 0 | 0 |
   | Kelmarsh | pitch | 41 | 15 | 4 |
   | Kelmarsh | yaw | 21 | 5 | 4 |
   | Penmanshiel | gearbox | 366 | 288 | 13 |
   | Penmanshiel | generator | 45 | 27 | 4 |
   | Penmanshiel | pitch | 103 | 43 | 11 |
   | Penmanshiel | yaw | 25 | 5 | 3 |

   Many Penmanshiel gearbox events are repeated "Missing gear oil (high
   rpm)" stops (354 in total), so evaluation (§7.10) has to merge
   back-to-back events.
4. **Penmanshiel layout:** see "Deviations from spec" below.

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
  - 2024 (WT11–15 only): about 3.8 GB per turbine file. Confirmed in §7.3:
    the year is repeated as in Kelmarsh 2023/2024 (T11: 2.45 million rows
    for 52,704 timestamps). The 2023 period files repeat too (T01 February:
    20,076 rows for 4,032 timestamps), so the "most complete row per
    timestamp" rule applies to Penmanshiel as well.
  - Unzipped size is about 62 GB, not the ~100 GB estimated in §3.1.
  - Penmanshiel data starts on 2016-06-06, before the commercial operations
    date 2016-09-01 in the static file (Kelmarsh: 15 April 2016).
- Static and mapping tables (§7.3):
  - `Penmanshiel_WT_static.csv` has 14 turbine rows followed by 15 empty
    rows and an empty trailing column; readers must drop them. Hub height
    is 59 m (Kelmarsh 78.5 m).
  - The mapping tables differ in columns: Kelmarsh CSV `Greenbyte Signal
    ID, Greenbyte Title, Manufacturer Title, Unit`; Penmanshiel xlsx (one
    sheet) `DataSignalId, TitleGreenbyte, TitleManufacturer, Unit,
    MinValue, MaxValue`.
- Status files (§3.3): the header ends at `IEC category`; the `Global
  contract category` and `Custom contract category` columns listed in the
  spec do not exist in either farm. Exports repeat events, so counts are
  of distinct (turbine, start, status, code, message).
- `config.yaml` `fault_labels` maps component → list of exact messages
  rather than message patterns: exact names keep sensor-defect messages
  with similar wording (e.g. "PT100 inlet gear defect") out.
- `config.yaml` gained `events.fault_statuses` (`Stop`, `Warning`).
- `inspect_data.py` also prints, per farm, how many files lack each
  configured column and the share of timestamps with a value per signal
  (first turbine file of each year), to decide decision 2 on real values
  rather than headers.
