# Wind Farm Maintenance Digital Twin (PoC)

A proof of concept of a maintenance digital twin for two UK wind farms,
Kelmarsh (6 × Senvion MM92) and Penmanshiel (14 × Senvion MM82), using a
hybrid physics-ML method: a physics model states what a healthy turbine
should do, machine learning learns what the physics misses, and the gap
between the healthy twin and the measurement reveals component degradation
before the turbine raises a fault. See [docs/spec.md](docs/spec.md) for the
full specification.

**Status:** milestones §7.1 (scaffold), §7.2 (download), §7.3
(inspection), §7.4 (load, events, clean), §7.5 (ontology), §7.6–7.7
(modelling conventions, LightGBM baseline) and the power half of §7.8
(physics and hybrid power) done. Later milestones add their pipeline steps
and Makefile targets.

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

```bash
make data         # load_scada + events + clean -> data/processed/<farm>/
```

`make data` takes about 9 minutes, almost all of it `load_scada` parsing
the raw CSVs (the 2023/2024 exports with repeated years dominate); `events`
and `clean` take a few seconds. It writes `data/processed/<farm>/<T>.parquet`
per turbine and `data/processed/<farm>/events.parquet`.

```bash
make ontology     # build ontology/wind_twin.ttl (about 2 s)
```

The RDF graph `ontology/wind_twin.ttl` (committed) is the source of
structural truth: later steps and the app discover farms, turbines,
components, sensors and Parquet paths through the SPARQL queries in
`src/graph_utils.py`, not through `config.yaml`. `make ontology` rewrites
the whole file, so steps that add models or alerts must run again after it.

```bash
make train        # baseline, physics and hybrid power models (~50 s)
```

`make train` writes `expected_power_<variant>` into each turbine Parquet,
model files under `models/<farm>/` (LightGBM boosters, and the fitted Cp
curve per turbine type as `power_physics_<model>.json`), the model nodes
(inputs, target, metrics, `prov:generatedAtTime`) into the graph, and the
metrics into `models/metrics.json`.

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

## Results

**Data and cleaning (§7.4):**

| Farm | Turbines | Rows | Normal | Normal rows train / test |
|---|---|---|---|---|
| Kelmarsh | 6 | 2,839,104 (473,184 each) | 78.5 % (75.6–80.6 %) | 1,464,798 / 762,653 |
| Penmanshiel | 14 | 5,758,976 (386k–447k each) | 76.3 % (71.9–79.6 %) | 3,085,057 / 1,309,634 |

- Penmanshiel T01–T10 run from 2016-06-06 to the end of 2023 (no 2024
  data); T11–T15 to the end of 2024.
- Missing values: Kelmarsh 1.9–5.5 % for every signal. Penmanshiel
  2.2–4.8 %, except pitch, main-bearing and gear-oil temperatures at 25 %
  (no values before mid-2018 on any turbine).
- 2.5–7.1 % of rows per turbine fall inside a Stop event and are not
  normal. No row before a farm's commercial operations date is normal.
- `yaw_misalignment` (wind direction − nacelle position) is the nacelle
  vane's relative wind angle: on normal rows its median is within ±0.6°,
  its IQR about ±5°, and 99.6–100 % lie within ±30°. Wind direction is
  therefore not independent of the nacelle position; §7.8 decides on the
  yaw model with that in mind.
- `events.parquet`: 503,150 Kelmarsh and 1,053,535 Penmanshiel distinct
  status rows; labelled fault events match the §7.3 table (Kelmarsh
  gearbox 57, generator 2, pitch 56, yaw 26; Penmanshiel gearbox 654,
  generator 72, pitch 146, yaw 30).

**Graph (§7.5):** `ontology/wind_twin.ttl` holds 20,228 triples: the
schema (20 classes, 46 properties), 2 farms with their datasets, 20
turbines with specs, 7 components and 15 sensors each (all 300 sensors
matched to their farm's signal mapping), 1,043 fault events, and the
provenance of 429 source files (33 downloads, 198 turbine data files, 198
status files), each raw file `kt:partOf` the zip it was extracted from.

**Baseline power model (§7.7):** LightGBM per turbine, trained on normal
rows of 2016–2021, tested on normal rows of 2022–2024 (rated power
2,050 kW for every turbine; nRMSE = RMSE / rated power):

| Farm | Turbine | RMSE (kW) | nRMSE | MAE (kW) | Train rows | Test rows |
|---|---|---|---|---|---|---|
| KLM | T01 | 74.6 | 3.64 % | 50.1 | 245,216 | 130,349 |
| KLM | T02 | 73.5 | 3.59 % | 52.1 | 249,019 | 132,271 |
| KLM | T03 | 68.5 | 3.34 % | 48.8 | 243,584 | 129,538 |
| KLM | T04 | 93.4 | 4.55 % | 56.2 | 249,542 | 121,104 |
| KLM | T05 | 73.0 | 3.56 % | 51.3 | 241,994 | 126,964 |
| KLM | T06 | 65.7 | 3.20 % | 46.7 | 235,443 | 122,427 |
| PEN | T01 | 50.6 | 2.47 % | 35.1 | 222,905 | 80,367 |
| PEN | T02 | 57.5 | 2.81 % | 39.6 | 218,489 | 78,421 |
| PEN | T04 | 55.5 | 2.71 % | 38.9 | 221,197 | 75,875 |
| PEN | T05 | 57.4 | 2.80 % | 39.7 | 218,701 | 75,856 |
| PEN | T06 | 53.8 | 2.62 % | 37.6 | 215,018 | 73,612 |
| PEN | T07 | 52.5 | 2.56 % | 36.6 | 209,781 | 73,546 |
| PEN | T08 | 133.8 | 6.53 % | 82.7 | 219,439 | 79,029 |
| PEN | T09 | 53.8 | 2.62 % | 36.1 | 212,530 | 75,451 |
| PEN | T10 | 54.0 | 2.64 % | 37.4 | 211,743 | 69,523 |
| PEN | T11 | 61.0 | 2.98 % | 41.1 | 223,968 | 120,896 |
| PEN | T12 | 63.1 | 3.08 % | 43.6 | 228,034 | 126,893 |
| PEN | T13 | 51.4 | 2.51 % | 34.9 | 228,348 | 127,846 |
| PEN | T14 | 60.0 | 2.93 % | 42.2 | 226,950 | 126,738 |
| PEN | T15 | 61.4 | 2.99 % | 42.8 | 227,954 | 125,581 |

- Kelmarsh MAE (46.7–56.2 kW) matches the Kelmarsh repo (47.3–56.7 kW)
  despite the added monotone constraint and the stricter cleaning.
- **Penmanshiel T08** underperforms for months: its quarterly mean
  normal-row residual is −62 and −44 kW in H1 2021, −119 kW in Q4 2021 and
  about −193 kW in Q1–Q2 2022, then back within ±15 kW from Q3 2022. Part
  of the episode lies in the training period, so this healthy-only model
  learned from some degraded rows; it is a case study for §7.10–7.12.

**Physics and hybrid power models (§7.8):** the physics model is one
power-coefficient curve Cp(v_n) per turbine type (method of bins on
density-normalised wind speed, 0.5 m/s bins), giving P = ½ ρ₀ A Cp(v_n)
v_n³ with each turbine's rotor area, capped at rated power. Fitted
maximum Cp: **0.504 (MM92)** and **0.476 (MM82)**, both below the Betz
limit 0.593 (nacelle anemometers behind the rotor read low, which raises
apparent Cp). The hybrid adds a per-turbine LightGBM model of the physics
residual. Test nRMSE (RMSE / rated power):

| Farm | Turbine | Baseline | Physics | Hybrid |
|---|---|---|---|---|
| KLM | T01 | 3.64 % | 3.75 % | 3.65 % |
| KLM | T02 | 3.59 % | 3.69 % | 3.58 % |
| KLM | T03 | 3.34 % | 3.62 % | 3.31 % |
| KLM | T04 | 4.55 % | 4.56 % | 4.54 % |
| KLM | T05 | 3.56 % | 3.87 % | 3.54 % |
| KLM | T06 | 3.20 % | 3.26 % | 3.22 % |
| PEN | T01 | 2.47 % | 2.87 % | 2.76 % |
| PEN | T02 | 2.81 % | 3.22 % | 3.16 % |
| PEN | T04 | 2.71 % | 3.21 % | 3.01 % |
| PEN | T05 | 2.80 % | 3.45 % | 3.13 % |
| PEN | T06 | 2.62 % | 3.07 % | 2.83 % |
| PEN | T07 | 2.56 % | 2.91 % | 2.56 % |
| PEN | T08 | 6.53 % | 6.62 % | 6.60 % |
| PEN | T09 | 2.62 % | 2.97 % | 2.63 % |
| PEN | T10 | 2.64 % | 3.42 % | 2.60 % |
| PEN | T11 | 2.98 % | 3.58 % | 2.91 % |
| PEN | T12 | 3.08 % | 3.54 % | 3.19 % |
| PEN | T13 | 2.51 % | 2.91 % | 2.83 % |
| PEN | T14 | 2.93 % | 3.44 % | 3.14 % |
| PEN | T15 | 2.99 % | 3.85 % | 3.22 % |
| **Mean** | KLM | 3.65 % | 3.79 % | 3.64 % |
| **Mean** | PEN | 3.02 % | 3.50 % | 3.18 % |

- One physics curve per turbine type is within 0.1–0.8 percentage points
  of a per-turbine LightGBM model. The hybrid beats physics on all 20
  turbines and matches the baseline (better on 6, within 0.3 points
  elsewhere). Within one site the baseline is hard to beat; the physics
  variants are built for transfer (§7.11).

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
- Cleaning (§7.4):
  - `is_normal` = complete required signals, wind speed 3–25 m/s, power
    > 0, pitch < 5° below rated wind speed, **on or after the commercial
    operations date** from the static file, and **no Stop event active**
    (any Stop status, not only labelled faults: manual stops, grid loss
    and tests are not normal operation either). A row counts as stopped
    when its 10-minute interval meets the event; a Stop without an end
    marks only the row containing its start.
  - A missing pitch passes the pitch rule (Penmanshiel has no pitch
    before mid-2018; rejecting those rows would drop two years of power
    data). Those rows cannot be checked for curtailment.
  - Rated wind speed per turbine model: MM92 12.0 m/s (Kelmarsh repo),
    MM82 14.0 m/s estimated from 2019–2021 data (no manufacturer value at
    hand): the first 0.5 m/s bin with median power ≥ 98 % of rated is
    14–15 m/s, and the median pitch there is 7°, as at the MM92's 12 m/s.
    The same 99 %-of-rated method gives exactly 12.0 m/s for the MM92.
  - No power-curve PNGs are written (the Kelmarsh repo did); plots come
    with the report (§7.12).
- Events (§7.4): `events.parquet` keeps every status (including
  Informational), de-duplicated on turbine, start, status, code and
  message; `component` is set only for Stop/Warning rows.
- New `src/static_data.py` reads the static file (drops empty rows,
  day-first dates, turbine ids from titles). `config.yaml` gained
  `farms.*.static_file`, `sampling`, `cleaning` (with `required` and
  per-model `rated_wind_speed`) and `events.stop_status`.
- Ontology (§7.5):
  - Namespace `https://example.org/wind-twin#` (prefix `kt:`). Besides
    the spec's components (`Gearbox`, `Generator`, `MainBearing`,
    `PitchSystem`, `YawSystem`), every turbine has a `Nacelle` (wind,
    ambient and nacelle-temperature sensors) and a `Rotor` (rotor speed),
    so every sensor monitors a component. `nacelle_direction` monitors
    the yaw system, `power` the generator.
  - `kt:Sensor` is a subclass of `sosa:Sensor`; sensors `sosa:observes`
    an observable property and carry a QUDT unit, as in the Kelmarsh repo.
  - Extra properties: `kt:farmCode`, `kt:turbineId`, `kt:componentKey`
    (every lookup binds farm code and turbine id), turbine specs,
    `kt:manufacturerTitle`/`kt:greenbyteSignalId` from the mapping, and
    for fault events `kt:affects` (event → component), `kt:startTime`,
    `kt:endTime`, `kt:eventStatus`, `kt:eventCode`, `kt:eventMessage`,
    `kt:iecCategory`. `kt:startTime`/`kt:endTime` have domain
    `rdfs:Resource` because alerts will use them too.
  - Provenance: farm `prov:wasDerivedFrom` its `kt:Dataset`; turbines
    derive from the static file; sensors from their raw data files and the
    mapping file; fault events from their status file. Each raw file is
    `kt:partOf` the zip that contains it, read from the zip member lists
    (several Penmanshiel zips share a year folder, so the folder alone
    does not identify the zip).
  - No model nodes yet: each training milestone adds its `kt:Model` nodes
    (the Kelmarsh repo created power-model stubs in `build_ontology`).
  - The lineage query leaves out fault events (hundreds on some
    turbines); `graph_utils.get_fault_events` returns them.
  - `events.parquet` gained a `source_file` column (the status file) for
    event provenance. `config.yaml` gained `farms.*.mapping_file`;
    `static_data.read_mapping` gives both farms' mapping tables the same
    column names.
- Modelling (§7.6–7.7):
  - Baseline features follow spec §7.7 (wind speed, wind direction
    sin/cos, ambient temperature, air density); the Kelmarsh repo also
    used the raw wind direction and not the ambient temperature. The
    model's sensor inputs in the graph are `wind_speed`, `wind_direction`
    and `ambient_temp`; the other features derive from them.
  - The baseline has a LightGBM monotone constraint on wind speed (spec
    §7.8 asks for it "in the ML-only variant as well").
  - Metrics are RMSE, MAE and nRMSE (RMSE / rated power for power models),
    stored per model in `models/metrics.json` keyed by the model's IRI
    local name (`PowerModel_KLM_T01_baseline`), which merges with entries
    from other scripts. The graph schema gained `kt:rmse`, `kt:mae`,
    `kt:nrmse`, `kt:nTrain`, `kt:nTest`.
  - Power models are `kt:modelOf` the generator component (the power
    sensor monitors the generator). Model IRIs carry no component for
    power models: `kt:PowerModel_KLM_T01_baseline`.
  - Predictions are written to the turbine Parquet as
    `expected_<target>_<variant>` (`expected_power_baseline`).
  - Train/test rows also require the model's features and target to be
    present (`split_masks(frame, train_end, required)`).
  - `scikit-learn` added: LightGBM's scikit-learn wrapper needs it.
- Physics and hybrid power (§7.8):
  - The physics curve is Cp(v_n) (power per rotor area / ½ ρ₀ v_n³ per
    bin), not P_bin(v_n), so it carries over to another rotor size. Cp is
    fitted only on the bins below the first one whose mean power reaches
    95 % of rated (`physics.max_power_fraction`); above that the
    controller, not aerodynamics, limits power, and keeping those bins
    under-predicted a smaller rotor by about 400 kW in a synthetic test.
    Cp is held flat past the last bin and the rated cap applies, so the
    model has a sharp knee at rated power, which adds error near rated.
    The curves stop at 10.7 m/s (MM92) and 12.7 m/s (MM82).
  - The physics model is fitted per turbine type through
    `fit(train_frames)` with all turbines of that type in the farm; the
    hybrid residual model is per turbine.
  - Turbine specs (rotor diameter, rated power) come from the graph and are
    added to each frame as constant columns
    (`modelling.with_turbine_specs`), so `predict(frame)` needs nothing
    else.
  - The hybrid residual model has no monotone constraint (its target is
    a residual, not power); the hybrid output is clipped to [0, rated].
  - `config.yaml` gained `physics` (reference density, bins, cut-in and
    cut-out, `max_power_fraction`) and `models.power_physics`,
    `models.power_hybrid`. `scipy` added explicitly (approved; it was
    already installed through LightGBM).
  - Shared registration moved to `modelling.py` (`register_model`,
    `metrics_entry`, `model_id_of`); `src/binned.py` holds the method of
    bins for reuse by the pitch model.
  - Pitch and yaw models (also §7.8) follow in a separate commit.
- `file_columns`, `read_signals` and `read_status` moved from
  `inspect_data.py` to `scada_format.py` so the pipeline does not import
  the inspection script.
- `inspect_data.py` also prints, per farm, how many files lack each
  configured column and the share of timestamps with a value per signal
  (first turbine file of each year), to decide decision 2 on real values
  rather than headers.
