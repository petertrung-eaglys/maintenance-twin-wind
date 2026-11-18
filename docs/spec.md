# Wind Farm Maintenance Digital Twin PoC: Specification

## 1. Goal

Build a proof of concept of a **maintenance digital twin** for two UK wind
farms, Kelmarsh and Penmanshiel, using a **hybrid physics-ML** method. The
PoC is for an internal demo and customer pitches, so it must tell one clear
story:

> A physics model states what a healthy turbine should do; machine learning
> learns what the physics misses; the gap between the healthy twin and the
> measurement reveals degradation before the turbine raises a fault.

The PoC must show the full chain in one working demo:

1. Load and clean real SCADA data and turbine event logs for both farms.
2. Describe farms, turbines, components, sensors, models, alerts and fault
   events in an RDF ontology.
3. Build **normal-behaviour models** per component (power, drivetrain
   temperatures, pitch/yaw) as physics, ML and hybrid physics-ML variants.
4. Turn model residuals into component health alerts.
5. Validate alerts against the real fault events in the status logs
   (detection rate and lead time).
6. Compare the variants against a LightGBM baseline, including a cross-site
   experiment (train on one farm, test on the other).
7. Show everything in a Streamlit dashboard with historical replay and a
   component health view, plus a written report.

The PoC favours a working end-to-end slice over accuracy or completeness.
Federated learning is **future work** (§10), but the training code must be
shaped so it can be added without redesign (§7.9).

## 2. Background

This project follows the Kelmarsh power-performance twin in the sibling
repository `~/digital-twin-wind-farm` (tag `v1.0-kelmarsh`). That repo
proved the data pipeline, the RDF-graph-as-structural-truth design and the
replay dashboard for Kelmarsh alone. Its code may be copied and adapted
(§7.1 lists what to reuse); it must not become a dependency. Record the
source tag in the README when copying.

## 3. Data sources

Both datasets are published by Cubico Sustainable Investments Ltd under
CC-BY-4.0. Attribution (publisher, DOI, licence) must appear in the README
and in the dashboard footer. Both are Greenbyte SCADA exports with 10-minute
data in UTC.

| Farm | Turbines | Rotor | Rated power | Zenodo record | DOI |
|---|---|---|---|---|---|
| Kelmarsh | 6 × Senvion MM92 (T01–T06) | 92 m | 2.05 MW | 16807551 | 10.5281/zenodo.16807551 |
| Penmanshiel | 14 × Senvion MM82 (T01–T15, **no T03**) | 82 m | 2.05 MW | 16807304 (v3, 2025-08-13) | 10.5281/zenodo.16807304 |

Same rated power with different rotor sizes is deliberate: it is what makes
the cross-site experiment (§7.11) meaningful. Do **not** use the older
Penmanshiel record 5946808; it stops in mid-2021.

### 3.1 Files

URLs follow `https://zenodo.org/records/<record>/files/<name>?download=1`
(URL-encode spaces as `%20`). MD5s were read from the Zenodo API on
2026-11-17.

**Kelmarsh (record 16807551)**, about 3.7 GB zipped, about 43 GB unzipped:

| File | MD5 |
|---|---|
| `Kelmarsh_WT_static.csv` | af3a038f0f7fddfc1608ad0bfc8cf5ba |
| `Kelmarsh_WT_dataSignalMapping.csv` | ca8fe399ab15ae7111d3214c632efd36 |
| `Kelmarsh_SCADA_2016_3082.zip` | ca8efbb6dbcfae3516583937eb5b5402 |
| `Kelmarsh_SCADA_2017_3083.zip` | c78263ee52ee0e48e2cb4bbaa1ba211a |
| `Kelmarsh_SCADA_2018_3084.zip` | 972e5280172216a5fe76dcab19c741b0 |
| `Kelmarsh_SCADA_2019_3085.zip` | 14348a1572b722e1be10e53cd145c39a |
| `Kelmarsh_SCADA_2020_3086.zip` | e189e02da98f32e74acc72e8559ef0db |
| `Kelmarsh_SCADA_2021_4456.zip` | 4e0ef3d65f067fd3d8f3d30e79ab01ae |
| `Kelmarsh_SCADA_2022_4457.zip` | b695d2b21cd5a236b2e690e0a43df5ab |
| `Kelmarsh_SCADA_2023_5961.zip` | 685d9589347d1866f6f2cfe45b9af4f4 |
| `Kelmarsh_SCADA_2024_5962.zip` | e9e3318f5b393d8596ff1801f8da0414 |

**Penmanshiel (record 16807304)**, about 8.4 GB zipped (unzipped size
unknown, expect on the order of 100 GB):

| File | MD5 |
|---|---|
| `Penmanshiel_WT_static.csv` | c4cd4191234c1a67a391fe5d2978256b |
| `Penmanshiel_WT_dataSignalMapping.xlsx` | 2eae1b84a014619268a6e4f53e6b1c91 |
| `Penmanshiel_SCADA_2016_WT01-10_3107.zip` | 3819e1b2168b84da195b64e1e9d78b03 |
| `Penmanshiel_SCADA_2016_WT11-15_3107.zip` | 92c30453d43a38e985af3e2e305eedd4 |
| `Penmanshiel_SCADA_2017_WT01-10_3114.zip` | c8279e34a5e33d688e2219aa9abab74e |
| `Penmanshiel_SCADA_2017_WT11-15_3115.zip` | d41874e3d195e88506949492b5758cf6 |
| `Penmanshiel_SCADA_2018_WT01-10_3113.zip` | 6389e5c4ea59d6565f6712db43a504ce |
| `Penmanshiel_SCADA_2018_WT11-15_3116.zip` | 7c40dc00df6d6301427213e15170b927 |
| `Penmanshiel_SCADA_2019_WT01-10_3112.zip` | 78389603cb4ae20a6a9b878f5a84bc7b |
| `Penmanshiel_SCADA_2019_WT11-15_3117.zip` | 92747a976716c5f35d9ee6462958fd7a |
| `Penmanshiel_SCADA_2020_WT01-10_3109.zip` | 46f679f2e4ab3d69c04b31c8dc625106 |
| `Penmanshiel_SCADA_2020_WT11-15_3118.zip` | c73323f21451c08247dbdb5d1a1434a9 |
| `Penmanshiel_SCADA_2021_WT01-10_4460.zip` | 7a0839cf3cd5e7044d6ab814fef3c43f |
| `Penmanshiel_SCADA_2021_WT11-15_4461.zip` | f5804b90089ce4d50dc5117fda6f675c |
| `Penmanshiel_SCADA_2022_WT01-10_4462.zip` | 91351469cdeb3ca95be1a0414dcfec8f |
| `Penmanshiel_SCADA_2022_WT11-15_4463.zip` | 5f4743133a9cf65ad33de139b43f96b4 |
| `Penmanshiel_SCADA_2023_02_WT_01-10_5981.zip` | 05ac6e90b516fa30db05dfe6361099a7 |
| `Penmanshiel_SCADA_2023_03_WT_01-10_5982.zip` | 6fae49a3603dd2a503126e5e518eaf55 |
| `Penmanshiel_SCADA_2023_Q2_WT_01-10_5977.zip` | 7ed1d095183469e399054695b8c4fb5e |
| `Penmanshiel_SCADA_2023_Q3 Q4_WT_01-10_5973.zip` | f201240d4072dd4a5a8492725eb49787 |
| `Penmanshiel_SCADA_2023_WT_11-15_5964.zip` | 813ad4635cf50b9d96b253ee53175086 |
| `Penmanshiel_SCADA_2024_WT_11-15_5966.zip` | e0a700cad1d45f4130272feaef244b12 |

Not used: Penmanshiel PMU and grid zips and the KMZ file (§4).

### 3.2 Known data facts and gaps

Verified in the Kelmarsh repo (assume the same for Penmanshiel until
`inspect_data.py` proves otherwise):

- Turbine data files (`Turbine_Data_<Farm>_<N>_<from>_-_<to>_<id>.csv`):
  9 `#` metadata lines, then the column header is itself a `#` comment
  line; CRLF line endings. Static CSV has a UTF-8 BOM.
- Column count differs per year (299 / 303 / 312 for Kelmarsh); read only
  configured columns.
- Kelmarsh 2023 and 2024 exports repeat the year many times (~2.17M rows for
  52,560 timestamps). Repeats are exact or all-NaN copies and a NaN copy can
  come first: keep the **most complete row per timestamp**, never "keep
  first".
- Kelmarsh was commissioned 15 April 2016; no earlier row is normal operation.
- The Kelmarsh mapping CSV lists `Power` in kWh, but `Power (kW)` is the
  10-minute mean power.

Penmanshiel-specific (from the Zenodo file list, not yet inspected):

- SCADA zips are split by turbine group (WT01–10, WT11–15). All zips of one
  year unzip into the same `scada_<year>/` folder.
- **2023, WT01–10:** no January archive (Feb, Mar, Q2, Q3–Q4 only).
- **2024:** only WT11–15. WT01–10 test data therefore ends in December 2023.
- The signal mapping is an `.xlsx`, not a CSV.
- File-name pattern and whether turbine numbers are zero-padded are unknown.
- The dataset notes that not all signals exist for the whole period.

Record every confirmed difference under "Deviations from spec" in the README.

### 3.3 Status (event) files

Each SCADA zip also holds `Status_<Farm>_<N>_....csv` per turbine (Kelmarsh
verified). Format: 9 `#` metadata lines, then a **plain (non-comment)**
header:

```
Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,
Service contract category,IEC category,Global contract category,
Custom contract category
```

`Status` is `Stop`, `Warning` or `Informational`; informational rows have
`-` as end and duration. Kelmarsh 2016–2024 contains component-level events
usable as fault labels, for example (count, status, message):

| Count | Event |
|---|---|
| 28 | Stop, Pitch run-away (hub box v.>=4) |
| 15 | Warning, Particle Gear Alarm 10min |
| 11 | Stop, Uncontrolled yaw movement |
| 8 | Stop, Implausible gear speed |
| 7 | Stop, Low gearbox oil pressure |
| 6 | Warning, High temp. gear bearing 1 |
| 6 | Stop, Pitch angle deviation |
| 4 | Warning, Oil filter gear choked |

Fault events are **rare**: evaluation must report counts honestly and use
case studies where statistics are too thin.

### 3.4 Candidate signals

Core signals (confirmed for Kelmarsh 2016–2024):

| Key | Column |
|---|---|
| `wind_speed` | `Wind speed (m/s)` |
| `power` | `Power (kW)` |
| `wind_direction` | `Wind direction (°)` |
| `nacelle_direction` | `Nacelle position (°)` |
| `ambient_temp` | `Ambient temperature (converter) (°C)` |
| `pitch` | `Blade angle (pitch position) A (°)` |
| `rotor_speed` | `Rotor speed (RPM)` |

Component signals (seen in the Kelmarsh 2022 header only; must be confirmed
for every year of both farms):

| Key | Column |
|---|---|
| `front_bearing_temp` | `Front bearing temperature (°C)` |
| `rear_bearing_temp` | `Rear bearing temperature (°C)` |
| `gear_oil_temp` | `Gear oil temperature (°C)` |
| `gear_oil_inlet_temp` | `Gear oil inlet temperature (°C)` |
| `generator_bearing_front_temp` | `Generator bearing front temperature (°C)` |
| `generator_bearing_rear_temp` | `Generator bearing rear temperature (°C)` |
| `stator_temp` | `Stator temperature 1 (°C)` |
| `nacelle_temp` | `Nacelle temperature (°C)` |

## 4. Scope

In scope:

- SCADA and status data 2016–2024 for both farms, all turbines.
- Train on 2016–2021, test on 2022–2024, for both farms.
- Normal-behaviour models: power (§7.7–7.8), drivetrain thermal (§7.9),
  pitch and yaw (§7.8).
- Physics-only, ML-only (LightGBM baseline) and hybrid variants; a
  physics-informed neural network (PINN) for the thermal model.
- Residual-based component alerts and event-based validation.
- Cross-site experiment Kelmarsh ↔ Penmanshiel.
- RDF graph with rdflib, serialised to Turtle.
- Streamlit dashboard and a written report.

Stretch (only after §9 is met):

- Wake-aware expected power with PyWake (§7.13).

Out of scope:

- Federated learning implementation (future work, §10).
- PMU and grid meter data.
- Triple stores, SHACL, RDS-PP designation codes.
- Real streaming infrastructure (MQTT, Kafka).
- Remaining-useful-life prediction.

## 5. Tech stack

- Python 3.11 (pinned), managed exclusively with **uv**: `uv init`,
  `uv python pin 3.11`, `uv add <pkg>` (`--dev` for dev tools), commit
  `uv.lock`, run everything with `uv run`. No pip, no `requirements.txt`, no
  manual venv.
- Data: pandas, pyarrow, duckdb, openpyxl (Penmanshiel mapping `.xlsx`)
- Models: lightgbm, scikit-learn, scipy (grey-box parameter fitting), torch
  (PINN; a GPU is available, install a CUDA build if `uv` makes it simple,
  otherwise CPU is acceptable at this data size)
- Graph: rdflib
- App and plots: streamlit, plotly, matplotlib
- Utilities: pyyaml, requests, tqdm
- Dev: pytest, ruff
- Stretch: py_wake. Future: flwr.

All packages above are approved. Ask before adding anything else.

## 6. Repository layout and configuration

```
maintenance-twin-wind/
  README.md
  CLAUDE.md
  pyproject.toml, uv.lock, .python-version
  config.yaml
  Makefile
  docs/
    spec.md         # this file
    report.md       # results report (§7.12)
    figures/        # report figures (generated, committed)
  data/             # git-ignored
    raw/<farm>/     # downloads; SCADA unzipped into scada_<year>/
    processed/<farm>/  # <T>.parquet per turbine, events.parquet
  ontology/
    wind_twin.ttl   # generated graph (committed)
  models/           # git-ignored: model files, metrics.json
  src/
  app/
    app.py
  tests/
```

`config.yaml` holds every path, URL, MD5, column name, model setting and
threshold. No hard-coded values in scripts. Shape:

```yaml
years: [2016, ..., 2024]
paths: {raw, processed, ontology, models}
farms:
  kelmarsh:
    code: KLM               # prefix for farm-scoped ids: KLM_T01
    scada_name: Kelmarsh    # farm name in file names
    dataset: {title, doi, publisher, license}
    downloads: [{name, url, md5, extract_to?}, ...]
  penmanshiel: {code: PEN, scada_name: Penmanshiel, ...}
scada: {timestamp_column: "Date and time"}
signals: {...}              # §3.4 core
component_signals: {...}    # §3.4 components
cleaning: {...}
fault_labels: {...}         # §7.4 message -> component mapping
models: {...}               # per model: inputs, target, hyperparameters
train_test_split: {train_end: "2021-12-31 23:50"}
alerts: {...}
evaluation: {...}           # lookback window for lead time etc.
dashboard: {...}
```

Turbine-specific physical constants (rotor diameter, rated power, location)
come from each farm's static file, not from `config.yaml`.

## 7. Components

Implement in order; each step is a milestone with its own commit (§8).
Every script runs standalone with `uv run python -m src.<name>` and has a
Makefile target.

### 7.1 Scaffold and reuse

Initialise the uv project, `config.yaml`, `Makefile`, `src/config.py`
(`load_config`, `farm_names`, `farm_raw_dir`, `scada_archives`,
`scada_dirs` with de-duplicated shared folders), pytest and ruff.

Reuse from `~/digital-twin-wind-farm` at `v1.0-kelmarsh`, adapting to
multiple farms:

| Source module | Reuse | Change needed |
|---|---|---|
| `download.py` | MD5 check, skip, `.part` streaming, unzip | Loop over farms, `data/raw/<farm>/` |
| `scada_format.py` | Header detection, BOM/CRLF, file grouping | Turbine id regex from `scada_name`; status file header is not a comment |
| `inspect_data.py` | Column listing, missing-column report | Per farm; `.xlsx` mapping; component signals; status inventory |
| `load_scada.py`, `clean.py` | "Most complete row" dedup, year filter, `is_normal`, air density | Per farm paths; component columns |
| `graph_utils.py`, `build_ontology.py` | SPARQL-constant pattern, deterministic IRIs, provenance | Farm-scoped IRIs, new classes (§7.5) |
| `dashboard.py`, `ontology_view.py`, `app/app.py` | Replay, KPIs, lineage rendering | Farm selector, health view |
| `train.py`, `alerts.py` | LightGBM baseline, rolling-residual alerts | Become the baseline and generic residual alerting |

### 7.2 `download.py`

Download each farm's files into `data/raw/<farm>/`, verify MD5 (raise on
mismatch), skip valid files, unzip every archive into its `extract_to`
folder (several archives may share one), skip already-extracted archives.

Acceptance: re-running downloads nothing; every `scada_<year>/` folder of
both farms exists.

### 7.3 `inspect_data.py` (decision gate)

Print, per farm and year folder: file groups, raw head and column list of
one turbine file, and the configured core and component columns missing per
file. Print the static and mapping tables. For status files, print the
header and the counts of `Stop`/`Warning` messages per farm.

Acceptance and decisions, recorded in the README and the change log:

1. Real column names for both farms in `config.yaml` (do not guess).
2. Which component signals exist for all years in both farms. Thermal models
   (§7.9) are built only for components covered by both farms; if a
   component exists only in Kelmarsh, it may still be modelled there, noted
   as a limitation.
3. The fault-label mapping (§7.4) from the real messages.
4. Any Penmanshiel layout deviations (file names, padding, duplicates).

### 7.4 `load_scada.py`, `clean.py`, `events.py`

- `load_scada.py`: per farm and turbine, combine all yearly files into
  `data/processed/<farm>/<T>.parquet`: timestamp, core and component
  signals; rows within `years`; duplicate timestamps resolved to the most
  complete row.
- `clean.py`: add `air_density` (from ambient temperature and standard
  pressure), `yaw_misalignment` (wind direction minus nacelle position,
  wrapped to ±180°, if the semantics are confirmed) and `is_normal`
  (operating-range rules as in the Kelmarsh repo, plus "no Stop event
  active", once §7.4 events exist). Rated wind speed per turbine type comes
  from `config.yaml` `cleaning` keyed by turbine model.
- `events.py`: parse all status files into
  `data/processed/<farm>/events.parquet` (turbine, start, end, status, code,
  message, IEC category) and add a `component` column from
  `config.yaml` `fault_labels` (message pattern → component: `gearbox`,
  `generator`, `pitch`, `yaw`, `main_bearing`, ...). Unmatched messages are
  `None`.

Acceptance: print rows per turbine and normal fraction; print fault events
per farm and component.

### 7.5 `build_ontology.py` and `graph_utils.py`

The RDF graph `ontology/wind_twin.ttl` is the **source of structural
truth**; apps and scripts discover farms, turbines, components, sensors,
models and their inputs through `graph_utils.py` SPARQL queries, never
through `config.yaml`. `graph_utils.py` is the only module that touches the
graph.

Classes (prefix `kt:`): `Dataset`, `WindFarm`, `Turbine`, `Component`
(subclasses `Gearbox`, `Generator`, `MainBearing`, `PitchSystem`,
`YawSystem`), `Sensor`, `Model` (subclasses `PowerModel`, `ThermalModel`,
`PitchModel`, `YawModel`), `Alert`, `FaultEvent`.

Key properties: `kt:hasTurbine`, `kt:hasComponent`, `kt:hasSensor`,
`kt:monitors` (sensor → component), `kt:storedIn` (Parquet path),
`kt:scadaColumn`, `kt:signalKey`, `kt:modelOf` (model → component),
`kt:modelVariant` (`baseline`, `physics`, `hybrid`, `pinn`), `kt:hasInput`,
`kt:hasTarget`, metric properties, `kt:raisedBy` (alert → model),
`kt:concerns` (alert → component), `kt:precedes` (alert → fault event it
anticipated, with `kt:leadTimeHours`). Provenance via PROV-O as in the
Kelmarsh repo (sources → zips → datasets; `prov:generatedAtTime` on models).

IRIs are deterministic and farm-scoped: `kt:KLM`, `kt:KLM_T01`,
`kt:KLM_T01_gearbox`, `kt:KLM_T01_gear_oil_temp`,
`kt:ThermalModel_KLM_T01_gearbox_hybrid`, `kt:Alert_KLM_T01_gearbox_001`,
`kt:FaultEvent_KLM_T01_0001`. Re-running a step replaces its nodes.

Fault events in the graph: only events with a mapped component (not every
informational row).

### 7.6 Shared modelling conventions

- Train rows: `is_normal` and timestamp ≤ `train_end`. Test: after
  `train_end`. Healthy-only training is what makes residuals meaningful.
- Each model variant implements the same small interface
  (`fit(train_frames)`, `predict(frame)`), where `train_frames` is a
  **mapping turbine id → DataFrame**. Never pool data in a way that hides
  which turbine it came from: this is the hook for federated learning
  (§10).
- Metrics per turbine and variant in `models/metrics.json`: RMSE, MAE and
  normalised RMSE (power: % of rated; temperature: °C).
- Random seeds fixed in `config.yaml`.

### 7.7 Baseline: LightGBM power model

Port the Kelmarsh model per turbine (inputs: wind speed, wind direction
sin/cos, ambient temperature, air density). This is the reference every
other variant is compared against.

### 7.8 Physics and hybrid power, pitch and yaw models

**Physics power model** (per turbine type):

- Density-normalised wind speed, IEC 61400-12-1 style:
  `v_n = v · (ρ / ρ₀)^(1/3)`.
- Expected power from the method of bins on training data
  (`P_bin(v_n)`), or equivalently `P = ½ ρ A C_p(v_n) v³` with
  `C_p` per bin, using the rotor area `A` from the static file, clipped to
  `[0, P_rated]` and zero outside cut-in/cut-out. Report the fitted
  `C_p(v)` curve and check it stays below the Betz limit (16/27).

**Hybrid power model:** `P̂ = P_phys + r_ML(features)`, where `r_ML` is a
LightGBM (or small NN) residual model, and the result is clipped to
`[0, P_rated]`. Use LightGBM's monotone constraint so expected power never
decreases with wind speed in the ML-only variant as well.

**Pitch model:** above rated wind speed, expected pitch angle from a binned
physics/hybrid model of pitch vs `v_n`; residuals flag pitch faults.

**Yaw model:** static yaw misalignment per turbine from the power loss vs
misalignment relation (`P ∝ cos^k(γ)`); alerts on drifts in estimated
misalignment over rolling windows. If wind direction is not independent of
the nacelle position in the data, document it and drop the yaw model.

### 7.9 Thermal health models (main contribution)

For each component temperature `T` confirmed in §7.3, with the nacelle
temperature `T_nac` as the reference:

**Physics (grey-box) model**, a first-order lumped heat balance:

```
C · dT/dt = a·P + b·P² − h·(T − T_nac)
T_{k+1} = T_k + Δt/C · (a·P_k + b·P_k² − h·(T_k − T_nac,k))   (Δt = 600 s)
```

Parameters `(a/C, b/C, h/C)` are fitted per turbine and component on
healthy training data (scipy least squares, one-step-ahead).

**Hybrid model:** physics one-step prediction plus an ML residual model
(LightGBM) using rotor speed, ambient temperature and recent history.

**PINN:** a small PyTorch MLP that predicts `T` from inputs and recent
history, trained with

```
loss = MSE(T̂, T) + λ · MSE(ODE residual of T̂)
```

where the ODE residual uses the heat-balance equation with learnable,
positive physical parameters. `λ` lives in `config.yaml`. Use one shared
network per component across all turbines of both farms with a learned
turbine embedding and turbine-type features; train with per-turbine data
loaders (§7.6).

Compare physics, hybrid, PINN and a LightGBM-only temperature model on test
RMSE, and on the alert evaluation in §7.10.

### 7.10 `alerts.py` and `evaluate.py`

- `alerts.py`: for each model, residual = measured − expected. Smooth with a
  rolling mean (window from config); thresholds derived from the
  **training** residual distribution (e.g. a high percentile, sign per
  model: low power, high temperature). Contiguous exceedances longer than
  a minimum duration become one alert per period. Write alerts to the graph
  (`kt:Alert`), replacing that model's previous alerts.
- `evaluate.py`: match alerts to fault events of the same turbine and
  component. An event is **detected** if an alert starts within the
  lookback window before it (window in config, e.g. 30 days). Report per
  variant: events detected / total, lead-time distribution (hours), false
  alarms per turbine-year (alerts matching no event within the window).
  Link matched alerts with `kt:precedes`. Write `models/evaluation.json`.

Acceptance: printed summary table per farm, component and variant.

### 7.11 Cross-site experiment

Train power and thermal variants on Kelmarsh only (2016–2021) and test on
Penmanshiel (2022–2024), and the reverse. The hypothesis to test: physics
and hybrid models transfer across rotor sizes better than the ML-only
baseline. Report results whether they support it or not.

### 7.12 Report

`docs/report.md` with figures in `docs/figures/` generated by a script
(`src/report.py`, matplotlib): data overview and gaps, model comparison
tables, cross-site results, fault case studies (residual time series
before a real event), limitations. Written for an internal technical
audience; the dashboard is the pitch surface.

### 7.13 Stretch: wake-aware expected power

PyWake model per farm from the static-file layout (e.g. a Bastankhah
Gaussian wake model) giving an expected wake loss per turbine, wind speed and
direction; multiply into the physics power model. Show whether it reduces
false underperformance alerts for waked turbines.

### 7.14 `app/app.py` (Streamlit)

Rendering only; logic in testable modules (`src/dashboard.py`,
`src/ontology_view.py`). Tabs:

- **Replay:** farm selector, map, turbine status, replay slider and play,
  power and component temperature vs expected, active alerts.
- **Component health:** per turbine and component, health index (latest
  smoothed residual vs threshold), residual time series with alerts and
  real fault events marked.
- **Model comparison:** metrics and evaluation tables per variant, cross-site
  results.
- **Ontology & lineage:** the SPARQL queries used and the lineage graph.

Footer: attribution for both datasets.

### 7.15 Makefile

Targets: `setup`, `download`, `inspect`, `data` (load, clean, events),
`ontology`, `train` (all variants), `alerts`, `evaluate`, `report`, `app`,
`all`, `test`, `lint`.

### 7.16 README

Description, pipeline diagram (Mermaid), setup and run, results tables,
known limitations, "Deviations from spec", attribution for both datasets,
and the Kelmarsh repo tag the code was derived from.

## 8. Instructions for Claude Code

- Work through §7 in order; check each milestone's acceptance criteria by
  running it before starting the next.
- Run `inspect_data.py` before writing the loader; never guess column
  names. If data differs from this spec, adapt the code to the data and log
  the change under "Deviations from spec" in the README.
- Tests use small synthetic data, never the network or real data. Write
  meaningful tests for every change; keep `pytest` and `ruff check` green;
  do not change existing assertions while refactoring.
- Type hints and short docstrings; `logging` for script output (`print`
  only for required acceptance output).
- Functional style, short functions, one responsibility per module, no
  hard-coded values.
- Never commit anything under `data/` or `models/`.
- One commit per request/milestone, 50/70 message format, tests included.
- Log every milestone and change in §11.
- Ask before adding dependencies not listed in §5.

## 9. Definition of done

- From a clean clone, `uv sync` then `make all` runs without errors.
- The graph holds both farms, 20 turbines, their components and sensors, all
  model variants with metrics, alerts and mapped fault events, with
  provenance.
- `models/metrics.json` and `models/evaluation.json` cover every turbine,
  component and variant that the data supports.
- The cross-site experiment results are in the report.
- The Streamlit app runs for both farms; replay and the health view work.
- README and `docs/report.md` document results, limitations, deviations and
  attribution.

## 10. Future work

- **Federated learning, per turbine:** each turbine is a client holding its
  own data; train the shared PINN/hybrid models with Flower (FedAvg, then
  FedProx for non-IID farms). Compare local-only vs federated vs
  centralised. The per-turbine training interface (§7.6) is the integration
  point.
- Secure or privacy-preserving aggregation for multi-operator scenarios.
- Remaining-useful-life estimation from health-index trends.
- Grid meter reconciliation, triple store and SHACL validation.

### Open decisions

- Report format beyond `docs/report.md` (slide deck for pitches?).
- Whether to include Penmanshiel PMU/grid data later.

## 11. Change log

### Spec created (2026-11-17)

- Initial specification derived from the Kelmarsh PoC (`v1.0-kelmarsh`) and
  the planning discussion: maintenance-twin goal, hybrid physics-ML models,
  two farms, train 2016–2021 / test 2022–2024, federated learning as future
  work.

### Milestone 7.1: scaffold (2026-11-17)

- `config.yaml` with both farms (codes `KLM`/`PEN`, dataset metadata, all
  11 + 22 downloads with MD5 and `extract_to`), core signals, years and the
  train/test split. Penmanshiel archives of one year share a `scada_<year>`
  folder.
- `src/config.py`: `load_config`, `farm_names`, `farm_raw_dir`,
  `scada_archives`, `scada_dirs` (de-duplicated, config order), adapted
  from `v1.0-kelmarsh`.
- Tests for the helpers (synthetic config) and the shipped config (URLs
  match each farm's Zenodo record, scada folders match `years`).
- `Makefile` (`setup`, `test`, `lint`), `README.md`, `CLAUDE.md`.
  Pipeline targets and dependencies arrive with their milestones.

### Milestone 7.2: download (2026-11-18)

- `src/download.py` adapted from `v1.0-kelmarsh`: `download_farm` fetches
  one farm into `data/raw/<farm>/` (MD5 check, skip, `.part` streaming)
  and unzips each archive into its `extract_to` folder, several archives
  sharing one folder. `make download` target.
- Tests: helpers plus a synthetic two-archive farm (shared folder filled,
  re-run fetches and extracts nothing).
- Kelmarsh raw files hard-linked from the Kelmarsh repo; Penmanshiel
  downloaded (22 files). Acceptance met: re-run downloads and extracts
  nothing; `scada_2016`–`scada_2024` exist for both farms.
- Penmanshiel layout findings (zero-padded turbine numbers, 2023 period
  files, large 2024 files, ~62 GB unzipped) recorded in the README
  deviations.

### Milestone 7.3: inspection decision gate (2026-11-18)

- `src/scada_format.py` ported: turbine ids from `scada_name` (padded or
  not), `status_files`, `status_header_index` (status header is a plain
  line). `src/inspect_data.py` ported per farm: file groups, head and
  columns per year, missing columns per file and per farm, filled share
  of timestamps per signal, static/mapping tables (`.xlsx` too), status
  header and distinct Stop/Warning counts. `make inspect` target.
- Decisions (README "Data inspection decisions"): all 15 configured
  columns confirmed in both farms; all 8 component signals usable in both,
  with Penmanshiel pitch, main-bearing and gear-oil values only from
  mid-2018; `fault_labels` (35 messages → gearbox, generator, pitch, yaw;
  no main-bearing messages); Penmanshiel layout deviations logged.
- `config.yaml`: `component_signals`, `events.fault_statuses`,
  `fault_labels`.

### Milestone 7.4: load, events, clean (2026-11-18)

- `src/load_scada.py` ported per farm: core and component signals into
  `data/processed/<farm>/<T>.parquet`, most complete row per timestamp.
- `src/events.py`: all status rows de-duplicated into
  `data/processed/<farm>/events.parquet` with a `component` column from
  `fault_labels` (Stop/Warning only).
- `src/clean.py`: `air_density`, `yaw_misalignment` (confirmed to be the
  nacelle vane's relative angle) and `is_normal` (operating-range rules,
  commercial operations date from the static file, no Stop event active;
  missing pitch accepted). `src/static_data.py` reads the static file.
- MM82 rated wind speed 14.0 m/s estimated from data. `pyarrow` added.
  `make data` target (about 9 minutes).
- Acceptance: 20 turbines, 8.6 million rows, 72–81 % normal per turbine;
  fault events per farm and component printed and match §7.3.

### Milestone 7.5: ontology (2026-11-18)

- `src/graph_utils.py` ported for two farms: farm-scoped IRI helpers,
  SPARQL lookups `get_farms`, `get_turbines`, `get_components`,
  `get_sensors`, `get_fault_events`, `get_lineage`, `get_schema`,
  `parquet_path_of`, and load/save with the schema block first.
- `src/build_ontology.py`: schema for all §7.5 classes and properties,
  datasets and source files (raw files linked to their real zip), farms,
  turbines with specs, 7 components and 15 sensors per turbine, labelled
  fault events. `make ontology` target; `rdflib` added.
- `events.parquet` gained `source_file`; `static_data.read_mapping`;
  `config.yaml` `farms.*.mapping_file`.
- Acceptance: `ontology/wind_twin.ttl` with both farms, 20 turbines, 300
  sensors, 1,043 fault events and the provenance of 429 source files.

### Milestones 7.6–7.7: modelling conventions and baseline (2026-11-18)

- `src/modelling.py`: `NormalBehaviourModel` protocol (`fit` on a mapping
  turbine id → DataFrame, `predict`), `split_masks`, `evaluate` (RMSE,
  MAE, nRMSE), `prediction_column`, `turbine_paths` (via the graph),
  `merge_metrics` for `models/metrics.json`.
- `src/power_baseline.py`: LightGBM power model per turbine with spec
  features and a monotone constraint on wind speed; writes
  `expected_power_baseline`, model files, graph model nodes and metrics.
  `make train` target; `lightgbm` and `scikit-learn` added; `seed` and
  `models.power_baseline` in `config.yaml`.
- `graph_utils`: `model_iri`, `local_name`, `add_model`, `get_models`,
  `get_model_inputs`; metric properties in the schema.
- Acceptance: 20 models; nRMSE 3.2–4.6 % (Kelmarsh), 2.5–3.1 %
  (Penmanshiel) except PEN T08 (6.5 %, a months-long underperformance
  episode in 2021–2022).

### Milestone 7.8, part 1: physics and hybrid power (2026-11-18)

- `src/physics_power.py`: `PhysicsPowerModel` (Cp(v_n) curve per turbine
  type from the aerodynamic bins, P = ½ ρ₀ A Cp v_n³ capped at rated, zero
  outside cut-in/cut-out, Betz check) and `HybridPowerModel` (physics +
  LightGBM residual per turbine). `src/binned.py` (method of bins).
  Added to `make train`; `scipy` added.
- Refactor: `metrics_entry` moved to `modelling.py` with
  `register_model`, `model_id_of`, `with_turbine_specs`.
- Acceptance: max Cp 0.504 (MM92) and 0.476 (MM82), below Betz; mean test
  nRMSE baseline / physics / hybrid 3.65 / 3.79 / 3.64 % (Kelmarsh) and
  3.02 / 3.50 / 3.18 % (Penmanshiel).
