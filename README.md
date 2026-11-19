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
(modelling conventions, LightGBM baseline), §7.8 (physics and hybrid
power, pitch and yaw), §7.9 (thermal models) and §7.10 (alerts and
evaluation) done. Later milestones add their pipeline steps and Makefile
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
`src/graph/graph_utils.py`, not through `config.yaml`. `make ontology` rewrites
the whole file, so steps that add models or alerts must run again after it.

```bash
make train        # power, pitch, yaw (~2 min), then thermal (~15 min)
```

`make train` writes `expected_power_<variant>` and
`expected_pitch_<variant>` into each turbine Parquet, model files under
`models/<farm>/` (LightGBM boosters, the fitted Cp and pitch curves per
turbine type as `power_physics_<model>.json` and
`pitch_physics_<model>.json`, and each yaw fit as
`YawModel_<FARM>_<T>_physics.json`), the model nodes (inputs, target,
metrics, `prov:generatedAtTime`) into the graph, and the metrics into
`models/metrics.json`. The yaw step also writes the rolling-window
misalignment estimates to `data/processed/<farm>/yaw_offsets.parquet`.

The thermal step writes `expected_<temperature>_<variant>` (7 temperatures
× physics, baseline, hybrid, pinn) and, under `models/thermal/`, the
heat-balance parameters (`physics_<target>.json`) and the shared networks
(`pinn_<target>.pt`); it uses the GPU when available. Every training step
runs at low priority (`nice`) with `compute.threads` (8) CPU threads, and
each Parquet file is written atomically (`src/storage.py`). To retrain
selected temperatures only (the others keep their saved models):

```bash
uv run python -m src.models.thermal.train --targets front_bearing_temp
```

```bash
make alerts       # residual alerts for every model -> graph (~15 s)
make evaluate     # match alerts to fault events (~15 s)
make curves       # detection vs false alarms at 6 thresholds (~1 min)
```

`make alerts` replaces each model's `kt:Alert` nodes in the graph and
writes `models/alerts.json` (threshold, alert count and yearly mean
residual score per model). `make evaluate` links alerts to the fault
events they precede (`kt:precedes`, `kt:leadTimeHours`), prints the
summary per farm, component, target and variant, and writes
`models/evaluation.json` (summary in both scopes, episode counts, drift
table, drift alerts with the events after them); `make alerts` also writes
the drift alerts to `models/drift_alerts.json`. `make curves` writes
`models/curves.json`. Run them in this order after `make train`.

```bash
make cross-site   # train on one farm, test on the other (~30 min, GPU)
```

`make cross-site` fits every power and thermal variant on one farm's
training years and scores it on the other farm's test rows, in both
directions, next to the in-site predictions from `make train` (run that
first). It writes `models/cross_site.json` (RMSE per turbine, signal and
variant) and prints the mean per direction. Each signal is also saved to
`models/cross_site/<signal>.json` when it is done; a rerun skips the saved
signals (delete the folder to recompute). Nothing is written to the graph.

```bash
make report       # docs/figures/*.png and the report tables (~20 s)
```

`make report` writes the figures of `docs/report.md` (data coverage,
model comparison, cross-site, detection vs chance, fault case studies)
into `docs/figures/` and prints the report's tables as Markdown. It reads
the processed data, `models/*.json` and the graph, so run it last.

Every pipeline script runs standalone with
`uv run python -m src.<package>.<name>` and has a Makefile target (added
per milestone). `src/` is grouped by concern: `data/` (raw files to clean
Parquet), `graph/` (the RDF graph) and `models/` (normal-behaviour models).

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

The written report with figures is `docs/report.md`; this section keeps
the full per-turbine results.

**Data and cleaning (§7.4):**

| Farm | Turbines | Rows | Normal | Normal rows 2016–2021 / 2022–2024 |
|---|---|---|---|---|
| Kelmarsh | 6 | 2,839,104 (473,184 each) | 78.5 % (75.6–80.6 %) | 1,464,798 / 762,653 |
| Penmanshiel | 14 | 5,758,976 (386k–447k each) | 76.3 % (71.9–79.6 %) | 3,085,057 / 1,309,634 |

- Penmanshiel T01–T10 run from 2016-06-06 to the end of 2023 (no 2024
  data); T11–T15 to the end of 2024. T01–T10 also have no January 2023
  (their 2023 exports start on 1 February).
- PEN T15 reports −273.2 °C (sensor module offline) on four temperatures
  for 442 rows on 18–21 April 2024; none of them is a normal row.
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

All models below fit normal rows of **2016–2020**; 2021 is the calibration
year for alert thresholds (§7.10; no model sees it) and the test period is
2022–2024 (Penmanshiel T01–T10 end in 2023). The models were retrained when
§7.10 introduced the calibration year; until then they fit 2016–2021.

**Baseline power model (§7.7):** LightGBM per turbine (rated power
2,050 kW for every turbine; nRMSE = RMSE / rated power):

| Farm | Turbine | RMSE (kW) | nRMSE | MAE (kW) | Train rows | Test rows |
|---|---|---|---|---|---|---|
| KLM | T01 | 76.5 | 3.73 % | 51.4 | 204,621 | 130,349 |
| KLM | T02 | 72.4 | 3.53 % | 51.3 | 207,544 | 132,271 |
| KLM | T03 | 69.2 | 3.38 % | 49.3 | 202,478 | 129,538 |
| KLM | T04 | 93.4 | 4.56 % | 56.2 | 207,125 | 121,104 |
| KLM | T05 | 73.1 | 3.57 % | 51.4 | 203,444 | 126,964 |
| KLM | T06 | 66.1 | 3.22 % | 47.1 | 198,315 | 122,427 |
| PEN | T01 | 52.9 | 2.58 % | 35.4 | 183,500 | 80,367 |
| PEN | T02 | 57.9 | 2.83 % | 38.8 | 179,735 | 78,421 |
| PEN | T04 | 56.7 | 2.77 % | 39.2 | 182,703 | 75,875 |
| PEN | T05 | 58.6 | 2.86 % | 40.2 | 180,608 | 75,856 |
| PEN | T06 | 54.6 | 2.66 % | 37.9 | 177,259 | 73,612 |
| PEN | T07 | 53.6 | 2.61 % | 36.9 | 173,775 | 73,546 |
| PEN | T08 | 145.4 | 7.09 % | 88.6 | 181,932 | 79,029 |
| PEN | T09 | 56.8 | 2.77 % | 36.6 | 175,605 | 75,451 |
| PEN | T10 | 55.8 | 2.72 % | 37.7 | 174,757 | 69,523 |
| PEN | T11 | 62.1 | 3.03 % | 41.5 | 185,365 | 120,896 |
| PEN | T12 | 64.2 | 3.13 % | 44.1 | 188,050 | 126,893 |
| PEN | T13 | 53.4 | 2.60 % | 35.7 | 188,641 | 127,846 |
| PEN | T14 | 61.3 | 2.99 % | 42.4 | 187,014 | 126,738 |
| PEN | T15 | 64.1 | 3.12 % | 43.3 | 188,048 | 125,581 |

- Kelmarsh MAE (47.1–56.2 kW) matches the Kelmarsh repo (47.3–56.7 kW)
  despite the added monotone constraint, the stricter cleaning and one
  training year less.
- **Penmanshiel T08** underperforms for months: its quarterly mean
  normal-row residual (model trained on 2016–2021) was −62 and −44 kW in
  H1 2021, −119 kW in Q4 2021 and about −193 kW in Q1–Q2 2022, then back
  within ±15 kW from Q3 2022. With 2021 out of the training data its test
  nRMSE rose from 6.53 % to 7.09 %: the model no longer learns from
  degraded rows. A case study for §7.10–7.12.

**Physics and hybrid power models (§7.8):** the physics model is one
power-coefficient curve Cp(v_n) per turbine type (method of bins on
density-normalised wind speed, 0.5 m/s bins), giving P = ½ ρ₀ A Cp(v_n)
v_n³ with each turbine's rotor area, capped at rated power. Fitted
maximum Cp: **0.502 (MM92)** and **0.477 (MM82)**, both below the Betz
limit 0.593 (nacelle anemometers behind the rotor read low, which raises
apparent Cp); the curves end at 10.7 m/s (MM92) and 12.7 m/s (MM82). The
hybrid adds a per-turbine LightGBM model of the physics residual. Test
nRMSE (RMSE / rated power):

| Farm | Turbine | Baseline | Physics | Hybrid |
|---|---|---|---|---|
| KLM | T01 | 3.73 % | 3.77 % | 3.75 % |
| KLM | T02 | 3.53 % | 3.68 % | 3.52 % |
| KLM | T03 | 3.38 % | 3.65 % | 3.35 % |
| KLM | T04 | 4.56 % | 4.57 % | 4.54 % |
| KLM | T05 | 3.57 % | 3.91 % | 3.54 % |
| KLM | T06 | 3.22 % | 3.27 % | 3.25 % |
| PEN | T01 | 2.58 % | 2.89 % | 2.77 % |
| PEN | T02 | 2.83 % | 3.26 % | 3.34 % |
| PEN | T04 | 2.77 % | 3.18 % | 3.13 % |
| PEN | T05 | 2.86 % | 3.49 % | 3.26 % |
| PEN | T06 | 2.66 % | 3.05 % | 2.90 % |
| PEN | T07 | 2.61 % | 2.93 % | 2.64 % |
| PEN | T08 | 7.09 % | 6.67 % | 7.15 % |
| PEN | T09 | 2.77 % | 2.98 % | 2.63 % |
| PEN | T10 | 2.72 % | 3.47 % | 2.63 % |
| PEN | T11 | 3.03 % | 3.61 % | 2.93 % |
| PEN | T12 | 3.13 % | 3.55 % | 3.22 % |
| PEN | T13 | 2.60 % | 2.92 % | 2.89 % |
| PEN | T14 | 2.99 % | 3.43 % | 3.11 % |
| PEN | T15 | 3.12 % | 3.83 % | 3.29 % |
| **Mean** | KLM | 3.66 % | 3.81 % | 3.66 % |
| **Mean** | PEN | 3.13 % | 3.52 % | 3.28 % |

- One physics curve per turbine type is within 0.1–0.7 percentage points
  of a per-turbine LightGBM model, and better than it on PEN T08 (6.67 vs
  7.09 %): a curve shared by 14 turbines is not pulled by one turbine's
  history. The hybrid beats physics on 18 of 20 turbines and matches the
  baseline (better on 7, at most 0.5 points worse). Within one site the
  baseline is hard to beat; the physics variants are built for transfer
  (§7.11).

**Pitch models (§7.8):** above rated wind speed (MM92 12 m/s, MM82
14 m/s) the controller pitches to hold rated power. Physics: mean pitch
per 0.5 m/s bin of v_n, one curve per turbine type; hybrid: physics plus a
per-turbine LightGBM residual model (wind speed, ambient temperature, air
density). Test RMSE in degrees, above rated only:

| Farm | Turbine | Physics | Hybrid | Train rows | Test rows |
|---|---|---|---|---|---|
| KLM | T01 | 1.14 | 1.18 | 6,706 | 3,987 |
| KLM | T02 | 0.99 | 0.99 | 7,612 | 5,014 |
| KLM | T03 | 0.84 | 0.82 | 5,600 | 3,754 |
| KLM | T04 | 1.52 | 1.59 | 7,198 | 3,563 |
| KLM | T05 | 0.82 | 0.84 | 5,980 | 3,953 |
| KLM | T06 | 1.00 | 1.03 | 4,205 | 2,811 |
| PEN | T01 | 0.91 | 0.96 | 10,222 | 5,848 |
| PEN | T02 | 1.08 | 1.09 | 9,216 | 5,417 |
| PEN | T04 | 0.86 | 0.96 | 9,260 | 5,170 |
| PEN | T05 | 1.01 | 1.21 | 8,758 | 4,854 |
| PEN | T06 | 1.01 | 1.14 | 9,138 | 4,622 |
| PEN | T07 | 1.07 | 1.19 | 6,814 | 3,634 |
| PEN | T08 | 1.91 | 2.20 | 7,530 | 5,183 |
| PEN | T09 | 1.06 | 1.22 | 6,425 | 3,482 |
| PEN | T10 | 1.04 | 1.19 | 6,834 | 3,564 |
| PEN | T11 | 0.86 | 0.93 | 9,272 | 8,525 |
| PEN | T12 | 1.05 | 1.20 | 11,018 | 10,863 |
| PEN | T13 | 0.85 | 0.99 | 11,131 | 10,976 |
| PEN | T14 | 0.92 | 1.07 | 10,929 | 10,131 |
| PEN | T15 | 0.95 | 1.08 | 10,068 | 9,544 |
| **Mean** | KLM | 1.05 | 1.08 | | |
| **Mean** | PEN | 1.04 | 1.17 | | |

- One binned curve per turbine type predicts pitch to about 1°; the
  physics model is as good as or better than the hybrid on 18 of 20
  turbines, so the residual model adds nothing here. With wind direction
  sin/cos as extra features the hybrid overfitted (PEN T01: 0.64° train,
  1.34° test), so they were dropped.
- PEN T08 (1.91°) and KLM T04 (1.52°) stand out.

**Yaw model (§7.8):** below rated (v_n 5–10 m/s) a turbine misaligned by
γ to the wind produces about cos^k(γ) of its aligned power. The measured
misalignment γ is the nacelle vane's relative wind angle, so the power
ratio to the turbine's own training power curve, binned by γ (1° bins
within ±25°), peaks at the vane's static offset γ₀: ratio = a·cos^k(γ − γ₀)
(scipy least squares, bins weighted by their row count). The fit is
repeated over rolling 90-day windows every 30 days to track drift. An
offset counts as **identified** only if the mean ratio of the three
outermost bins on each side lies at least 5 % below the fitted peak, i.e.
the data really show a peak.

| Farm | Turbine | Static offset γ₀ | k | Peak drop | Identified | Windows: median γ₀ (std) |
|---|---|---|---|---|---|---|
| KLM | T01–T06 | +8.6 to +17.4° | 1.3–1.8 | −6.5 to −0.7 % | no | none identified |
| PEN | T01 | +7.1° | 2.30 | 9.4 % | yes | 6.6° (0.7) |
| PEN | T02 | +8.9° | 2.37 | 9.6 % | yes | 7.1° (2.3) |
| PEN | T04 | +5.1° | 2.40 | 12.0 % | yes | 4.5° (0.8) |
| PEN | T05 | +5.3° | 2.70 | 11.9 % | yes | 5.3° (1.1) |
| PEN | T06 | +7.3° | 2.46 | 10.0 % | yes | 6.8° (1.2) |
| PEN | T07 | +6.4° | 2.72 | 11.1 % | yes | 6.2° (0.6) |
| PEN | T08 | +2.2° | 2.44 | 14.8 % | yes | 2.2° (2.3) |
| PEN | T09 | +4.1° | 2.67 | 16.0 % | yes | 4.6° (0.9) |
| PEN | T10 | +5.9° | 2.67 | 13.4 % | yes | 5.8° (1.0) |
| PEN | T11 | +7.6° | 2.34 | 11.0 % | yes | 7.3° (1.0) |
| PEN | T12 | +5.6° | 2.68 | 13.4 % | yes | 5.4° (0.7) |
| PEN | T13 | +3.4° | 2.05 | 11.5 % | yes | 2.8° (0.8) |
| PEN | T14 | +8.2° | 2.57 | 10.7 % | yes | 7.4° (1.2) |
| PEN | T15 | +9.2° | 2.82 | 11.9 % | yes | 8.7° (0.7) |

- **Penmanshiel:** every turbine has a clear peak with k = 2.0–2.8, in
  line with published cos² to cos³ losses. All vanes read 2–9° off in the
  same direction, so each turbine runs that far misaligned; at k = 2.5 a
  7° offset costs about 2 % of below-rated power. 63 % of the 1,277
  rolling windows are identified (short windows often lack data at the
  edges).
- **PEN T08:** the window offset drifts from about 5° (2016) to 0° (2020–
  2021), then jumps to about 10° from mid-2022 (std 2.3°; most turbines
  0.6–1.2°). The jump follows the end of its underperformance episode
  (Q4 2021–Q2 2022), which suggests the vane was replaced or recalibrated
  during that work; the data cannot confirm it. PEN T02 also varies
  (std 2.3°).
- **Kelmarsh:** no offset is identified. On every turbine the power
  ratio keeps rising across the whole ±25° range (e.g. T01: 0.84 at −15°,
  1.07 at +15°), so the fitted γ₀ (+9 to +17°) is pinned to where the data
  run out, not a real peak. The asymmetry is the same on all six turbines,
  which points at the measurement (vane or nacelle anemometer in the rotor
  wake reacting to misalignment) rather than at six identical offsets.
  Kelmarsh therefore gets no yaw alerts; the yaw model stays for
  Penmanshiel.
- The yaw model's power prediction (power curve × fitted cosine loss) has
  test nRMSE 2.4–3.6 % on Penmanshiel (T08 7.2 %) and 3.6–4.4 % on
  Kelmarsh, on 5–10 m/s rows only (not comparable with the power models).

**Thermal models (§7.9):** 7 temperatures (gear oil and gear-oil inlet;
generator bearings front and rear, stator; main bearing front and rear),
nacelle temperature as the reference, 4 variants each, 560 models. Every
variant predicts from the operating inputs only (never from the measured
temperature after a simulation start), so a component that runs hotter
than normal shows up as a residual:

- **physics:** heat balance
  `T[k+1] = T[k] + c + a·P + b·P² − (h₀ + h₁·ω)(T[k] − T_nac) − h₂·max(T[k] − T_set, 0)`
  per turbine (P in MW, ω = rotor speed / 15 rpm, h₀, h₁, h₂ > 0; the last
  term is a cooler or fan that works above a learned set point), fitted on
  the error of day-long free-run simulations (PyTorch, all turbines at
  once, each with its own parameters); predicted as a free run that
  restarts from the measurement after gaps and is hidden for 6 h after a
  restart.
- **baseline (LightGBM only):** power, rotor speed, ambient and nacelle
  temperature and their 1 h, 6 h and 24 h rolling means, per turbine.
- **hybrid:** physics free run plus a per-turbine LightGBM model of its
  residual (same features plus the physics prediction).
- **pinn:** one MLP per temperature for all 20 turbines (same features,
  rotor diameter as the turbine-type feature, a learned turbine
  embedding), loss = MSE + λ·MSE(ODE residual in °C/h) with learnable
  heat-balance parameters per turbine (without the cooler term), λ = 0.01,
  per-turbine batches.

Mean test RMSE (°C, 2022–2024; all variants scored on the same normal rows,
68k–131k per model):

| Temperature | Farm | Physics | Baseline | Hybrid | PINN |
|---|---|---|---|---|---|
| Gear oil | KLM | 3.27 | 1.62 | **1.61** | 1.63 |
| Gear oil | PEN | 2.82 | 1.48 | **1.46** | **1.46** |
| Gear-oil inlet | KLM | 5.52 | 3.67 | 3.67 | **3.65** |
| Gear-oil inlet | PEN | 7.10 | 5.38 | 5.39 | **5.32** |
| Generator bearing front | KLM | 3.48 | 2.19 | 2.20 | **2.12** |
| Generator bearing front | PEN | 4.06 | 3.31 | 3.32 | **3.16** |
| Generator bearing rear | KLM | 3.00 | 2.56 | 2.56 | **2.41** |
| Generator bearing rear | PEN | 2.62 | 1.79 | 1.80 | **1.69** |
| Stator | KLM | 3.77 | 1.63 | **1.60** | 1.62 |
| Stator | PEN | 4.39 | 1.95 | **1.90** | 1.94 |
| Main bearing front | KLM | 3.66 | **1.54** | 1.55 | 1.59 |
| Main bearing front | PEN | 3.60 | 2.04 | **2.03** | 2.06 |
| Main bearing rear | KLM | 3.80 | **2.13** | 2.14 | 2.16 |
| Main bearing rear | PEN | 3.38 | 2.11 | **2.10** | 2.12 |
| **Mean** | both | 3.93 | 2.46 | 2.46 | **2.42** |

- The PINN is best on 73 of 140 turbine-temperature pairs (hybrid 47,
  baseline 20); baseline, hybrid and PINN are within 0.06 °C of each other
  except on the generator bearings, where the PINN gains up to 0.15 °C.
- **The cooler term made physics 0.8 °C better** (mean 4.73 → 3.93 °C;
  gear oil on Penmanshiel 4.22 → 2.82 °C) and the hybrid built on it
  slightly better (2.49 → 2.46 °C). The learned set points are physical:
  gear oil 55.8 °C (median over turbines), where the data showed it
  regulated (51–57 °C at any power); main bearings 61–67 °C; the cooler
  gain is largest for gear oil and the main bearings and near zero for the
  rear generator bearing. Standstill time constants are 4–12 h.
- The physics model is still 1.5 °C worse: the gear-oil inlet and front
  generator bearing get **cooler** as power rises (coolers and fans
  engage with load), which one lumped balance does not capture.
- The gear-oil inlet is the hardest signal for every variant (it switches
  with the oil cooler).
- Tried and rejected: a LightGBM monotone constraint (expected temperature
  never falls as ambient or nacelle temperature rise) made the baseline
  worse (mean 2.46 → 2.60 °C; KLM stator 1.63 → 2.00 °C), because the
  nacelle temperature falls at high power as the ventilation runs harder.
  The warm-year drift it targeted is handled by fleet-relative scoring
  (§7.10).
- PINN λ (stator, 10,000 steps, fit on 2016–2021): 0 and 0.01 give the
  same test RMSE (1.61 / 1.91 °C), but 0.01 cuts the network's ODE
  residual from 25.5 to 6.5 °C/h, i.e. its predictions stop jumping
  between 10-minute steps; λ = 1 and 10 cost 1.1–1.3 and 2.1–2.6 °C of
  RMSE, because the lumped heat balance itself is only accurate to about
  5 °C.
- The thermal step takes about 2 minutes per temperature (physics and PINN
  on the GPU, LightGBM on 8 threads); system memory stays below 11 GB.

**Alerts and evaluation (§7.10):** for every power, pitch and thermal
model, residual = measured − expected on normal rows (temperatures only
at ≥ 500 kW, as they depend on load), a 24 h rolling mean (at least 6 h of
rows), made **fleet-relative** (minus the median of the other turbines of
the farm for the same model at the same time, which removes weather,
season and farm-wide drift) and turned into a problem score (low power,
high temperature, either side for pitch). Threshold: the 99.5th
percentile of the score in the calibration year 2021. Exceedances of at
least 6 h are alerts, and a model's alerts less than 7 days apart are
merged. Yaw models alert when an identified window offset drifts more
than 4° from the static offset. **Drift alerts** flag at least 3 months
in a row whose mean score is above the calibration year's by 1 °C, 20 kW
or 0.5° (pitch). All settings were fixed before looking at results.

Evaluation (test period 2022+): fault events of one turbine and component
less than 7 days apart are one episode; an episode is **detected** when an
alert of the same turbine and component starts at most 30 days before
it. **Chance** is what alerts at the same rate, placed at random, would
detect (1 − e^(−rate·30 days)). Two scopes: **component** (every labelled
event of the component, as the spec says) and **signal** (only the events
a signal can physically precede: gear oil and gear-oil inlet ← high gear
bearing temperature, oil-cooler fan overload, choked oil filter, gearbox
derating; generator bearings ← high/max bearing temperature; stator ←
thermistor trip). `make curves` repeats the evaluation at six threshold
quantiles (90–99.9 %, all set on 2021) with raw and fleet-relative
scores; a curve's **lift** is detection rate minus chance averaged over
the quantiles.

Test-period episodes, component scope: Kelmarsh gearbox 8 (27 events),
pitch 9, yaw 5, no generator events; Penmanshiel gearbox 111 (288 events),
generator 13, pitch 22, yaw 5. Signal scope: Kelmarsh gearbox 2,
Penmanshiel gearbox 45, and no generator-bearing or stator trips on
either farm. No main-bearing events. 3,478 alerts after train_end
(Kelmarsh 1,477, Penmanshiel 2,001; 13,404 before merging, load filter
and fleet-relative scoring); 133 test alerts precede an episode
(`kt:precedes`); 176 drift alerts.

Component scope at the 99.5 % threshold (fleet-relative):

| Farm | Model (component) | Variant | Detected | Rate | Chance | Median lead | False alarms / turbine-year |
|---|---|---|---|---|---|---|---|
| KLM | gear oil (gearbox) | physics | 1/8 | 0.12 | 0.26 | 2.8 d | 2.4 |
| KLM | gear oil (gearbox) | baseline | 2/8 | 0.25 | 0.28 | 15.9 d | 2.7 |
| KLM | gear oil (gearbox) | hybrid | 3/8 | 0.38 | 0.32 | 22.2 d | 3.0 |
| KLM | gear oil (gearbox) | pinn | 3/8 | 0.38 | 0.32 | 9.7 d | 3.3 |
| PEN | gear oil (gearbox) | physics | 0/111 | 0.00 | 0.01 | – | 0.3 |
| PEN | gear oil (gearbox) | baseline | 22/111 | 0.20 | 0.19 | 16.5 d | 1.3 |
| PEN | gear oil (gearbox) | hybrid | 8/111 | 0.07 | 0.11 | 9.0 d | 1.3 |
| PEN | gear oil (gearbox) | pinn | 25/111 | 0.23 | 0.21 | 15.5 d | 1.8 |
| PEN | power (generator) | physics | 5/13 | 0.38 | 0.46 | 21.0 d | 4.8 |
| PEN | power (generator) | baseline | 3/13 | 0.23 | 0.25 | 25.1 d | 3.8 |
| PEN | stator (generator) | pinn | 4/13 | 0.31 | 0.07 | 14.4 d | 1.6 |
| PEN | stator (generator) | others | 0/13 | 0.00 | 0.03–0.04 | – | 0.3–1.0 |
| both | pitch | physics, hybrid | 0/31 | 0.00 | ≤ 0.01 | – | ≤ 0.1 |
| PEN | yaw | physics | 0/5 | 0.00 | 0.03 | – | 0.1 |

The full tables are printed by `make evaluate` and `make curves` and
stored in `models/evaluation.json` and `models/curves.json`.

- **Event-timed detection stays at chance level, even measured fairly.**
  Across six thresholds, both scopes and raw or fleet-relative scores, the
  lift lies between −0.24 and +0.14 for every model with at least 5
  episodes (mostly within ±0.06), and
  between −0.09 and +0.03 on the 111 Penmanshiel gearbox episodes
  (component scope) and the 45 signal-scope episodes. The one row well
  above chance (PEN stator PINN, 4/13 vs 0.07; fleet lift +0.14) rests on
  13 episodes among about 140 rows, so one such row is expected by
  chance. The load filter, fleet-relative scoring and merging together cut
  false alarms (Kelmarsh gear oil, LightGBM: 29 → 2.7 per turbine-year
  since the first §7.10 run) without improving detection. The
  labelled events, mostly stops such as "Missing gear oil" and particle
  alarms, are not preceded by a 24-hour temperature or power anomaly in
  10-minute SCADA data.
- **Drift alerts work as a turbine-level health indicator for the
  gearbox.** The turbines with a gear-oil or gear-oil-inlet drift alert
  from at least 3 of the 4 variants (KLM T01, T04, T06; PEN T08, T09,
  T15) had more gearbox events in the test period: Kelmarsh 8.3 vs 0.7 per
  turbine, Penmanshiel 43.7 vs 14.3, including the turbine with the most
  events on each farm (PEN T15: 97, KLM T04: 20). An exact permutation
  test of the flagged turbines' event ranks within each farm gives
  p = 0.074: suggestive, not significant with 3 flagged turbines per farm.
  - KLM T04 gear oil runs **+3 to +5 °C above the rest of its farm** from
    July 2023 (all four variants, physics included); above 500 kW its
    quarterly median no longer follows the seasons (51–55 °C in 2022, about
    57 °C in every quarter from Q3 2023). It followed 16 particle alarms in
    January–April 2023: debris first, heat later.
  - PEN T09 gear-oil inlet drifts from May 2022, a year before its
    "Missing gear oil" stops escalate (8 in Q3 2023, 14 in Q4 2023); but
    the drift is only +1–2 °C on top of an inlet that already ran 5–7 °C
    above the farm in 2021.
  - PEN T15 gear-oil inlet runs up to +9 °C above its farm in 2024 (all
    variants), alongside 97 gearbox events from 2022 on. The jump follows a
    3-day sensor-module outage (18–21 April 2024) and the gear-oil sump
    temperature does not move, so it more likely reflects cooler or sensor
    work than a hotter gearbox (`docs/report.md` §6).
  - KLM T06 gear-oil inlet +6 °C from 2022, followed only by five
    "Implausible gear speed" events (a sensor fault, probably unrelated).
- **Generator drift alerts are not specific:** they flag 13 of 20
  turbines, and the flagged ones had fewer generator events (Penmanshiel
  0.5 vs 3.8 per turbine). There are no stator or bearing trips in the
  test period to validate the generator temperatures against.
- The yearly drift table (each year's mean score minus 2021's) in
  `evaluation.json` gives the same picture per model.

Cross-site experiment (§7.11): each variant is trained on one farm (MM92,
92 m rotor, Kelmarsh; MM82, 82 m, Penmanshiel) and tested on the other.
Mean test RMSE over the test farm's turbines, in-site → cross-site, on the
same rows (normal, after 2021, every prediction defined):

| Signal | Variant | KLM → PEN | Increase | PEN → KLM | Increase |
|---|---|---|---|---|---|
| power (kW) | baseline | 64.1 → 230.0 | +259 % | 75.1 → 225.2 | +200 % |
| power (kW) | physics | 72.1 → 93.7 | +30 % | 78.1 → 96.0 | +23 % |
| power (kW) | hybrid | 67.2 → 90.8 | +35 % | 75.0 → 102.6 | +37 % |
| 7 temperatures (°C, mean) | baseline | 2.4 → 4.9 | +124 % | | |
| | physics | 3.9 → 5.2 | +38 % | | |
| | hybrid | 2.4 → 4.9 | +123 % | | |
| | pinn | 2.4 → 5.5 | +157 % | | |

(The temperature rows average both directions and all 7 targets; the
increase is the mean of the 14 per-target increases. Per direction and
target: `models/cross_site.json` or `make cross-site` output.)

- **Power: the hypothesis holds.** The LightGBM baseline, which has no
  rotor-size input, triples its error on the other farm; the physics
  model (Cp curve per type, scaled by each turbine's rotor area) loses
  only 23–30 % and becomes the most accurate cross-site model in one
  direction, the hybrid in the other.
- **Temperatures: physics transfers most robustly, but not most
  accurately.** The physics heat balance has the smallest relative
  increase for all 14 target × direction pairs (+38 % on average vs
  +123–157 % for the ML variants), yet the ML variants start so much
  better in-site that the LightGBM baseline still has the lowest
  cross-site RMSE in 8 of 14 pairs (physics, hybrid and PINN 2 each).
  Mean cross-site RMSE KLM → PEN: baseline 4.90, hybrid 5.04, physics
  5.73, PINN 6.29 °C; PEN → KLM: all four within 4.66–4.85 °C.
- The thermal hybrid transfers like the ML baseline (its residual model
  dominates); the PINN transfers worst (its mean turbine embedding is a
  poor stand-in for an unseen turbine; gear oil KLM → PEN: 1.5 → 9.1 °C).
- Largest thermal losses for every variant are the main bearings
  (rear bearing: 2.1 → 7–9 °C): the two farms' main-bearing temperatures
  differ in level more than any input explains.

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
- New `src/data/static_data.py` reads the static file (drops empty rows,
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
    `metrics_entry`, `model_id_of`); `src/models/binned.py` holds the method of
    bins for reuse by the pitch model.
- Pitch and yaw (§7.8):
  - The yaw model is **kept**, although spec §7.8 says to drop it when
    wind direction is not independent of the nacelle position (it is not:
    wind direction = nacelle position + vane angle, see Results §7.4).
    The standard SCADA method needs exactly that relative vane angle: a
    vane offset shifts the power-vs-misalignment curve, so its peak gives
    the offset. The method works on Penmanshiel and fails on Kelmarsh,
    which the `identified` flag reports per turbine and per window
    (`yaw.edge_bins`, `yaw.min_peak_drop`).
  - The yaw fit uses `a·cos^k(γ − γ₀)` (spec: `P ∝ cos^k(γ)`): the gain
    `a` absorbs the level of the power curve and `γ₀` is the static
    misalignment. Its power reference is the turbine's own binned training
    power curve on v_n (5–10 m/s, below rated for both turbine types), not
    the power models, so the yaw step does not depend on them.
  - Yaw models are registered as `kt:YawModel_<FARM>_<T>_physics` with
    target `power` (their prediction is power under misalignment) and
    `kt:modelOf` the yaw system. `models/metrics.json` adds `offset_deg`,
    `exponent`, `peak_drop` and `identified`; the graph holds the standard
    metrics only. No `expected_power_physics` column is written for yaw
    (it would clash with the physics power model's column and covers only
    5–10 m/s).
  - Rolling windows (90 days, every 30) go to
    `data/processed/<farm>/yaw_offsets.parquet` (turbine_id,
    window_start, window_end, offset_deg, exponent, peak_drop, identified,
    n); only full windows with at least 8 usable bins are kept. Alerts on
    drift come in §7.10.
  - Pitch metrics are in degrees (nRMSE = RMSE, as for temperatures, scale
    1). Pitch predictions are NaN below rated wind speed (measured wind
    speed, as in cleaning); the pitch curves are fitted on v_n.
  - The pitch hybrid has no wind-direction features (they overfitted; see
    Results). `ResidualModel` (`src/models/residual.py`) is the shared
    physics-plus-LightGBM-residual class; `HybridPowerModel` (clips to
    [0, rated]) and `HybridPitchModel` build on it. It skips rows where
    the physics model is undefined (NaN).
  - `config.yaml` gained `yaw` and `models.pitch_physics`,
    `models.pitch_hybrid`, `models.yaw`.
- Thermal models (§7.9), decided with the user after testing the spec's
  method on real data (KLM T01, test RMSE):
  - **Extended heat balance:** the spec's `a·P + b·P² − h·(T − T_nac)`
    gained a standstill heat term `c` and rotor-speed cooling `h₁·ω`. Fitted
    one step ahead as specified, it gave negative `h` (unphysical) for the
    regulated temperatures and diverged when simulated (gear oil, main
    bearing).
  - **Fitted on free-run error, predicted as a free run:** the spec fits
    and predicts one step ahead from the measured T[k]. One-step RMSE
    equals persistence ("T stays the same", e.g. stator 1.35 vs 1.36 °C),
    so its residual follows the measurement and a component that slowly
    runs hotter never raises an alert. The free run predicts from the
    inputs only (scipy least squares replaced by PyTorch, which
    differentiates through the simulation).
  - **"Recent history" = rolling means of the inputs,** never lagged
    measured temperature, for the hybrid, baseline and PINN, for the same
    reason.
  - The hybrid's residual model also sees the physics prediction (without
    it the hybrid was worse than the baseline: stator 1.83 vs 1.62 °C).
  - PINN: the ODE residual is evaluated on every contiguous row pair
    (stops included, the heat balance holds there too); data loss on
    normal rows; h₀, h₁ positive through softplus, c, a, b free (some
    temperatures fall with power). λ = 0.01 (see Results). Rotor diameter
    is the turbine-type feature (both types have the same rated power).
    "Per-turbine data loaders" = every step draws the same number of rows
    (256) from each turbine.
  - All 7 component temperatures of §7.3 are modelled; the model IRI has
    the temperature as a qualifier, e.g.
    `kt:ThermalModel_KLM_T01_gear_oil_temp_hybrid` (the spec's
    `…_gearbox_hybrid` would collide for gear oil and gear-oil inlet).
    `graph_utils.model_iri`/`add_model` and `modelling.model_id_of` take an
    optional qualifier; power, pitch and yaw IRIs are unchanged.
  - The LightGBM-only variant is called `baseline`, as for power.
  - Physics and PINN files are shared by all turbines
    (`models/thermal/physics_<target>.json`, `pinn_<target>.pt`); each
    turbine's model node points to them.
  - `torch` added (approved in §5).
- Compute and robustness (added during §7.9): this workstation reset
  itself four times during thermal training runs on the GPU, the fourth
  during the §7.10 retrain. None was a clean shutdown; the monitored runs
  stayed at or below 62 °C and 154 W on the GPU, 80 °C on the CPU and
  10 of 60 GB of RAM, and other GPU runs completed, so the cause is
  unknown. One reset left a 0-byte turbine Parquet, rebuilt from the raw
  files. Since then:
  - `src/storage.write_parquet` writes every Parquet atomically
    (temporary file, fsync, rename);
  - `compute.threads` (8) caps LightGBM (`n_jobs` in every `lightgbm`
    block, via a YAML alias) and PyTorch; `make train` runs under
    `nice -n 10`;
  - the thermal step reads only the columns it needs, slices instead of
    copying, saves the graph, metrics and predictions after every
    temperature, and retrains selected temperatures with `--targets`;
  - the physics fit uses 4,096 random sequences per step and the PINN 256
    rows per turbine per step (lighter GPU load).
  - `ResidualModel` predicts the base per training frame, takes a row
    filter and can use the base prediction as a feature.
- Alerts and evaluation (§7.10), decided with the user:
  - **Calibration year:** `train_test_split` has `train_end`
    2020-12-31 23:50 (models fit 2016–2020) and a new `calibration_end`
    2021-12-31 23:50; thresholds come from 2021, test and evaluation are
    2022+. The spec derives thresholds from the training residuals, but the
    ML models fitted those rows: on PEN T08 gear oil the LightGBM threshold
    was 0.67 °C and it raised 57 test alerts against 7 in six training
    years. All models were retrained; `split_masks(frame, split, required)`
    and `split_times` take the whole `train_test_split` block. Penmanshiel
    pitch, gear-oil and main-bearing models now train on mid-2018 to 2020.
  - Thresholds are a quantile (99.5 %) of a problem score: the residual
    turned so that larger is worse (`alerts.direction` per model class:
    power low, temperature high, pitch either side).
  - A model's alerts less than 7 days apart are merged
    (`alerts.merge_gap_days`, the same gap as event episodes, chosen
    before looking at detection results): a sustained problem crosses the
    threshold many times (13,404 alerts before merging, 5,883 after).
  - Alerts are raised after `train_end` (calibration year and test), not
    on the fitted years; only test-period alerts are evaluated.
  - Fault events of one turbine and component less than 7 days apart are
    one episode (Penmanshiel repeats "Missing gear oil" stops), detection
    counts episodes, and lead time is that of the earliest alert within
    30 days. The summary adds a chance level per row (random alerts at the
    same rate) and a drift table (yearly mean score minus the calibration
    year's) for slow changes that event matching cannot credit.
  - An alert can fall within 30 days of two episodes; `kt:precedes` links
    it to the nearest one, with that lead time.
  - Alert IRIs carry the model id, e.g.
    `kt:Alert_ThermalModel_KLM_T01_gear_oil_temp_hybrid_001` (the spec's
    `kt:Alert_KLM_T01_gearbox_001` would collide between models). No new
    schema terms: alerts use `kt:raisedBy`, `kt:concerns`, `kt:startTime`,
    `kt:endTime`, `kt:precedes` and `kt:leadTimeHours`; peak scores and
    thresholds stay in `models/alerts.json`.
  - Power models concern the generator, so power alerts are matched to
    generator events only (spec: same component); Kelmarsh has no
    generator events in the test period, and no farm has main-bearing
    events, so those alerts can only be false alarms or case studies.
  - The committed graph now holds the alerts (2.8 MB instead of 1.5 MB).
- Improvements after §7.10 (decided with the user; all settings fixed
  before looking at results):
  - Thermal physics gained a cooler term `− h₂·max(T − T_set, 0)` with a
    learnable set point per turbine (start: the median training
    temperature); older saved parameters without it still load. The PINN's
    ODE keeps the spec's balance without it.
  - Scores: temperatures only at ≥ 500 kW (`alerts.min_power_kw`) and
    fleet-relative (`alerts.fleet_relative`, median of the other turbines
    of the farm, at least 3 of them). The scoring moved to
    `src/health/scores.py` and runs per farm.
  - Drift alerts (`alerts.drift`): ≥ 3 months with a mean score above the
    calibration year's by a fixed margin. They are written to
    `models/drift_alerts.json`, not the graph (the schema has one alert
    class; adding one needs a full ontology rebuild).
  - Evaluation adds a signal scope (`evaluation.signal_events`, checked by
    a test against `fault_labels`) next to the spec's component scope, and
    `src/health/curves.py` (`make curves`) evaluates six threshold
    quantiles with raw and fleet-relative scores.
  - Tried and rejected: a monotone constraint on ambient and nacelle
    temperature in the thermal LightGBM model (worse RMSE, see Results);
    the code was removed.
  - Not done: CPU-only thermal training and a GPU power limit (the user
    left these out).
- Cross-site experiment (§7.11):
  - Spec periods "train 2016–2021, test 2022–2024" are applied as in the
    in-site models: fit up to `train_end` (2020), test after
    `calibration_end` (2021), so in-site and cross-site RMSE use the same
    rows and the in-site predictions from `make train` can be reused.
  - One model per farm must serve unseen turbines: the thermal physics
    model is pooled (one parameter set, `pooled=True`), the PINN uses its
    mean turbine embedding (`unseen_turbines="mean"`); the in-site models
    keep per-turbine parameters and embeddings.
  - Lives in `src/experiments/` (not `src/models/`): it registers no
    models in the graph and only writes `models/cross_site.json`.
  - The first run was lost when the workstation reset after 9 minutes, so
    results are saved per signal and a rerun resumes.
- Report (§7.12):
  - `src/report/` is a package (`build.py` plus one module per report
    section: `coverage`, `comparison`, `detection`, `cases`, `style`)
    rather than the single `src/report.py` of spec §6; run as
    `uv run python -m src.report.build` (`make report`).
  - The script writes the figures and prints the tables; `docs/report.md`
    is written by hand around them (interpretation cannot be generated),
    so its numbers are a snapshot of the 2026-11-19 run.
  - Colours, case studies and figure settings live in `config.yaml`
    (`report`).
- `file_columns`, `read_signals` and `read_status` moved from
  `inspect_data.py` to `scada_format.py` so the pipeline does not import
  the inspection script.
- `inspect_data.py` also prints, per farm, how many files lack each
  configured column and the share of timestamps with a value per signal
  (first turbine file of each year), to decide decision 2 on real values
  rather than headers.
