# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project state

`docs/spec.md` is the authoritative specification. Work through its §7
milestones in order; §11 logs every milestone and change. `README.md` holds
results, limitations and "Deviations from spec". Read both before changing
behaviour. Done so far: §7.1 (scaffold), §7.2 (download), §7.3
(inspection; decisions in the README "Data inspection decisions"), §7.4
(load, events, clean), §7.5 (ontology), §7.6–7.7 (modelling conventions,
LightGBM power baseline), §7.8 (physics and hybrid power, pitch, yaw),
§7.9 (thermal models: physics, baseline, hybrid, PINN), §7.10 (alerts,
evaluation; then fleet-relative scores, drift alerts, curves), §7.11
(cross-site experiment), §7.12 (report), §7.13 (stretch: wake-aware
power).

## Package management: uv only

- Python 3.11 pinned; `uv add <pkg>` (`--dev` for dev tools), commit
  `uv.lock`, run everything with `uv run`. No pip, no requirements.txt.
- Only the packages in spec §5 are approved; ask before adding others.
  Add each dependency in the milestone that first needs it.

## Commands

```
make setup     # uv sync
make download  # fetch into data/raw/<farm>/ (MD5-verified, skips), unzip
make inspect   # raw structure, column coverage, events (~1 min)
make data      # load_scada -> events -> clean (~9 min, order matters)
make ontology  # rewrite ontology/wind_twin.ttl (~2 s; rerun later steps)
make train     # power, wake, pitch, yaw (~3 min), thermal (~15 min); nice'd
make wake      # wake-aware power alone (~40 s); then alerts/evaluate/report
make alerts    # residual alerts -> graph, models/alerts.json (~15 s)
make evaluate  # alerts vs fault events -> models/evaluation.json (~15 s)
make curves    # detection vs false alarms, 6 thresholds -> models/curves.json
make cross-site # train on one farm, test on the other (~30 min, resumes)
make report    # docs/figures/*.png + report tables printed (~20 s)
make test      # uv run pytest
make lint      # uv run ruff check .
```

Scripts run standalone as `uv run python -m src.<package>.<name>`; each
gets a Makefile target in its milestone. Thermal for selected temperatures
only: `uv run python -m src.models.thermal.train --targets <target> ...`.

Long runs on this workstation (it has reset itself during training): warn
the user first, launch detached (`setsid nohup ... > log 2>&1 &`) so they
survive a session disconnect, and keep `compute.threads` and `nice`.

## Module map

`src/` is grouped by concern: `data/` (raw files → clean Parquet),
`graph/` (the RDF graph), `models/` (normal-behaviour models), `health/`
(alerts and their evaluation), `experiments/` (studies that register
nothing in the graph), `report/` (figures and tables of docs/report.md). Tests stay
flat in `tests/` (`test_<module>.py`).

- `src/storage.py`: `write_parquet` (atomic: temp file, fsync, rename).
  Use it for every Parquet write; a reset once left a 0-byte file.
- `src/config.py`: `load_config`, `farm_names`, `farm_raw_dir`,
  `farm_processed_dir`, `scada_archives`, `scada_dirs` (per farm, shared
  folders listed once).
  `config.yaml` holds every path, URL, MD5, column name and threshold; no
  hard-coded values in scripts.

`src/data/`:

- `download.py`: `download_farm` per farm: MD5-checked `.part` streaming
  (skips valid files), unzip each archive into its `extract_to` folder
  unless all its members already exist there.
- `scada_format.py`: Greenbyte export layout: file discovery
  (`turbine_data_files`, `status_files`, `group_by_turbine`), header
  detection (data header = last `#` line, status header = first plain
  line), turbine ids `T01` from `<scada_name>_<N>_` (padded or not),
  readers `file_columns`, `read_signals`, `read_status`.
- `static_data.py`: `read_static` → static table indexed by turbine id
  (specs, location, UTC commissioning date); `read_mapping` → both farms'
  signal mappings with the same column names.
- `inspect_data.py`: §7.3 decision gate, printed output; testable helpers
  `filled_fraction`, `event_counts`, `read_table`.
- `load_scada.py` → `data/processed/<farm>/<T>.parquet` (timestamp,
  turbine_id, core + component signals). `events.py` →
  `data/processed/<farm>/events.parquet` (turbine_id, start, end, status,
  code, message, iec_category, source_file, component). `clean.py` adds
  `air_density`, `yaw_misalignment`, `is_normal` in place (needs
  events.parquet).

`src/graph/`:

- `graph_utils.py`: the only module that reads the graph. Farm-scoped IRI
  helpers (`turbine_iri("KLM", "T01")`, `component_iri`, `sensor_iri`,
  `fault_event_iri`, `source_iri`), SPARQL `*_QUERY` constants and `get_*`
  lookups binding `farm_code` and `turbine_id`. Model nodes: `add_model`
  (replaces), `get_models`, `get_model_inputs`; IRIs
  `kt:<Class>_<FARM>_<T>_<variant>`, or `kt:<Class>_<FARM>_<T>_<qualifier>_<variant>`
  with an optional qualifier (spec key `qualifier`; thermal: the target).
  Alerts: `add_alert`, `remove_alerts`, `get_alerts`, `add_precedes`,
  `clear_precedes`; IRIs `kt:Alert_<model id>_<NNN>`.
- `build_ontology.py`: `COMPONENTS` (component key → class) and
  `SIGNAL_SEMANTICS` (signal → component, observable, unit); builds the
  whole graph from `FarmSources` per farm (testable without disk).

`src/models/`:

- `modelling.py`: §7.6 conventions shared by every variant:
  `NormalBehaviourModel` (`fit(train_frames: Mapping[tid, DataFrame])`,
  `predict(frame)`), `split_masks`, `evaluate`, `prediction_column`
  (`expected_<target>_<variant>`), `split_times`, `turbine_paths`,
  `with_turbine_specs`,
  `register_model` (graph node + metrics record), `merge_metrics`.
- `binned.py`: method of bins (`binned_means`, `interpolate`).
- `residual.py`: `ResidualModel` (base model + LightGBM residual; base
  predicted per training frame, NaN rows skipped, optional row filter and
  `base_as_feature`), base of every hybrid model.
- `power_baseline.py`: §7.7 LightGBM baseline; `config.yaml`
  `models.power_baseline` holds inputs, features, target, constraints.
- `physics_power.py`: `PhysicsPowerModel` (Cp(v_n) per turbine type,
  aerodynamic bins only, uses each turbine's rotor area) and
  `HybridPowerModel` (physics + LightGBM residual per turbine).
- `wake.py` (§7.13): `WakeTable` (PyWake speed ratios per turbine, wd, ws;
  `lookup`), `wake_table` (Bastankhah–Porté-Agel, rotor averaging, k),
  `free_stream` (farm direction, mean speed of unwaked turbines),
  `turbine_views`, `variant_outputs` (`freestream`, `wake` PowerModels),
  `calibrate_k` (k per farm from training rows, `brentq`), `bias_records`
  → `models/wake.json`; `PhysicsPowerModel.from_dict` loads saved curves.
- `pitch.py`: `PhysicsPitchModel` (binned pitch(v_n) per turbine type, NaN
  below rated wind speed), `HybridPitchModel`; metrics in degrees.
- `yaw.py`: `YawModel` (per turbine, `a·cos^k(γ − γ₀)` of the power
  ratio), `fit_offset`, `peak_drop` (identified flag), `offset_windows` →
  `data/processed/<farm>/yaw_offsets.parquet`.
- `thermal/` (§7.9; frames keyed by `turbine_key` `KLM_T01`, models shared
  across both farms): `features.py` (rolling-mean history features,
  `normal_target_rows`, `target_spec`), `physics.py`
  (`ThermalPhysicsModel`: heat balance with a cooler above a learned set
  point, fitted on free-run error with PyTorch, all turbines batched;
  `simulate` free run with restarts, warm-up and cooler), `baseline.py` (`LightGBMThermalModel`), `pinn.py`
  (`ThermalPINN`, `ode_residual`), `train.py` (orchestration, per-target
  saving, `--targets`; `variant_specs` and `hybrid_model` shared with
  the cross-site experiment).

`src/health/` (§7.10):

- `scores.py`: `smoothed_residual` (optional `min_power`),
  `problem_score` (direction per model class), `fleet_relative` (minus the
  other turbines' median), `farm_scores` (every model of a farm, one
  turbine read at a time).
- `alerts.py`: `calibration_threshold`, `alert_periods`, `merge_periods`,
  `yearly_means` (drift KPI), `drift_periods` (drift alerts),
  `yaw_alert_periods`; writes the graph, `models/alerts.json` and
  `models/drift_alerts.json`.
- `evaluate.py`: `merge_episodes`, `match_alerts` (all pairs),
  `nearest_matches` (graph links), `chance_detection`, `summarise`
  (groups from the models, so models without alerts still appear),
  `evaluate_scopes` (component and signal scope), `drift_table`,
  `drift_cases`; writes `models/evaluation.json`.
- `curves.py`: `threshold_alerts` (alerts at one quantile), `lift`; writes
  `models/curves.json`.

`src/experiments/`:

- `cross_site.py` (§7.11): `compare` (in-site vs cross-site RMSE on the
  same rows), `summarise`, `power_experiment`, `thermal_experiment` (one
  target), `saved_or_run` (per-signal results in
  `models/cross_site/`; a rerun skips saved signals). Uses
  `ThermalPhysicsModel(pooled=True)` and
  `ThermalPINN(unseen_turbines="mean")`.

`src/report/` (§7.12; settings in `config.yaml` `report`):

- `build.py` (entry point: figures to `docs/figures/`, tables printed),
  `coverage.py` (normal-row share per month, signal availability),
  `comparison.py` (in-site means, cross-site panels/table), `detection.py`
  (detection vs chance), `cases.py` (fleet-relative case scores with
  events), `wake.py` (waked share of power-alert time), `style.py`
  (palette, `save`, `markdown_table`).
- `docs/report.md` is hand-written around the generated figures and
  tables; update its numbers when results change.

## Rules (spec §8)

- Run `inspect_data.py` before writing the loader; never guess column names.
  Record data deviations in the README.
- Tests use small synthetic data, never the network or real data.
- The RDF graph (from §7.5) is the source of structural truth, read only
  through `src/graph/graph_utils.py`.
- Per-turbine training interface: `fit(train_frames)` takes a mapping
  turbine id → DataFrame (federated-learning hook).
- `logging` for script output; `print` only for acceptance output.
- Never commit `data/` or `models/` (top level; `.gitignore` anchors them
  as `/data/`, `/models/` so the `src/data/` and `src/models/` packages are
  tracked). One commit per milestone, 50/70 format.

## Known data gotchas (from the Kelmarsh repo)

- Turbine files: 9 `#` metadata lines, then the header is itself a `#`
  line; CRLF; static CSV has a UTF-8 BOM. Status files have a plain header.
- Kelmarsh 2023/2024 exports repeat the year many times: keep the most
  complete row per timestamp, never "keep first".
- Kelmarsh mapping lists `Power` in kWh; `Power (kW)` is 10-minute mean power.
- Penmanshiel file names zero-pad turbine numbers (`Penmanshiel_01`);
  Kelmarsh does not (`Kelmarsh_1`). Penmanshiel 2023 WT01–10 has four
  period files per turbine; 2024 files are ~3.8 GB each. Penmanshiel
  repeats timestamps too (2023 and 2024), so "most complete row" applies.
- Penmanshiel has no pitch, main-bearing or gear-oil values before
  mid-2018; data starts 2016-06-06, before COD 2016-09-01.
- `Penmanshiel_WT_static.csv` has 15 empty trailing rows and an empty
  column. Mapping tables have different column names per farm.
- Status headers end at `IEC category` (no contract-category columns).
  Status exports repeat events: de-duplicate on turbine, start, status,
  code, message.
- `fault_labels` are exact messages per component; there are no
  main-bearing fault messages.
- Wind direction = nacelle position + vane angle (not independent), so
  `yaw_misalignment` is the vane's relative wind angle.
- SPARQL UNION branches are evaluated bottom-up: a branch cannot use a
  variable bound outside it (`BIND(?outer AS ?s)` matches everything).
  Repeat the turbine pattern inside each branch (see `LINEAGE_QUERY`).
- `ontology/wind_twin.ttl` is committed; tests use the synthetic graph in
  `tests/conftest.py`, never the real file. It changes on every
  `make train` (new `prov:generatedAtTime`); commit it when its structure
  changed.
- LightGBM's `LGBMRegressor` needs scikit-learn installed.
- Physics Cp curves must stop at the first power-limited bin: above
  rated, Cp falls only because the controller caps power, which does not
  transfer between rotor sizes; storm control lowers mean power again at
  high wind, so filtering bins one by one is not enough.
- Penmanshiel T08 underperforms from 2021 to mid-2022 (partly in the
  training period): expected, not a bug. Its yaw offset jumps from about
  0° to 10° in mid-2022.
- Yaw offsets are identifiable on Penmanshiel only: Kelmarsh's power
  ratio rises across the whole ±25° misalignment range (no peak), so its
  fits are flagged `identified = False`. Don't "fix" this by widening
  the bounds.
- Wind-direction features overfit the pitch hybrid; keep them out.
- Thermal: never use the measured temperature (lags, one-step-ahead) as
  a model input; residuals must reveal slow overheating. Most of these
  temperatures are regulated by coolers (some fall as power rises), so the
  physics model stays 1–3 °C worse than the ML variants; that is expected.
  PINN λ above ~0.1 costs accuracy (the lumped balance is only ~5 °C good).
- Split: models fit up to `train_end` (2020), alert thresholds come from
  the calibration year up to `calibration_end` (2021), test is after it.
  Never set thresholds on rows a model was fitted on (ML residuals there
  are far too small).
- Alerts: event-timed detection is at chance level for every model; the
  ML thermal alerts mostly follow slow drift (some real, e.g. KLM T04 gear
  oil +4–6 °C at power since 2023; some model extrapolation in warm
  years). Don't tune thresholds or windows on test detection results.
  Even with fair curves and fleet-relative scores the lift stays near 0;
  gearbox drift alerts flag the turbines with the most gearbox events
  (p = 0.074), generator drift alerts are not specific.
- Thermal LightGBM: a monotone constraint on ambient/nacelle temperature
  was tried and removed (worse: nacelle temperature falls at high power).
- Cross-site: power physics/hybrid transfer far better than LightGBM;
  thermal physics degrades least but LightGBM stays most accurate on the
  other farm in 8 of 14 cases. Main bearings transfer worst (level
  differs between farms).
- Penmanshiel T01–T10 have no January 2023 (2023 files start 1 Feb).
  PEN T15 has −273.2 °C placeholders (18–21 Apr 2024, non-normal rows).
- PEN T15's 2024 gear-oil inlet jump is inlet-only (sump unchanged):
  report it as cooler/sensor work, not overheating.
- Wakes: never multiply a wake loss into the nacelle-wind physics model
  (its wind is already waked). On Kelmarsh waked rows even over-perform
  vs that model (anemometer reads low in wakes). PyWake default k = 0.032
  with hub-centre deficits is far too deep here; k is fitted per farm on
  training rows (KLM ~0.04, PEN ~0.12).
- Define functions above `if __name__ == "__main__":`; ruff does not flag
  a function appended below it (NameError at run time).
- Thermal frames must stay sorted by timestamp and contiguous (the free
  run restarts at gaps); slice the training period, don't mask it.
- pandas 3: string columns keep NaN (not None) for missing values; parquet
  timestamps may not be ns, so convert with `.to_numpy("datetime64[ns]")`
  before numpy comparisons.
