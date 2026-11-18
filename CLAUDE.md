# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project state

`docs/spec.md` is the authoritative specification. Work through its §7
milestones in order; §11 logs every milestone and change. `README.md` holds
results, limitations and "Deviations from spec". Read both before changing
behaviour. Done so far: §7.1 (scaffold), §7.2 (download), §7.3
(inspection; decisions in the README "Data inspection decisions"), §7.4
(load, events, clean), §7.5 (ontology), §7.6–7.7 (modelling conventions,
LightGBM power baseline), §7.8 (physics and hybrid power, pitch, yaw).

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
make train     # all model variants (so far: power, pitch, yaw)
make test      # uv run pytest
make lint      # uv run ruff check .
```

Scripts run standalone as `uv run python -m src.<name>`; each gets a
Makefile target in its milestone.

## Module map

- `src/config.py`: `load_config`, `farm_names`, `farm_raw_dir`,
  `farm_processed_dir`, `scada_archives`, `scada_dirs` (per farm, shared
  folders listed once).
  `config.yaml` holds every path, URL, MD5, column name and threshold; no
  hard-coded values in scripts.
- `src/download.py`: `download_farm` per farm: MD5-checked `.part`
  streaming (skips valid files), unzip each archive into its `extract_to`
  folder unless all its members already exist there.
- `src/scada_format.py`: Greenbyte export layout: file discovery
  (`turbine_data_files`, `status_files`, `group_by_turbine`), header
  detection (data header = last `#` line, status header = first plain
  line), turbine ids `T01` from `<scada_name>_<N>_` (padded or not),
  readers `file_columns`, `read_signals`, `read_status`.
- `src/static_data.py`: `read_static` → static table indexed by turbine
  id (specs, location, UTC commissioning date); `read_mapping` → both
  farms' signal mappings with the same column names.
- `src/graph_utils.py`: the only module that reads the graph. Farm-scoped
  IRI helpers (`turbine_iri("KLM", "T01")`, `component_iri`,
  `sensor_iri`, `fault_event_iri`, `source_iri`), SPARQL `*_QUERY`
  constants and `get_*` lookups binding `farm_code` and `turbine_id`.
- `src/modelling.py`: §7.6 conventions shared by every variant:
  `NormalBehaviourModel` (`fit(train_frames: Mapping[tid, DataFrame])`,
  `predict(frame)`), `split_masks`, `evaluate`, `prediction_column`
  (`expected_<target>_<variant>`), `turbine_paths`, `with_turbine_specs`,
  `register_model` (graph node + metrics record), `merge_metrics`.
- `src/binned.py`: method of bins (`binned_means`, `interpolate`).
- `src/physics_power.py`: `PhysicsPowerModel` (Cp(v_n) per turbine type,
  aerodynamic bins only, uses each turbine's rotor area) and
  `HybridPowerModel` (physics + LightGBM residual per turbine).
- `src/residual.py`: `ResidualModel` (base model + LightGBM residual,
  skips rows where the base is NaN), base of both hybrid models.
- `src/pitch.py`: `PhysicsPitchModel` (binned pitch(v_n) per turbine type,
  NaN below rated wind speed), `HybridPitchModel`; metrics in degrees.
- `src/yaw.py`: `YawModel` (per turbine, `a·cos^k(γ − γ₀)` of the power
  ratio), `fit_offset`, `peak_drop` (identified flag), `offset_windows` →
  `data/processed/<farm>/yaw_offsets.parquet`.
- `src/power_baseline.py`: §7.7 LightGBM baseline; `config.yaml`
  `models.power_baseline` holds inputs, features, target, constraints.
  Model nodes: `graph_utils.add_model` (replaces), `get_models`,
  `get_model_inputs`; IRIs `kt:<Class>_<FARM>_<T>_<variant>`.
- `src/build_ontology.py`: `COMPONENTS` (component key → class) and
  `SIGNAL_SEMANTICS` (signal → component, observable, unit); builds the
  whole graph from `FarmSources` per farm (testable without disk).
- `src/inspect_data.py`: §7.3 decision gate, printed output; testable
  helpers `filled_fraction`, `event_counts`, `read_table`.
- `src/load_scada.py` → `data/processed/<farm>/<T>.parquet` (timestamp,
  turbine_id, core + component signals). `src/events.py` →
  `data/processed/<farm>/events.parquet` (turbine_id, start, end, status,
  code, message, iec_category, source_file, component). `src/clean.py` adds
  `air_density`, `yaw_misalignment`, `is_normal` in place (needs
  events.parquet).

## Rules (spec §8)

- Run `inspect_data.py` before writing the loader; never guess column names.
  Record data deviations in the README.
- Tests use small synthetic data, never the network or real data.
- The RDF graph (from §7.5) is the source of structural truth, read only
  through `src/graph_utils.py`.
- Per-turbine training interface: `fit(train_frames)` takes a mapping
  turbine id → DataFrame (federated-learning hook).
- `logging` for script output; `print` only for acceptance output.
- Never commit `data/` or `models/`. One commit per milestone, 50/70 format.

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
- pandas 3: string columns keep NaN (not None) for missing values; parquet
  timestamps may not be ns, so convert with `.to_numpy("datetime64[ns]")`
  before numpy comparisons.
