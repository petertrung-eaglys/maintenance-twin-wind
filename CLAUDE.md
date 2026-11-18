# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project state

`docs/spec.md` is the authoritative specification. Work through its §7
milestones in order; §11 logs every milestone and change. `README.md` holds
results, limitations and "Deviations from spec". Read both before changing
behaviour. Done so far: §7.1 (scaffold), §7.2 (download), §7.3
(inspection; decisions in the README "Data inspection decisions"), §7.4
(load, events, clean).

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
  id (specs, location, UTC commissioning date).
- `src/inspect_data.py`: §7.3 decision gate, printed output; testable
  helpers `filled_fraction`, `event_counts`, `read_table`.
- `src/load_scada.py` → `data/processed/<farm>/<T>.parquet` (timestamp,
  turbine_id, core + component signals). `src/events.py` →
  `data/processed/<farm>/events.parquet` (turbine_id, start, end, status,
  code, message, iec_category, component). `src/clean.py` adds
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
- pandas 3: string columns keep NaN (not None) for missing values; parquet
  timestamps may not be ns, so convert with `.to_numpy("datetime64[ns]")`
  before numpy comparisons.
