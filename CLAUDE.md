# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project state

`docs/spec.md` is the authoritative specification. Work through its §7
milestones in order; §11 logs every milestone and change. `README.md` holds
results, limitations and "Deviations from spec". Read both before changing
behaviour. Done so far: §7.1 (scaffold), §7.2 (download).

## Package management: uv only

- Python 3.11 pinned; `uv add <pkg>` (`--dev` for dev tools), commit
  `uv.lock`, run everything with `uv run`. No pip, no requirements.txt.
- Only the packages in spec §5 are approved; ask before adding others.
  Add each dependency in the milestone that first needs it.

## Commands

```
make setup     # uv sync
make download  # fetch into data/raw/<farm>/ (MD5-verified, skips), unzip
make test      # uv run pytest
make lint      # uv run ruff check .
```

Scripts run standalone as `uv run python -m src.<name>`; each gets a
Makefile target in its milestone.

## Module map

- `src/config.py`: `load_config`, `farm_names`, `farm_raw_dir`,
  `scada_archives`, `scada_dirs` (per farm, shared folders listed once).
  `config.yaml` holds every path, URL, MD5, column name and threshold; no
  hard-coded values in scripts.
- `src/download.py`: `download_farm` per farm: MD5-checked `.part`
  streaming (skips valid files), unzip each archive into its `extract_to`
  folder unless all its members already exist there.

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
  period files per turbine; 2024 files are ~3.8 GB each.
