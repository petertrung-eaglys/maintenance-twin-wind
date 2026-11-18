"""Load each farm's raw SCADA files into one Parquet per turbine.

Row counts and missing percentages are printed, as spec §7.4 asks.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import farm_names, farm_processed_dir, load_config, scada_dirs
from src.scada_format import group_by_turbine, read_signals, turbine_data_files

logger = logging.getLogger(__name__)


def all_signals(config: dict[str, Any]) -> dict[str, str]:
    """Core and component signals: config key -> SCADA column name."""
    return config["signals"] | config["component_signals"]


def tidy(
    raw: pd.DataFrame,
    timestamp_column: str,
    signals: dict[str, str],
    turbine_id: str,
    years: list[int],
) -> pd.DataFrame:
    """Rename to config keys, parse UTC timestamps, keep years, dedupe.

    Duplicate timestamps (year-file boundaries, repeated rows in the 2023 and
    2024 exports) keep the row with the most non-missing signals; ties keep
    the first row in file order.
    """
    renamed = raw.rename(columns={v: k for k, v in signals.items()})
    timestamp = pd.to_datetime(raw[timestamp_column], utc=True)
    frame = renamed.assign(timestamp=timestamp, turbine_id=turbine_id)[
        ["timestamp", "turbine_id", *signals]
    ]
    in_years = frame[frame["timestamp"].dt.year.isin(years)]
    present = in_years[list(signals)].notna().sum(axis=1).rename("present")
    return (
        in_years.join(present)
        .sort_values(["timestamp", "present"], ascending=[True, False], kind="stable")
        .drop_duplicates(subset="timestamp")
        .drop(columns="present")
        .reset_index(drop=True)
    )


def missing_percent(frame: pd.DataFrame) -> pd.Series:
    """Percentage of missing values per column, rounded to 0.1."""
    return (frame.isna().mean() * 100).round(1)


def rows_per_year(frame: pd.DataFrame) -> dict[int, int]:
    """Number of rows per calendar year of the timestamp."""
    return frame["timestamp"].dt.year.value_counts().sort_index().to_dict()


def load_turbine(
    turbine_id: str, paths: list[Path], config: dict[str, Any]
) -> pd.DataFrame:
    """Read, concatenate in the given order and tidy all files of one turbine."""
    timestamp_column = config["scada"]["timestamp_column"]
    signals = all_signals(config)
    wanted = [timestamp_column, *signals.values()]
    raw = pd.concat([read_signals(p, wanted) for p in paths], ignore_index=True)
    return tidy(raw, timestamp_column, signals, turbine_id, config["years"])


def load_farm(config: dict[str, Any], farm: str) -> None:
    """Write <processed>/<farm>/<turbine_id>.parquet for every turbine."""
    out_dir = farm_processed_dir(config, farm)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = turbine_data_files(scada_dirs(config, farm))
    for turbine_id, files in group_by_turbine(
        paths, config["farms"][farm]["scada_name"]
    ).items():
        logger.info("Loading %d files for %s %s", len(files), farm, turbine_id)
        frame = load_turbine(turbine_id, files, config)
        frame.to_parquet(out_dir / f"{turbine_id}.parquet", index=False)
        missing = missing_percent(frame[list(all_signals(config))])
        print(f"\n{farm} {turbine_id}: {len(frame)} rows")
        print(f"  rows per year: {rows_per_year(frame)}")
        print(f"  missing %: {missing.to_dict()}")


def main() -> None:
    """Load every farm's turbines."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    for farm in farm_names(config):
        load_farm(config, farm)


if __name__ == "__main__":
    main()
