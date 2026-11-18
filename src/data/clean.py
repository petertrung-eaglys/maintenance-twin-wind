"""Add air_density, yaw_misalignment and is_normal to each turbine Parquet.

The share of normal rows is printed, as spec §7.4 asks. Run after
load_scada.py and events.py.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import farm_names, farm_processed_dir, farm_raw_dir, load_config
from src.data.static_data import read_static
from src.storage import write_parquet

logger = logging.getLogger(__name__)

DRY_AIR_GAS_CONSTANT = 287.05  # J/(kg K)
KELVIN_OFFSET = 273.15
NS = "datetime64[ns]"


def air_density(temp_c: pd.Series, pressure_pa: float) -> pd.Series:
    """Air density in kg/m3 from temperature via the ideal gas law."""
    return pressure_pa / (DRY_AIR_GAS_CONSTANT * (temp_c + KELVIN_OFFSET))


def yaw_misalignment(wind_dir: pd.Series, nacelle_dir: pd.Series) -> pd.Series:
    """Wind direction minus nacelle position, wrapped to [-180, 180) degrees."""
    return (wind_dir - nacelle_dir + 180.0) % 360.0 - 180.0


def stop_active(
    timestamps: pd.Series, stops: pd.DataFrame, interval: pd.Timedelta
) -> np.ndarray:
    """True for rows whose interval [t, t + interval) meets a stop event.

    `timestamps` must be sorted; `stops` has UTC `start` and `end` (a missing
    end counts as an instant at `start`).
    """
    times = timestamps.to_numpy(NS)
    starts = stops["start"].to_numpy(NS)
    ends = stops["end"].fillna(stops["start"]).to_numpy(NS)
    valid = ends >= starts
    first = np.searchsorted(times, starts[valid] - interval.to_numpy(), "right")
    last = np.searchsorted(times, ends[valid], "right")
    delta = np.zeros(len(times) + 1, dtype=int)
    np.add.at(delta, first, 1)
    np.add.at(delta, last, -1)
    return np.cumsum(delta)[:-1] > 0


def normal_mask(
    frame: pd.DataFrame, cleaning: dict[str, Any], rated_wind_speed: float
) -> pd.Series:
    """True for rows that pass the operating-range rules.

    Below rated wind speed the pitch must be small (no curtailment); a
    missing pitch is accepted, as Penmanshiel lacks pitch before mid-2018.
    """
    complete = frame[cleaning["required"]].notna().all(axis=1)
    wind_ok = frame["wind_speed"].between(
        cleaning["min_wind_speed"], cleaning["max_wind_speed"]
    )
    power_ok = frame["power"] > cleaning["min_power_kw"]
    pitch_ok = (
        (frame["wind_speed"] >= rated_wind_speed)
        | (frame["pitch"] < cleaning["max_pitch_deg"])
        | frame["pitch"].isna()
    )
    return complete & wind_ok & power_ok & pitch_ok


def add_cleaning_columns(
    frame: pd.DataFrame,
    cleaning: dict[str, Any],
    rated_wind_speed: float,
    commissioned: pd.Timestamp,
    stopped: np.ndarray,
) -> pd.DataFrame:
    """Return the frame with air_density, yaw_misalignment and is_normal."""
    operating = normal_mask(frame, cleaning, rated_wind_speed)
    in_service = frame["timestamp"] >= commissioned
    pressure = cleaning["standard_pressure_pa"]
    return frame.assign(
        air_density=air_density(frame["ambient_temp"], pressure),
        yaw_misalignment=yaw_misalignment(
            frame["wind_direction"], frame["nacelle_direction"]
        ),
        is_normal=operating & in_service & ~stopped,
    )


def clean_farm(config: dict[str, Any], farm: str) -> None:
    """Clean every turbine Parquet of one farm in place."""
    farm_config = config["farms"][farm]
    static = read_static(farm_raw_dir(config, farm) / farm_config["static_file"])
    processed = farm_processed_dir(config, farm)
    events = pd.read_parquet(processed / "events.parquet")
    stops = events[events["status"] == config["events"]["stop_status"]]
    interval = pd.Timedelta(minutes=config["sampling"]["interval_minutes"])
    cleaning = config["cleaning"]
    for turbine_id, spec in static.iterrows():
        path = processed / f"{turbine_id}.parquet"
        frame = pd.read_parquet(path)
        stopped = stop_active(
            frame["timestamp"], stops[stops["turbine_id"] == turbine_id], interval
        )
        cleaned = add_cleaning_columns(
            frame,
            cleaning,
            cleaning["rated_wind_speed"][spec["model"]],
            spec["commercial_operations_date"],
            stopped,
        )
        write_parquet(cleaned, path)
        logger.info("Cleaned %s", path)
        print(
            f"{farm} {turbine_id}: {len(cleaned)} rows, "
            f"{cleaned['is_normal'].mean():.1%} normal, "
            f"{stopped.mean():.1%} in a stop event"
        )


def main() -> None:
    """Clean every farm's turbines."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    for farm in farm_names(config):
        clean_farm(config, farm)


if __name__ == "__main__":
    main()
