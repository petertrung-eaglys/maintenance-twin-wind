"""Read a farm's static turbine file (specs, location, commissioning date)."""

from pathlib import Path

import pandas as pd

from src.scada_format import format_turbine_id

STATIC_COLUMNS = {
    "Title": "title",
    "Manufacturer": "manufacturer",
    "Model": "model",
    "Rated power (kW)": "rated_power_kw",
    "Hub Height (m)": "hub_height_m",
    "Rotor Diameter (m)": "rotor_diameter_m",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Elevation (m)": "elevation_m",
    "Commercial Operations Date": "commercial_operations_date",
}


def turbine_id_from_title(title: str) -> str:
    """Turbine id from a static title such as 'Kelmarsh 1' or 'Penmanshiel 01'."""
    return format_turbine_id(int(title.split()[-1]))


def read_static(path: Path) -> pd.DataFrame:
    """Static table indexed by turbine id, without empty rows or columns.

    Commissioning dates are day-first (15/04/2016) and taken as UTC midnight.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=["Title"])
    frame = raw.rename(columns=STATIC_COLUMNS)[list(STATIC_COLUMNS.values())]
    cod = pd.to_datetime(
        frame["commercial_operations_date"], format="%d/%m/%Y", utc=True
    )
    tidy = frame.assign(
        commercial_operations_date=cod,
        turbine_id=frame["title"].map(turbine_id_from_title),
    )
    return tidy.set_index("turbine_id")
