"""Tests for the static turbine file reader."""

from pathlib import Path

import pandas as pd

from src.static_data import read_static, turbine_id_from_title

STATIC = (
    "﻿Wind Farm,Title,Alternative Title,Identity,Manufacturer,Model,"
    "Rated power (kW),Hub Height (m),Rotor Diameter (m),Latitude,Longitude,"
    "Elevation (m),Country,Commercial Operations Date,\n"
    "Pen,Penmanshiel 02,T02,X,Senvion,MM82,2050,59,82,55.9,-2.3,200.5,UK,"
    "01/09/2016,\n"
    "Pen,Penmanshiel 15,T15,Y,Senvion,MM82,2050,59,82,55.8,-2.2,228.1,UK,"
    "15/04/2016,\n"
    ",,,,,,,,,,,,,,\n"
)


def test_turbine_id_from_title_padded_or_not() -> None:
    assert [turbine_id_from_title(t) for t in ("Kelmarsh 1", "Penmanshiel 01")] == [
        "T01",
        "T01",
    ]


def test_read_static_drops_empty_rows_and_parses_dayfirst_dates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "static.csv"
    path.write_text(STATIC, encoding="utf-8")

    static = read_static(path)

    assert static["commercial_operations_date"].to_dict() == {
        "T02": pd.Timestamp("2016-09-01", tz="UTC"),
        "T15": pd.Timestamp("2016-04-15", tz="UTC"),
    }


def test_read_static_keeps_specs_under_short_names(tmp_path: Path) -> None:
    path = tmp_path / "static.csv"
    path.write_text(STATIC, encoding="utf-8")

    row = read_static(path).loc["T02"]

    assert (row["model"], row["rotor_diameter_m"], row["rated_power_kw"]) == (
        "MM82",
        82,
        2050,
    )
