"""Tests for the static turbine file reader."""

from pathlib import Path

import pandas as pd

from src.data.static_data import read_mapping, read_static, turbine_id_from_title

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


def test_read_mapping_unifies_csv_and_xlsx_column_names(tmp_path: Path) -> None:
    csv_path, xlsx_path = tmp_path / "m.csv", tmp_path / "m.xlsx"
    csv_path.write_text(
        "﻿Greenbyte Signal ID,Greenbyte Title,Manufacturer Title,Unit\n"
        "5,Power,Active power,kWh\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "DataSignalId": [5],
            "TitleGreenbyte": ["Power"],
            "TitleManufacturer": ["Active power"],
            "Unit": ["kWh"],
            "MinValue": [0],
            "MaxValue": [3000],
        }
    ).to_excel(xlsx_path, index=False)

    expected = {
        "signal_id": [5],
        "greenbyte_title": ["Power"],
        "manufacturer_title": ["Active power"],
        "unit": ["kWh"],
    }
    assert [read_mapping(p).to_dict("list") for p in (csv_path, xlsx_path)] == [
        expected
    ] * 2


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
